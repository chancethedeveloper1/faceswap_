#!/usr/bin/env python3


""" Base Trainer Class for Faceswap

    Trainers should be inherited from this class.

    A training_opts dictionary can be set in the corresponding model.
    Accepted values:
        alignments:         dict containing paths to alignments files for keys 'a' and 'b'
        preview_scaling:    How much to scale the preview out by
        training_size:      Size of the training images
        coverage_ratio:     Ratio of face to be cropped out for training
        mask_type:          Type of mask to use. See lib.model.masks for valid mask names.
                            Set to None for not used
        no_logs:            Disable tensorboard logging
        snapshot_interval:  Interval for saving model snapshots
        warp_to_landmarks:  Use random_warp_landmarks instead of random_warp
        augment_color:      Perform random shifting of L*a*b* colors
        no_flip:            Don't perform a random flip on the image
        pingpong:           Train each side seperately per save iteration rather than together
"""

import logging
import os
import time

import cv2
import numpy as np

import tensorflow as tf
from tensorflow.python import errors_impl as tf_errors  # pylint:disable=no-name-in-module

from lib.alignments import Alignments
from lib.faces_detect import DetectedFace
from lib.training_data import TrainingDataGenerator
from lib.utils import FaceswapError, get_folder, get_image_paths
from lib.image import read_image
from plugins.train._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_config(plugin_name, configfile=None):
    """ Return the config for the requested model """
    return Config(plugin_name, configfile=configfile).config_dict


class TrainerBase():
    """ Base Trainer """

    def __init__(self, model, images, batch_size, configfile):
        logger.debug("Initializing %s: (model: '%s', batch_size: %s)",
                     self.__class__.__name__, model, batch_size)
        self.config = get_config(".".join(self.__module__.split(".")[-2:]), configfile=configfile)
        self.batch_size = batch_size
        self.model = model
        self.model.state.add_session_batchsize(batch_size)
        self.images = images
        self.sides = sorted(key for key in self.images.keys())
        self.frame_size = read_image(self.images['a'][0]).shape[0]
        self.process_training_opts()
        self.pingpong = PingPong(model, self.sides)

        self.batchers = {side: Batcher(side,
                                       images[side],
                                       self.model,
                                       self.use_mask,
                                       batch_size,
                                       self.config)
                         for side in self.sides}

        self.tensorboard = self.set_tensorboard()
        self.samples = Samples(self.model,
                               self.use_mask,
                               self.frame_size,
                               self.model.training_opts["coverage_ratio"],
                               self.model.training_opts["preview_scaling"])
        self.timelapse = Timelapse(self.model,
                                   self.use_mask,
                                   self.frame_size,
                                   self.model.training_opts["coverage_ratio"],
                                   self.config.get("preview_images", 14),
                                   self.batchers)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def timestamp(self):
        """ Standardised timestamp for loss reporting """
        return time.strftime("%H:%M:%S")

    @property
    def landmarks_required(self):
        """ Return True if Landmarks are required """
        opts = self.model.training_opts
        retval = bool(opts.get("mask_type", None) or opts["warp_to_landmarks"])
        logger.debug(retval)
        return retval

    @property
    def use_mask(self):
        """ Return True if a mask is requested """
        retval = bool(self.model.training_opts.get("mask_type", None))
        logger.debug(retval)
        return retval

    def process_training_opts(self):
        """ Override for processing model specific training options """
        logger.debug(self.model.training_opts)
        if self.landmarks_required:
            landmarks = Landmarks(self.model.training_opts).landmarks
            self.model.training_opts["landmarks"] = landmarks

    def set_tensorboard(self):
        """ Set up tensorboard callback """
        if self.model.training_opts["no_logs"]:
            logger.verbose("TensorBoard logging disabled")
            return None
        if self.pingpong.active:
            # Currently TensorBoard uses the tf.session, meaning that VRAM does not
            # get cleared when model switching
            # TODO find a fix for this
            logger.warning("Currently TensorBoard logging is not supported for Ping-Pong "
                           "training. Session stats and graphing will not be available for this "
                           "training session.")
            return None

        logger.debug("Enabling TensorBoard Logging")
        tensorboard = dict()

        for side in self.sides:
            logger.debug("Setting up TensorBoard Logging. Side: %s", side)
            log_dir = os.path.join(str(self.model.model_dir),
                                   "{}_logs".format(self.model.name),
                                   side,
                                   "session_{}".format(self.model.state.session_id))
            tbs = tf.keras.callbacks.TensorBoard(log_dir=log_dir, **self.tensorboard_kwargs)
            tbs.set_model(self.model.predictors[side])
            tensorboard[side] = tbs
        logger.info("Enabled TensorBoard Logging")
        return tensorboard

    @property
    def tensorboard_kwargs(self):
        """ TF 1.13 + needs an additional kwarg which is not valid for earlier versions """
        kwargs = dict(histogram_freq=0,  # Must be 0 or hangs
                      batch_size=64,
                      write_graph=True,
                      write_grads=True)
        tf_version = [int(ver) for ver in tf.__version__.split(".") if ver.isdigit()]
        logger.debug("Tensorflow version: %s", tf_version)
        if tf_version[0] > 1 or (tf_version[0] == 1 and tf_version[1] > 12):
            kwargs["update_freq"] = "batch"
        if tf_version[0] > 1 or (tf_version[0] == 1 and tf_version[1] > 13):
            kwargs["profile_batch"] = 0
        logger.debug(kwargs)
        return kwargs

    def print_loss(self, loss):
        """ Override for specific model loss formatting """
        logger.trace(loss)
        output = ["Loss {}: {:.5f}".format(side.capitalize(), loss[side][0])
                  for side in sorted(loss.keys())]
        output = ", ".join(output)
        print("[{}] [#{:05d}] {}".format(self.timestamp, self.model.iterations, output), end='\r')

    def train_one_step(self, viewer, timelapse_kwargs):
        """ Train a batch """
        logger.trace("Training one step: (iteration: %s)", self.model.iterations)
        do_preview = viewer is not None
        do_timelapse = timelapse_kwargs is not None
        snapshot_interval = self.model.training_opts.get("snapshot_interval", 0)
        do_snapshot = (snapshot_interval != 0 and
                       self.model.iterations >= snapshot_interval and
                       self.model.iterations % snapshot_interval == 0)

        loss = dict()
        try:
            for side, batcher in self.batchers.items():
                if self.pingpong.active and side != self.pingpong.side:
                    continue
                loss[side] = batcher.train_one_batch(do_preview)
                if not do_preview and not do_timelapse:
                    continue
                if do_preview:
                    self.samples.images[side] = batcher.compile_sample(None)
                if do_timelapse:
                    self.timelapse.get_sample(side, timelapse_kwargs)

            self.model.state.increment_iterations()

            for side, side_loss in loss.items():
                self.store_history(side, side_loss)
                self.log_tensorboard(side, side_loss)

            if not self.pingpong.active:
                self.print_loss(loss)
            else:
                for key, val in loss.items():
                    self.pingpong.loss[key] = val
                self.print_loss(self.pingpong.loss)

            if do_preview:
                preview = self.samples.create_preview_window()
                if preview is not None:
                    viewer(preview, "Training - 'S': Save Now. 'ENTER': Save and Quit")

            if do_timelapse:
                self.timelapse.output_timelapse()

            if do_snapshot:
                self.model.do_snapshot()
        except Exception as err:
            raise err

    def store_history(self, side, loss):
        """ Store the history of this step """
        logger.trace("Updating loss history: '%s'", side)
        self.model.history[side].append(loss[0])  # Either only loss or total loss
        logger.trace("Updated loss history: '%s'", side)

    def log_tensorboard(self, side, loss):
        """ Log loss to TensorBoard log """
        if not self.tensorboard:
            return
        logger.trace("Updating TensorBoard log: '%s'", side)
        logs = {log[0]: log[1]
                for log in zip(self.model.state.loss_names[side], loss)}
        self.tensorboard[side].on_batch_end(self.model.state.iterations, logs)
        logger.trace("Updated TensorBoard log: '%s'", side)

    def clear_tensorboard(self):
        """ Indicate training end to Tensorboard """
        if not self.tensorboard:
            return
        for side, tensorboard in self.tensorboard.items():
            logger.debug("Ending Tensorboard. Side: '%s'", side)
            tensorboard.on_train_end(None)


class Batcher():
    """ Batch images from a single side """
    def __init__(self, side, images, model, use_mask, batch_size, config):
        logger.debug("Initializing %s: side: '%s', num_images: %s, batch_size: %s, config: %s)",
                     self.__class__.__name__, side, len(images), batch_size, config)
        self.model = model
        self.use_mask = use_mask
        self.side = side
        self.images = images
        self.config = config
        self.target = None
        self.samples = None
        self.mask = None

        generator = self.load_generator()
        self.feed = generator.minibatch_ab(images, batch_size, self.side)

        self.preview_feed = None
        self.timelapse_feed = None

    def load_generator(self):
        """ Pass arguments to TrainingDataGenerator and return object """
        logger.debug("Loading generator: %s", self.side)
        input_size = self.model.input_shape[0]
        output_shapes = self.model.output_shapes
        logger.debug("input_size: %s, output_shapes: %s", input_size, output_shapes)
        generator = TrainingDataGenerator(input_size,
                                          output_shapes,
                                          self.model.training_opts,
                                          self.config)
        return generator

    def train_one_batch(self, do_preview):
        """ Train a batch """
        logger.trace("Training one step: (side: %s)", self.side)
        batch = self.get_next(do_preview)
        try:
            loss = self.model.predictors[self.side].train_on_batch(*batch)
        except tf_errors.ResourceExhaustedError as err:
            msg = ("You do not have enough GPU memory available to train the selected model at "
                   "the selected settings. You can try a number of things:"
                   "\n1) Close any other application that is using your GPU (web browsers are "
                   "particularly bad for this)."
                   "\n2) Lower the batchsize (the amount of images fed into the model each "
                   "iteration)."
                   "\n3) Try 'Memory Saving Gradients' and/or 'Optimizer Savings' and/or 'Ping "
                   "Pong Training'."
                   "\n4) Use a more lightweight model, or select the model's 'LowMem' option "
                   "(in config) if it has one.")
            raise FaceswapError(msg) from err
        loss = loss if isinstance(loss, list) else [loss]
        return loss

    def get_next(self, do_preview):
        """ Return the next batch from the generator
            Items should come out as: (warped, target [, mask]) """
        batch = next(self.feed)
        if self.use_mask:
            batch = [[batch["feed"], batch["masks"]], batch["targets"] + [batch["masks"]]]
        else:
            batch = [batch["feed"], batch["targets"]]
        self.generate_preview(do_preview)
        return batch

    def generate_preview(self, do_preview):
        """ Generate the preview if a preview iteration """
        if not do_preview:
            self.samples = None
            self.target = None
            return
        logger.debug("Generating preview")
        if self.preview_feed is None:
            self.set_preview_feed()
        batch = next(self.preview_feed)
        self.samples = batch["samples"]
        self.target = [batch["targets"][self.model.largest_face_index]]
        if self.use_mask:
            self.target += [batch["masks"]]

    def set_preview_feed(self):
        """ Set the preview dictionary """
        logger.debug("Setting preview feed: (side: '%s')", self.side)
        preview_images = self.config.get("preview_images", 14)
        preview_images = min(max(preview_images, 2), 16)
        batchsize = min(len(self.images), preview_images)
        self.preview_feed = self.load_generator().minibatch_ab(self.images,
                                                               batchsize,
                                                               self.side,
                                                               do_shuffle=True,
                                                               is_preview=True)
        logger.debug("Set preview feed. Batchsize: %s", batchsize)

    def compile_sample(self, batch_size, samples=None, images=None):
        """ Training samples to display in the viewer """
        num_images = self.config.get("preview_images", 14)
        num_images = min(batch_size, num_images) if batch_size is not None else num_images
        logger.debug("Compiling samples: (side: '%s', samples: %s)", self.side, num_images)
        images = images if images is not None else self.target
        retval = [samples[0:num_images]] if samples is not None else [self.samples[0:num_images]]
        if self.use_mask:
            retval.extend(tgt[0:num_images] for tgt in images)
        else:
            retval.extend(images[0:num_images])
        return retval

    def compile_timelapse_sample(self):
        """ Timelapse samples """
        batch = next(self.timelapse_feed)
        batchsize = len(batch["samples"])
        images = [batch["targets"][self.model.largest_face_index]]
        if self.use_mask:
            images = images + [batch["masks"]]
        sample = self.compile_sample(batchsize, samples=batch["samples"], images=images)
        return sample

    def set_timelapse_feed(self, images, batchsize):
        """ Set the timelapse dictionary """
        logger.debug("Setting timelapse feed: (side: '%s', input_images: '%s', batchsize: %s)",
                     self.side, images, batchsize)
        self.timelapse_feed = self.load_generator().minibatch_ab(images[:batchsize],
                                                                 batchsize, self.side,
                                                                 do_shuffle=False,
                                                                 is_timelapse=True)
        logger.debug("Set timelapse feed")


class Samples():
    """ Display samples for preview and timelapse """
    def __init__(self, model, use_mask, frame_size, coverage_ratio, scaling):
        logger.debug("Initializing %s: model: '%s', use_mask: %s, coverage_ratio: %s)",
                     self.__class__.__name__, model, use_mask, coverage_ratio)
        self.model = model
        self.use_mask = use_mask
        self.frame_size = frame_size
        self.images = dict()
        self.coverage_ratio = coverage_ratio
        self.scaling = scaling
        self.header = self._get_headers(frame_size)

        logger.debug("Initialized %s", self.__class__.__name__)

    def create_preview_window(self):
        """ Display preview data """
        if len(self.images) == 2:
            logger.debug("Showing sample")
            image_grid = dict()
            for side, samples in self.images.items():
                other_side = "a" if side == "b" else "b"
                feed = samples[1:]
                target_scale = self.model.input_shape[0] / feed[0].shape[1]
                feed[0] = self.resize_samples(side, feed[0], target_scale)
                preds = self.get_predictions(side, other_side, feed)
                image_grid[side] = self.create_image_grid(side, samples, preds)
            full_grid = np.concatenate([image_grid["a"], image_grid["b"]], axis=1)
            print(full_grid.shape)
            full_display = np.concatenate([self.header, full_grid], axis=0)
            print(full_display.shape)
            print("scaling : ", self.scaling)
            full_display = np.squeeze(self.resize_samples('a', full_display[None, ...], self.scaling))
            print(full_display.shape)
            full_display = np.clip(full_display * 255., 0., 255.).astype('uint8')
            logger.debug("Compiled sample")
        else:
            full_display = None
            logger.debug("Ping Pong training - Only one side trained. Aborting preview")
        return full_display

    @staticmethod
    def resize_samples(side, samples, scale):
        """ Resize samples where predictor expects different shape from processed image """
        if scale != 1.:
            logger.debug("Resizing samples: (side: '%s', sample.shape: %s, scale: %s)",
                         side, samples[0].shape, scale)
            interp = cv2.INTER_CUBIC if scale > 1. else cv2.INTER_AREA
            gen = (cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=interp)
                   for img in samples)
            samples = np.array(tuple(gen))
            logger.debug("Resized sample: (side: '%s' shape: %s)", side, samples[0].shape)
        return samples

    def get_predictions(self, side, other_side, feed):
        """ Return the sample predictions from the model """
        logger.debug("Getting Predictions for side %s", side)
        reconstructions = self.model.predictors[side].predict(feed)
        swaps = self.model.predictors[other_side].predict(feed)
        if not isinstance(reconstructions, np.ndarray):
            reconstructions = reconstructions[self.model.largest_face_index]
            swaps = swaps[self.model.largest_face_index]
        logger.debug("Returning predictions: %s", swaps.shape)
        return [reconstructions, swaps]

    def create_image_grid(self, side, samples, predictions):
        """ Patch the images into the full frame / image grid """
        logger.debug("side: '%s', number of sample arrays: %s, prediction.shapes: %s)",
                     side, len(samples), [pred.shape for pred in predictions])

        frames, originals = samples[:2]
        target_scale = frames.shape[1] / originals.shape[1] * self.coverage_ratio
        images = np.concatenate([originals, *predictions], axis=0)
        masks = np.repeat(samples[2], 3, axis=3) if self.use_mask else np.ones_like(images)
        print(frames.shape)
        print(images.shape)
        images = self.tint_masked_areas(originals, images, masks)
        print(images.shape)
        images = self.resize_samples(side, images, target_scale)
        print(images.shape)
        if self.coverage_ratio != 1.:
            frames = self.frame_overlay(frames)
            images = self.overlay_foreground(frames, images)
        print(images.shape)
        images_list = np.split(images, 7, axis=0)
        for index, imgs in enumerate(images_list):
            print("imgs: ", imgs.shape)
            images_list[index] = imgs.reshape(self.frame_size, self.frame_size * 6, 3)
        print("imgs2 : ", images_list[0].shape)
        images = np.concatenate(images_list, axis=0)
        #images = images.reshape((self.frame_size * 7, self.frame_size * 6, 3))
        print(images.shape)
        return images

    def frame_overlay(self, frames):
        """ Add roi frame to a backfround image """
        logger.debug("full_size: %s", frames.shape[1])
        color = (0., 0., 1.)
        line_width = 3
        full_size = frames.shape[1]
        padding = int(full_size * (1. - self.coverage_ratio)) // 2
        length = int(full_size * self.coverage_ratio) // 4
        t_l = padding - line_width
        b_r = full_size - padding + line_width

        top_left = slice(t_l, t_l + length), slice(t_l, t_l + length)
        bot_left = slice(b_r - length, b_r), slice(t_l, t_l + length)
        top_right = slice(b_r - length, b_r), slice(b_r - length, b_r)
        bot_right = slice(t_l, t_l + length), slice(b_r - length, b_r)

        for roi in [top_left, bot_left, top_right, bot_right]:
            frames[:, roi[0], roi[1]] = color
        logger.debug("Overlayed background. Shape: %s", frames.shape)
        return frames

    @staticmethod
    def tint_masked_areas(originals, images, masks):
        """ Add the mask to the faces for masked preview """
        replace_area = (np.rint(masks) == 0.)
        originals = np.repeat(originals, 3, axis=2)
        images[replace_area] = originals[replace_area]
        images[..., -1][replace_area[..., 0]] += 0.3
        logger.debug("masked shapes: %s", originals.shape[1:])
        return images

    @staticmethod
    def overlay_foreground(backgrounds, foregrounds):
        """ Overlay the masked faces into the center of the background """
        backgrounds[..., -1] += 0.3
        offset = (backgrounds.shape[1] - foregrounds.shape[1]) // 2
        slice_y = slice(offset, offset + foregrounds.shape[1])
        slice_x = slice(offset, offset + foregrounds.shape[2])
        backgrounds = np.repeat(backgrounds, 3, axis=0)
        for background, foreground in zip(backgrounds, foregrounds):
            background[slice_y, slice_x] = foreground
        logger.debug("Foreground inserted. Shape: %s", backgrounds.shape)
        return backgrounds

    def _get_headers(self, width):
        """ Set headers for images """
        logger.debug("width: %s", width)

        def text_size(text, font):
            """ Helper function for list comprehension """
            [text_width, text_height] = cv2.getTextSize(text, font, 0.8, 1)[0]
            return [text_width, text_height]

        height = 64
        offsets = [0, width, width * 2]
        header_a = np.ones((height, width * 3, 3), dtype='float32')
        header_b = np.ones((height, width * 3, 3), dtype='float32')
        for sides, header in zip([['A', 'B'], ['B', 'A']], [header_a, header_b]):
            texts = ["Target {0}".format(sides[0]),
                     "{0} > {0}".format(sides[0]),
                     "{0} > {1}".format(sides[0], sides[1])]
            text_sizes = [text_size(text, cv2.FONT_HERSHEY_SIMPLEX) for text in texts]
            y_texts = [int((height + text[1]) / 2) for text in text_sizes]
            x_texts = [int((width - text[0]) / 2 + off) for off, text in zip(offsets, text_sizes)]
            for text_x, text_y, text in zip(x_texts, y_texts, texts):
                cv2.putText(header,
                            text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0., 0., 0.),
                            1,
                            lineType=cv2.LINE_AA)
        full_header = np.concatenate([header_a, header_a, header_b, header_b], axis=1)

        logger.debug("texts: %s, text sizes: %s, text_x: %s, text_y: %s",
                     texts, text_sizes, x_texts, y_texts)
        logger.debug("header shape: %s", full_header.shape)
        return full_header


class Timelapse():
    """ Create the timelapse """
    def __init__(self, model, use_mask, frame_size, coverage_ratio, preview_images, batchers):
        logger.debug("Initializing %s: model: %s, use_mask: %s, frame_size: %s, "
                     "coverage_ratio: %s, preview_images: %s, batchers: '%s')",
                     self.__class__.__name__, model, use_mask, frame_size, coverage_ratio,
                     preview_images, batchers)
        self.preview_images = preview_images
        self.samples = Samples(model, use_mask, frame_size, coverage_ratio)
        self.model = model
        self.batchers = batchers
        self.output_file = None
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_sample(self, side, timelapse_kwargs):
        """ Perform timelapse """
        logger.debug("Getting timelapse samples: '%s'", side)
        if not self.output_file:
            self.setup(**timelapse_kwargs)
        self.samples.images[side] = self.batchers[side].compile_timelapse_sample()
        logger.debug("Got timelapse samples: '%s' - %s", side, len(self.samples.images[side]))

    def setup(self, input_a=None, input_b=None, output=None):
        """ Set the timelapse output folder """
        logger.debug("Setting up timelapse")
        if output is None:
            output = str(get_folder(os.path.join(str(self.model.model_dir),
                                                 "{}_timelapse".format(self.model.name))))
        self.output_file = str(output)
        logger.debug("Timelapse output set to '%s'", self.output_file)

        images = {"a": get_image_paths(input_a), "b": get_image_paths(input_b)}
        batchsize = min(len(images["a"]),
                        len(images["b"]),
                        self.preview_images)
        for side, image_files in images.items():
            self.batchers[side].set_timelapse_feed(image_files, batchsize)
        logger.debug("Set up timelapse")

    def output_timelapse(self):
        """ Set the timelapse dictionary """
        logger.debug("Ouputting timelapse")
        image = self.samples.create_preview_window()
        if image is None:
            return
        filename = os.path.join(self.output_file, str(int(time.time())) + ".jpg")

        cv2.imwrite(filename, image)  # pylint: disable=no-member
        logger.debug("Created timelapse: '%s'", filename)


class PingPong():
    """ Side switcher for pingpong training """
    def __init__(self, model, sides):
        logger.debug("Initializing %s: (model: '%s')", self.__class__.__name__, model)
        self.active = model.training_opts.get("pingpong", False)
        self.model = model
        self.sides = sides
        self.side = sorted(sides)[0]
        self.loss = {side: [0] for side in sides}
        logger.debug("Initialized %s", self.__class__.__name__)

    def switch(self):
        """ Switch pingpong side """
        if not self.active:
            return
        retval = [side for side in self.sides if side != self.side][0]
        logger.info("Switching training to side %s", retval.title())
        self.side = retval
        self.reload_model()

    def reload_model(self):
        """ Load the model for just the current side """
        logger.verbose("Ping-Pong re-loading model")
        self.model.reset_pingpong()


class Landmarks():
    """ Set Landmarks for training into the model's training options"""
    def __init__(self, training_opts):
        logger.debug("Initializing %s: (training_opts: '%s')",
                     self.__class__.__name__, training_opts)
        self.size = training_opts.get("training_size", 256)
        self.paths = training_opts["alignments"]
        self.landmarks = self.get_alignments()
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_alignments(self):
        """ Obtain the landmarks for each faceset """
        landmarks = dict()
        for side, fullpath in self.paths.items():
            path, filename = os.path.split(fullpath)
            filename, extension = os.path.splitext(filename)
            serializer = extension[1:]
            alignments = Alignments(
                path,
                filename=filename,
                serializer=serializer)
            landmarks[side] = self.transform_landmarks(alignments)
        return landmarks

    def transform_landmarks(self, alignments):
        """ For each face transform landmarks and return """
        landmarks = dict()
        for _, faces, _, _ in alignments.yield_faces():
            for face in faces:
                detected_face = DetectedFace()
                detected_face.from_alignment(face)
                detected_face.load_aligned(None, size=self.size)
                landmarks[detected_face.hash] = detected_face.aligned_landmarks
        return landmarks
