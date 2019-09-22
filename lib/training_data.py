#!/usr/bin/env python3
""" Process training data for model training """

import logging

from hashlib import sha1
from pathlib import Path
from random import random, shuffle, choice

import numpy as np
import cv2
from scipy.interpolate import griddata

from lib.model import masks
from lib.multithreading import BackgroundGenerator
from lib.queue_manager import queue_manager
from lib.umeyama import umeyama
from lib.utils import cv2_read_img, FaceswapError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TrainingDataGenerator():
    """ Generate training data for models """
    def __init__(self, model_input_size, model_output_shapes, training_opts, config):
        logger.debug("Initializing %s: (model_input_size: %s, model_output_shapes: %s, "
                     "training_opts: %s, landmarks: %s, config: %s)",
                     self.__class__.__name__, model_input_size, model_output_shapes,
                     {key: val for key, val in training_opts.items() if key != "landmarks"},
                     bool(training_opts.get("landmarks", None)), config)
        self.batchsize = 0
        self.model_input_size = model_input_size
        self.model_output_shapes = model_output_shapes
        self.training_opts = training_opts
        self.mask_class = self.set_mask_class()
        self.landmarks = self.training_opts.get("landmarks", None)
        self._nearest_landmarks = {}
        self.processing = ImageManipulation(model_input_size,
                                            model_output_shapes,
                                            training_opts.get("coverage_ratio", 0.625),
                                            config)
        self.transforms = np.array([config.get("rotation_range", 10),
                                    config.get("zoom_range", 5) / 100.,
                                    config.get("shift_range", 5) / 100.,
                                    config.get("shift_range", 5) / 100.,
                                    100.])
        self.flip_chance = config.get("random_flip", 50) * 2. - 100.
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_mask_class(self):
        """ Set the mask function to use if using mask """
        mask_type = self.training_opts.get("mask_type", None)
        if mask_type:
            logger.debug("Mask type: '%s'", mask_type)
            mask_class = getattr(masks, mask_type)
        else:
            mask_class = None
        logger.debug("Mask class: %s", mask_class)
        return mask_class

    def get_landmarks(self, filename, image, side):
        """ Return the landmarks for this face """
        logger.trace("Retrieving landmarks: (filename: '%s', side: '%s'", filename, side)
        lm_key = sha1(image).hexdigest()
        try:
            src_points = self.landmarks[side][lm_key]
        except KeyError as err:
            msg = ("At least one of your images does not have a matching entry in your alignments "
                   "file."
                   "\nIf you are training with a mask or using 'warp to landmarks' then every "
                   "face you intend to train on must exist within the alignments file."
                   "\nThe specific file that caused the failure was '{}' which has a hash of {}."
                   "\nMost likely there will be more than just this file missing from the "
                   "alignments file. You can use the Alignments Tool to help identify missing "
                   "alignments".format(lm_key, filename))
            raise FaceswapError(msg) from err
        logger.trace("Returning: (src_points: %s)", src_points)
        return src_points

    def image_loader(self, filename, side):
        """ Load and resize images with opencv """
        image = cv2_read_img(filename, raise_error=True)
        landmarks = self.get_landmarks(filename, image, side) if self.landmarks else None
        return image, landmarks

    def dataset_setup(self, img_file_list, side):
        """ Create a mem-mapped image and landmark array for training"""
        height, width = cv2_read_img(img_file_list[0], raise_error=True).shape[:2]
        image_file = str(Path(img_file_list[0]).parents[0].joinpath(('Images_'))) + side + '.npy'
        channels = 4 if self.mask_class else 3
        image_shape = (len(img_file_list), height, width, channels)
        images = np.memmap(image_file, dtype='uint8', mode='w+', shape=image_shape)

        mark_file = str(Path(img_file_list[0]).parents[0].joinpath(('Landmarks_'))) + side + '.npy'
        mark_shape = (len(img_file_list), 68, 2)
        landmarks = np.memmap(mark_file, dtype='uint32', mode='w+', shape=mark_shape)

        generator = (self.image_loader(filename, side) for filename in img_file_list)
        for index, (image, landmark) in enumerate(generator):
            images[index, :, :, :3] = image[:, :, :3]
            if self.landmarks:
                landmarks[index] = landmark

        if self.mask_class:
            for index, (image, landmark) in enumerate(zip(images, landmarks)):
                images[index] = self.mask_class(landmark, image, channels=4).mask

        del images  # flush memmap to disk and save changes
        del landmarks  # flush memmap to disk and save changes

        return [image_file, mark_file, image_shape, mark_shape]

    def minibatch_ab(self, image_dataset, batchsize, side, do_shuffle=True, is_preview=False,
                     is_timelapse=False):
        """ Keep a queue filled to 8x Batch Size """
        logger.debug("Queue batches: (image_count: %s, batchsize: %s, side: '%s', "
                     "do_shuffle: %s, is_preview, %s, is_timelapse: %s)", image_dataset[2][0],
                     batchsize, side, do_shuffle, is_preview, is_timelapse)
        self.batchsize = batchsize
        is_display = is_preview or is_timelapse
        args = (image_dataset, side, is_display, do_shuffle, batchsize)
        batcher = BackgroundGenerator(self.minibatch, thread_count=2, args=args)
        return batcher.iterator()

    def validate_samples(self, length):
        """ Check the total number of images against batchsize and return
            the total number of images """
        msg = ("Number of images is lower than batch-size (Note that too few "
               "images may lead to bad training). # images: {}, "
               "batch-size: {}".format(length, self.batchsize))
        try:
            assert length >= self.batchsize, msg
        except AssertionError as err:
            msg += ("\nYou should increase the number of images in your training set or lower "
                    "your batch-size.")
            raise FaceswapError(msg) from err

    def cache_matrices(self, images, image_shape):
        """ Pre-compute affine transformation matrices for an entire epoch """
        logger.trace("Cache an epoch of transform matrices")

        affine = np.random.uniform(-1. ,1. ,(image_shape[0], 5)) * self.transforms[None, :]
        matrices = np.ones((image_shape[0], 2, 3), dtype='float32')
        flip = np.ones(image_shape[0], dtype='float32')
        alpha = (affine[:, 1] + 1.) * np.cos(np.radians(affine[:, 0]))
        beta = (affine[:, 1] + 1.) * np.sin(np.radians(affine[:, 0]))
        matrices[:, 0, 0] = alpha
        matrices[:, 1, 0] = -beta
        matrices[:, 0, 1] = beta
        matrices[:, 1, 1] = alpha
        matrices[:, 0, 2] = image_shape[1] * (0.5 * (-beta + (1 - alpha)) + affine[:, 2])
        matrices[:, 1, 2] = image_shape[1] * (0.5 * (beta + (1 - alpha)) + affine[:, 3])

        flips = affine[:, 4] >= self.flip_chance
        images[flips] = images[flips, : ,::-1]
        logger.debug("Finished caching an epoch of transform matrices")
        return images, matrices

    def minibatch(self, image_dataset, side, is_display, do_shuffle, batchsize):
        """ A generator function that yields epoch, batchsize of warped_img
            and batchsize of target_img from the load queue """
        logger.debug("Loading minibatch generator: (image_count: %s, side: '%s', is_display: %s)",
                     image_dataset[2][0], side, is_display)
        self.validate_samples(image_dataset[2][0])

        def _image_iterator(do_shuffle, dataset):
            """ Yield pairs of corresponding images and landmarks and shuffle as needed """
            while True:
                images = np.memmap(dataset[0], dtype='uint8', mode='c', shape=dataset[2])
                landmarks = np.memmap(dataset[1], dtype='uint32', mode='c', shape=dataset[3])
                images, matrices = self.cache_matrices(images, dataset[2])
                if do_shuffle:
                    rng_state = np.random.get_state()
                    np.random.set_state(rng_state)
                    np.random.shuffle(images)
                    np.random.set_state(rng_state)
                    np.random.shuffle(landmarks)
                for image, landmark, matrix in zip(images, landmarks, matrices):
                    yield image, landmark, matrix
                del images
                del landmarks

        image_iterator = _image_iterator(do_shuffle, image_dataset)
        while True:
            batch = list()
            for _ in range(batchsize):
                image, landmark, matrix = next(image_iterator)
                data = self.process_face(image, landmark, matrix, side, is_display)
                batch.append(data)
            batch = list(zip(*batch))
            batch = [np.array(x, dtype="float32") for x in batch]
            logger.trace("Yielding batch: (size: %s, item shapes: %s, side: '%s', "
                         "is_display: %s)",
                         len(batch), [item.shape for item in batch], side, is_display)
            yield batch

        logger.debug("Finished minibatch generator: (side: '%s', is_display: %s)",
                     side, is_display)

    def process_face(self, image, landmarks, matrix, side, is_display):
        """ Load an image and perform transformation and warping """
        image = self.processing.color_adjust(image,
                                             self.training_opts["augment_color"],
                                             is_display)
        if not is_display:
            image = self.processing.random_transform(image, matrix)
        sample = image.copy()[:, :, :3]

        if self.training_opts["warp_to_landmarks"]:
            warped_landmarks = self.get_closest_match(filename, side, landmarks)
            processed = self.processing.random_warp_landmarks(image, landmarks, warped_landmarks)
        else:
            processed = self.processing.random_warp(image)

        processed.insert(0, sample)
        logger.trace("Processed face: (side: '%s', shapes: %s)",
                     side, [img.shape for img in processed])
        return processed

    def get_closest_match(self, filename, side, src_points):
        """ Return closest matched landmarks from opposite set """
        logger.trace("Retrieving closest matched landmarks: (filename: '%s', src_points: '%s'",
                     filename, src_points)
        landmarks = self.landmarks["a"] if side == "b" else self.landmarks["b"]
        closest_hashes = self._nearest_landmarks.get(filename)
        if not closest_hashes:
            dst_points_items = list(landmarks.items())
            dst_points = list(x[1] for x in dst_points_items)
            closest = (np.mean(np.square(src_points - dst_points), axis=(1, 2))).argsort()[:10]
            closest_hashes = tuple(dst_points_items[i][0] for i in closest)
            self._nearest_landmarks[filename] = closest_hashes
        dst_points = landmarks[choice(closest_hashes)]
        logger.trace("Returning: (dst_points: %s)", dst_points)
        return dst_points


class ImageManipulation():
    """ Manipulations to be performed on training images """
    def __init__(self, input_size, output_shapes, coverage_ratio, config):
        """ input_size: Size of the face input into the model
            output_shapes: Shapes that come out of the model
            coverage_ratio: Coverage ratio of full image. Eg: 256 * 0.625 = 160
        """
        logger.debug("Initializing %s: (input_size: %s, output_shapes: %s, coverage_ratio: %s, "
                     "config: %s)", self.__class__.__name__, input_size, output_shapes,
                     coverage_ratio, config)
        self.config = config
        # Transform and Warp args
        self.input_size = input_size
        self.output_sizes = [shape[1] for shape in output_shapes if shape[2] == 3]
        logger.debug("Output sizes: %s", self.output_sizes)
        # Warp args
        self.coverage_ratio = coverage_ratio  # Coverage ratio of full image. Eg: 256 * 0.625 = 160
        self.scale = 5  # Normal random variable scale
        logger.debug("Initialized %s", self.__class__.__name__)

    def color_adjust(self, img, augment_color, is_display):
        """ Color adjust RGB image """
        logger.trace("Color adjusting image")
        if not is_display and augment_color:
            logger.trace("Augmenting color")
            face, _ = self.separate_mask(img)
            face = face.astype("uint8")
            face = self.random_clahe(face)
            face = self.random_lab(face)
            img[:, :, :3] = face
        return img.astype('float32') / 255.0

    def random_clahe(self, image):
        """ Randomly perform Contrast Limited Adaptive Histogram Equilization """
        contrast_random = random()
        if contrast_random > self.config.get("color_clahe_chance", 50) / 100:
            return image

        base_contrast = image.shape[0] // 128
        grid_base = random() * self.config.get("color_clahe_max_size", 4)
        contrast_adjustment = int(grid_base * (base_contrast / 2))
        grid_size = base_contrast + contrast_adjustment
        logger.trace("Adjusting Contrast. Grid Size: %s", grid_size)

        clahe = cv2.createCLAHE(clipLimit=2.0,  # pylint: disable=no-member
                                tileGridSize=(grid_size, grid_size))
        for chan in range(3):
            image[:, :, chan] = clahe.apply(image[:, :, chan])
        return image

    def random_lab(self, image):
        """ Perform random color/lightness adjustment in L*a*b* colorspace """
        amount_l = self.config.get("color_lightness", 30) / 100
        amount_ab = self.config.get("color_ab", 8) / 100

        randoms = [(random() * amount_l * 2) - amount_l,  # L adjust
                   (random() * amount_ab * 2) - amount_ab,  # A adjust
                   (random() * amount_ab * 2) - amount_ab]  # B adjust

        logger.trace("Random LAB adjustments: %s", randoms)
        image = cv2.cvtColor(  # pylint:disable=no-member
            image, cv2.COLOR_BGR2LAB).astype("float32") / 255.0  # pylint:disable=no-member

        for idx, adjustment in enumerate(randoms):
            if adjustment >= 0:
                image[:, :, idx] = ((1 - image[:, :, idx]) * adjustment) + image[:, :, idx]
            else:
                image[:, :, idx] = image[:, :, idx] * (1 + adjustment)
        image = cv2.cvtColor((image * 255.0).astype("uint8"),  # pylint:disable=no-member
                             cv2.COLOR_LAB2BGR)  # pylint:disable=no-member
        return image

    @staticmethod
    def separate_mask(image):
        """ Return the image and the mask from a 4 channel image """
        mask = None
        if image.shape[2] == 4:
            logger.trace("Image contains mask")
            mask = np.expand_dims(image[:, :, -1], axis=2)
            image = image[:, :, :3]
        else:
            logger.trace("Image has no mask")
        return image, mask

    def get_coverage(self, image):
        """ Return coverage value for given image """
        coverage = int(image.shape[0] * self.coverage_ratio)
        logger.trace("Coverage: %s", coverage)
        return coverage

    def random_transform(self, image, matrix):
        """ Randomly transform an image """
        logger.trace("Randomly transforming image")
        height, width = image.shape[0:2]
        result = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
        logger.trace("Randomly transformed image")
        return result

    def random_warp(self, image):
        """ get pair of random warped images from aligned face image """
        logger.trace("Randomly warping image")
        height, width = image.shape[0:2]
        coverage = self.get_coverage(image) // 2
        try:
            assert height == width and height % 2 == 0
        except AssertionError as err:
            msg = ("Training images should be square with an even number of pixels across each "
                   "side. An image was found with width: {}, height: {}."
                   "\nMost likely this is a frame rather than a face within your training set. "
                   "\nMake sure that the only images within your training set are faces generated "
                   "from the Extract process.".format(width, height))
            raise FaceswapError(msg) from err

        range_ = np.linspace(height // 2 - coverage, height // 2 + coverage, 5, dtype='float32')
        mapx = np.broadcast_to(range_, (5, 5)).copy()
        mapy = mapx.T
        # mapx, mapy = np.float32(np.meshgrid(range_,range_)) # instead of broadcast

        pad = int(1.25 * self.input_size)
        slices = slice(pad // 10, -pad // 10)
        dst_slices = [slice(0, (size + 1), (size // 4)) for size in self.output_sizes]
        interp = np.empty((2, self.input_size, self.input_size), dtype='float32')

        for i, map_ in enumerate([mapx, mapy]):
            map_ = map_ + np.random.normal(size=(5, 5), scale=self.scale)
            interp[i] = cv2.resize(map_, (pad, pad))[slices, slices]  # pylint:disable=no-member

        warped_image = cv2.remap(  # pylint:disable=no-member
            image, interp[0], interp[1], cv2.INTER_LINEAR)  # pylint:disable=no-member
        logger.trace("Warped image shape: %s", warped_image.shape)

        src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
        dst_points = [np.mgrid[dst_slice, dst_slice] for dst_slice in dst_slices]
        mats = [umeyama(src_points, True, dst_pts.T.reshape(-1, 2))[0:2]
                for dst_pts in dst_points]

        target_images = [cv2.warpAffine(image,  # pylint:disable=no-member
                                        mat,
                                        (self.output_sizes[idx], self.output_sizes[idx]))
                         for idx, mat in enumerate(mats)]

        logger.trace("Target image shapes: %s", [tgt.shape for tgt in target_images])
        return self.compile_images(warped_image, target_images)

    def random_warp_landmarks(self, image, src_points=None, dst_points=None):
        """ get warped image, target image and target mask
            From DFAKER plugin """
        logger.trace("Randomly warping landmarks")
        size = image.shape[0]
        coverage = self.get_coverage(image) // 2

        p_mx = size - 1
        p_hf = (size // 2) - 1

        edge_anchors = [(0, 0), (0, p_mx), (p_mx, p_mx), (p_mx, 0),
                        (p_hf, 0), (p_hf, p_mx), (p_mx, p_hf), (0, p_hf)]
        grid_x, grid_y = np.mgrid[0:p_mx:complex(size), 0:p_mx:complex(size)]

        source = src_points
        destination = (dst_points.copy().astype('float32') +
                       np.random.normal(size=dst_points.shape, scale=2.0))
        destination = destination.astype('uint8')

        face_core = cv2.convexHull(np.concatenate(  # pylint:disable=no-member
            [source[17:], destination[17:]], axis=0).astype(int))

        source = [(pty, ptx) for ptx, pty in source] + edge_anchors
        destination = [(pty, ptx) for ptx, pty in destination] + edge_anchors

        indicies_to_remove = set()
        for fpl in source, destination:
            for idx, (pty, ptx) in enumerate(fpl):
                if idx > 17:
                    break
                elif cv2.pointPolygonTest(face_core,  # pylint:disable=no-member
                                          (pty, ptx),
                                          False) >= 0:
                    indicies_to_remove.add(idx)

        for idx in sorted(indicies_to_remove, reverse=True):
            source.pop(idx)
            destination.pop(idx)

        grid_z = griddata(destination, source, (grid_x, grid_y), method="linear")
        map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(size, size)
        map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(size, size)
        map_x_32 = map_x.astype('float32')
        map_y_32 = map_y.astype('float32')

        warped_image = cv2.remap(image,  # pylint:disable=no-member
                                 map_x_32,
                                 map_y_32,
                                 cv2.INTER_LINEAR,  # pylint:disable=no-member
                                 cv2.BORDER_TRANSPARENT)  # pylint:disable=no-member
        target_image = image

        # TODO Make sure this replacement is correct
        slices = slice(size // 2 - coverage, size // 2 + coverage)
#        slices = slice(size // 32, size - size // 32)  # 8px on a 256px image
        warped_image = cv2.resize(  # pylint:disable=no-member
            warped_image[slices, slices, :], (self.input_size, self.input_size),
            cv2.INTER_AREA)  # pylint:disable=no-member
        logger.trace("Warped image shape: %s", warped_image.shape)
        target_images = [cv2.resize(target_image[slices, slices, :],  # pylint:disable=no-member
                                    (size, size),
                                    cv2.INTER_AREA)  # pylint:disable=no-member
                         for size in self.output_sizes]

        logger.trace("Target image shapea: %s", [img.shape for img in target_images])
        return self.compile_images(warped_image, target_images)

    def compile_images(self, warped_image, target_images):
        """ Compile the warped images, target images and mask for feed """
        warped_image, _ = self.separate_mask(warped_image)
        final_target_images = list()
        target_mask = None
        for target_image in target_images:
            image, mask = self.separate_mask(target_image)
            final_target_images.append(image)
            # Add the mask if it exists and is the same size as our largest output
            if mask is not None and mask.shape[1] == max(self.output_sizes):
                target_mask = mask

        retval = [warped_image] + final_target_images
        if target_mask is not None:
            logger.trace("Target mask shape: %s", target_mask.shape)
            retval.append(target_mask)

        logger.trace("Final shapes: %s", [img.shape for img in retval])
        return retval


def stack_images(images):
    """ Stack images """
    logger.debug("Stack images")

    def get_transpose_axes(num):
        if num % 2 == 0:
            logger.debug("Even number of images to stack")
            y_axes = list(range(1, num - 1, 2))
            x_axes = list(range(0, num - 1, 2))
        else:
            logger.debug("Odd number of images to stack")
            y_axes = list(range(0, num - 1, 2))
            x_axes = list(range(1, num - 1, 2))
        return y_axes, x_axes, [num - 1]

    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]
    logger.debug("Stacked images")
    return np.transpose(
        images,
        axes=np.concatenate(new_axes)
        ).reshape(new_shape)
