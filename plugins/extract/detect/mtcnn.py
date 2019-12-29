#!/usr/bin/env python3
""" MTCNN Face detection plugin """

from __future__ import absolute_import, division, print_function

import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D, Permute, PReLU
from lib.model.session import KSession
import cv2
from ._base import Detector, logger


class Detect(Detector):
    """ MTCNN detector for face recognition """
    def __init__(self, **kwargs):
        git_model_id = 2
        model_filename = ["mtcnn_det_v2.1.h5", "mtcnn_det_v2.2.h5", "mtcnn_det_v2.3.h5"]
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.name = "MTCNN"
        self.input_size = 640
        self.vram = 320
        self.vram_warnings = 64  # Will run at this with warnings
        self.vram_per_batch = 32
        self.batchsize = self.config["batch-size"]
        self.colorformat = "RGB"

    def init_model(self):
        """ Initialize MTCNN sub-models
        Parameters
        ----------
        model_path: str
            Weight file path to enable weight loading into the constructed model
        allow_growth: bool
            Tensorflow variable which delays pre-allocation of GPU memory
        threshold: list of floats
            Threshold used to filter box candidates returned from the neural networks
        """
        self.p_net = Pnet(self.model_path[0], self.config["allow_growth"])
        self.r_net = Rnet(self.model_path[1], self.config["allow_growth"])
        self.o_net = Onet(self.model_path[2], self.config["allow_growth"])

        # self.p_net._model.save("p_model.h5")
        # self.r_net._model.save("r_model.h5")
        # self.o_net._model.save("o_model.h5")
        self.threshold = np.clip(np.array([self.config["threshold_1"],
                                           self.config["threshold_2"],
                                           self.config["threshold_3"]], dtype=np.float32),
                                 0.0, 1.0)
        self._image_size = None
        self._input_scales = None

    def process_input(self, batch):
        """ Compile the detection image(s) for prediction """
        image_shape = batch["image"].shape[1:3]
        if self._image_size != image_shape:
            self._image_size = image_shape
            self._input_scales = self.calculate_scales(image_shape)
        batch["feed"] = (batch["image"] - 127.5) / 127.5
        return batch

    def predict(self, batch):
        """Detect faces in a batch of images, and returns an array of bounding boxes and
        landmarks points for each image in the batch.

        Parameters
        ----------
        batch: :class:`numpy.ndarry`
            Array of shape Nx3

        Returns
        -------
        batch["prediction"]: :class:`numpy.ndarry`
            Array of shape Nx5 with the bounding box vertice coordinates
        batch["mtcnn_points"]: :class:`numpy.ndarry`
            Array of shape Nx5 with the facial landmark points
        """
        squares = self.first_stage(batch["feed"])
        squares = self.second_stage(batch["feed"], squares)
        squares = self.third_stage(batch["feed"], squares)

        batch["prediction"] = [square[:5] if square.size != 0 else np.empty((0, 5))
                               for square in squares]
        batch["mtcnn_points"] = [square[5:]  if square.size != 0 else np.empty((0, 5))
                                 for square in squares]

        logger.trace("filename: %s, prediction: %s, landmarks: %s",
                     batch["filename"], batch["prediction"], batch["mtcnn_points"])
        return batch

    def process_output(self, batch):
        """ Post process the detected faces """
        return batch

    def first_stage(self, images):
        """ first stage - fast proposal network (pnet) to obtain face candidates """
        squares = [[] for _ in range(images.shape[0])]

        for scale in self._input_scales:
            scale_factor = images.shape[1] / scale
            batch = np.stack([cv2.resize(image, (scale, scale), interpolation=cv2.INTER_AREA)
                              for image in images])
            classifers, regressions = self.p_net.predict(batch, batch_size=128)
            longest_side = max(2, *classifers.shape[1:3])
            stride = (2.0 * longest_side - 1.0) / (longest_side - 1.0)
            for index, (classifer, regression) in enumerate(zip(classifers[..., 1], regressions)):
                boxes = self.process_first_stage(classifer,
                                                 regression,
                                                 stride,
                                                 scale_factor,
                                                 self.threshold[0])
                boxes = self.nms(boxes, 0.5, "iou")
                boxes = self.centered_square(boxes, self._image_size) # move to after second nms
                squares[index].extend(boxes)
        squares = [self.nms(np.stack(box_list), 0.7, 'iou') if box_list else np.empty((0, 5))
                   for box_list in squares]
        return squares

    def second_stage(self, image_batch, squares_batch):
        """ second stage - refinement of face candidates with rnet """
        squares_list = []
        for image, squares in zip(image_batch, squares_batch):
            if squares.size == 0:
                squares_list.append(squares)
                continue
            scaled_images = np.empty(((squares.shape[0],) + (24, 24, 3)), dtype=np.float32)
            for index, square in enumerate(squares):
                scaled_images[index] = self.crop_and_resize(image, (24, 24), square[0:4], True)
            classifer, regression = self.r_net.predict(scaled_images, batch_size=256)
            boxes = self.process_results(classifer[:, 1],
                                         regression,
                                         None,
                                         squares,
                                         self.threshold[1],
                                         2)
            boxes = self.nms(boxes, 0.7, 'iou')
            squares = self.centered_square(boxes, self._image_size)
            squares_list.append(squares)
        return squares_list

    def third_stage(self, image_batch, squares_batch):
        """ third stage - further refinement and facial landmarks positions with onet """
        squares_list = []
        for image, squares in zip(image_batch, squares_batch):
            if squares.size == 0:
                squares_list.append(squares)
                continue
            scaled_images = np.empty(((squares.shape[0],) + (48, 48, 3)), dtype=np.float32)
            for index, square in enumerate(squares):
                scaled_images[index] = self.crop_and_resize(image, (48, 48), square[0:4], True)
            scores, offsets, landmarks = self.o_net.predict(scaled_images, batch_size=256)
            boxes = self.process_results(scores[:, 1],
                                         offsets,
                                         landmarks,
                                         squares,
                                         self.threshold[2],
                                         3)
            boxes = self.nms(boxes, 0.7, 'iom')
            squares = self.centered_square(boxes, self._image_size)
            squares_list.append(squares)
        return squares_list

    def aggregate_batches(self, images, list_of_squares):
        """ blah """
        image_list, square_list = [], []
        for image, squares in zip(images, list_of_squares):
            if squares is not None:
                for square in squares:
                    image_list.append(self.crop_and_resize(image, (48, 48), square[0:4], True))
                square_list.extend(squares)
        return image_list, square_list

    @staticmethod
    def process_first_stage(classifer, regression, stride, scale, threshold):
        """ Detect face position and calibrate bounding box on 12net feature map(matrix version)
        Input:
            classification_probability : softmax feature map for face classify
            roi                 : feature map for regression
            stride              : bounding box XXXXXXXXXXXXXXXXXXX
            scale               : current input image scale in multi-scales
            threshold           : 0.6 can have 99% recall rate
        """
        y_coord, x_coord = np.where(classifer >= threshold)
        rectangle = np.array([x_coord, y_coord, x_coord, y_coord]) * stride
        rectangle[2:4] = rectangle[2:4] + 11.0
        offset = regression[x_coord, y_coord] * 12.0
        rectangle = (rectangle.T + offset) * scale # examine fix() of bbox

        #scores = classifer[coordinates]
        scores = np.array([classifer[y_coord, x_coord]]).T
        boxes = np.concatenate((rectangle, scores), axis=1)
        return boxes

    @staticmethod
    def process_results(face_classifer, offset_regression, landmarks, squares, threshold, stage):
        """ Process R-Net and O-Net's output

        Parameters
        ----------
        face_classifer: :class:`numpy.ndarry`
            classification output with a probability that a face is present in the image crop
        offset_regression: :class:`numpy.ndarry`
            regression with a predicted offset to the prior stage's results
        landmarks: :class:`numpy.ndarry`
            regression with the predicted coordinates of the 5 facial landmarks
        squares: :class:`numpy.ndarry`
            previous stage's output
        threshold: tuple of floats
            tuple containing the confidence threshold to select rectangles
        stage: int
            qualifier to select landmark logic chain

        Returns
        -------
        rects: :class:`numpy.ndarry`
            Array of shape Nx5 or Nx15 containing the rectangles vertices and face landmarks
        """
        confident_boxes = face_classifer >= threshold
        rects = squares[confident_boxes]

        width_height = np.stack([rects[:, 2] - rects[:, 0], rects[:, 3] - rects[:, 1]], axis=1)
        width_height = np.concatenate([width_height, width_height], axis=1)
        offsets = offset_regression[confident_boxes] * width_height

        rects[:, 0:4] = rects[:, 0:4] + offsets
        rects[:, 4] = face_classifer[confident_boxes]

        if stage == 3:
            x_landmarks = (landmarks[confident_boxes, 0:5] * width_height[:, 0:1] + rects[:, 0:1])
            y_landmarks = (landmarks[confident_boxes, 5:10] * width_height[:, 1:2] + rects[:, 1:2])
            rects = np.concatenate([rects,
                                    x_landmarks[:, 0:1], y_landmarks[:, 0:1],
                                    x_landmarks[:, 1:2], y_landmarks[:, 1:2],
                                    x_landmarks[:, 2:3], y_landmarks[:, 2:3],
                                    x_landmarks[:, 3:4], y_landmarks[:, 3:4],
                                    x_landmarks[:, 4:5], y_landmarks[:, 4:5]], axis=1)
        return rects

    @staticmethod
    def calculate_scales(orig_image_shape):
        """ Calculate image pyramid for NN to process. Limit image sizes to multiples of 8 for GPU
        best practices. Each level of the pyramid is approximately 70% of the next largest layer's
        size. Hard-code smallest face input to be 32 by 32 pixels to avoid poor extractions.

        Parameters
        ----------
        orig_image_shape: tuple of ints
            tuple containing the original image's height and width in pixels

        Returns
        -------
        applicable_scales: list of ints
            List of image sizes that will be included when downsizing and creating the image pyramid
            to send to the NN
        """
        largest_side = max(orig_image_shape)
        scales = [32, 40, 56, 80, 112, 160, 224, 320, 448, 632, 896, 1264, 1784, 2512, 3552]
        applicable_scales = [scale for scale in scales if scale <= largest_side]
        return applicable_scales

# MTCNN Detector
# Original Matlab implementation from:
# https://github.com/kpzhang93/MTCNN_face_detection_alignment
#
# Python implementation provided by Kyle Vrooman from
# https://github.com/deepfakes/faceswap/blob/master/plugins/extract/detect/mtcnn.py
#
# Trained Keras model weights from:
# https://github.com/xiangrufan/keras-mtcnn
#
# MIT License
#
# Copyright (c) 2016 Kaipeng Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class Pnet(KSession):
    """ Keras P_Net model for MTCNN """
    def __init__(self, model_path, allow_growth):
        super().__init__("MTCNN-PNet", model_path, allow_growth=allow_growth)
        self.define_model(self.model_definition)
        self.load_model_weights()

    @staticmethod
    def model_definition():
        """ Keras P_Net model for MTCNN """
        input_ = Input(shape=(None, None, 3))
        var_x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input_)
        var_x = PReLU(shared_axes=[1, 2], name='PReLU1')(var_x)
        var_x = MaxPool2D(pool_size=2)(var_x)
        var_x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='PReLU2')(var_x)
        var_x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='PReLU3')(var_x)
        classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(var_x)
        bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(var_x)
        return [input_], [classifier, bbox_regress]


class Rnet(KSession):
    """ Keras R_Net model for MTCNN """
    def __init__(self, model_path, allow_growth):
        super().__init__("MTCNN-RNet", model_path, allow_growth=allow_growth)
        self.define_model(self.model_definition)
        self.load_model_weights()

    @staticmethod
    def model_definition():
        """ Keras R_Net model for MTCNN """
        input_ = Input(shape=(24, 24, 3))
        var_x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input_)
        var_x = PReLU(shared_axes=[1, 2], name='prelu1')(var_x)
        var_x = MaxPool2D(pool_size=3, strides=2, padding='same')(var_x)

        var_x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='prelu2')(var_x)
        var_x = MaxPool2D(pool_size=3, strides=2)(var_x)

        var_x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='prelu3')(var_x)
        var_x = Permute((3, 2, 1))(var_x)
        var_x = Flatten()(var_x)
        var_x = Dense(128, name='conv4')(var_x)
        var_x = PReLU(name='prelu4')(var_x)
        classifier = Dense(2, activation='softmax', name='conv5-1')(var_x)
        bbox_regress = Dense(4, name='conv5-2')(var_x)
        return [input_], [classifier, bbox_regress]


class Onet(KSession):
    """ Keras O_Net model for MTCNN """
    def __init__(self, model_path, allow_growth):
        super().__init__("MTCNN-ONet", model_path, allow_growth=allow_growth)
        self.define_model(self.model_definition)
        self.load_model_weights()

    @staticmethod
    def model_definition():
        """ Keras O_Net model for MTCNN """
        input_ = Input(shape=(48, 48, 3))
        var_x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input_)
        var_x = PReLU(shared_axes=[1, 2], name='prelu1')(var_x)
        var_x = MaxPool2D(pool_size=3, strides=2, padding='same')(var_x)
        var_x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='prelu2')(var_x)
        var_x = MaxPool2D(pool_size=3, strides=2)(var_x)
        var_x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='prelu3')(var_x)
        var_x = MaxPool2D(pool_size=2)(var_x)
        var_x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(var_x)
        var_x = PReLU(shared_axes=[1, 2], name='prelu4')(var_x)
        var_x = Permute((3, 2, 1))(var_x)
        var_x = Flatten()(var_x)
        var_x = Dense(256, name='conv5')(var_x)
        var_x = PReLU(name='prelu5')(var_x)

        classifier = Dense(2, activation='softmax', name='conv6-1')(var_x)
        bbox_regress = Dense(4, name='conv6-2')(var_x)
        landmark_regress = Dense(10, name='conv6-3')(var_x)
        return [input_], [classifier, bbox_regress, landmark_regress]
