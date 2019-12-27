#!/usr/bin/env python3
""" MTCNN Face detection plugin """

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np

from scipy import ndimage
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D, Permute, PReLU

from lib.model.session import KSession
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
        self.kwargs = self.validate_kwargs()
        self.colorformat = "RGB"

    def validate_kwargs(self):
        """ Validate that config options are correct. If not reset to default """
        valid = True
        threshold = [self.config["threshold_1"],
                     self.config["threshold_2"],
                     self.config["threshold_3"]]
        kwargs = {"threshold": threshold,
                  "factor": self.config["scalefactor"]}

        if not all(0.0 < threshold <= 1.0 for threshold in kwargs['threshold']):
            valid = False
        elif not 0.0 < kwargs['factor'] < 1.0:
            valid = False

        if not valid:
            kwargs = {"threshold": [0.6, 0.7, 0.7],  # three steps threshold
                      "factor": 0.709}               # scale factor
            logger.warning("Invalid MTCNN options in config. Running with defaults")
        logger.debug("Using mtcnn kwargs: %s", kwargs)
        return kwargs

    def init_model(self):
        """ Initialize S3FD Model"""
        self.model = MTCNN(self.model_path, self.config["allow_growth"], **self.kwargs)

    def process_input(self, batch):
        """ Compile the detection image(s) for prediction """
        batch["feed"] = (batch["image"] - 127.5) / 127.5
        return batch

    def predict(self, batch):
        """ Run model to get predictions """
        prediction, points = self.model.detect_faces(batch["feed"])
        logger.trace("filename: %s, prediction: %s, mtcnn_points: %s",
                     batch["filename"], prediction, points)
        batch["prediction"], batch["mtcnn_points"] = prediction, points
        return batch

    def process_output(self, batch):
        """ Post process the detected faces """
        return batch


# MTCNN Detector
# Trained model weights adapted from: https://github.com/xiangrufan/keras-mtcnn
#
# Keras implementation of the face detection / alignment algorithm
# found at
# https://github.com/kpzhang93/MTCNN_face_detection_alignment
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


class P_Net(KSession):
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


class R_Net(KSession):
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


class O_Net(KSession):
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


class MTCNN():
    """ MTCNN Face Detector """

    def __init__(self, model_path, allow_growth, threshold):
        """
        threshold: threshold=[th1, th2, th3], th1-3 are three steps threshold
        factor: the factor used to create a scaling pyramid of face sizes to
                detect in the image.
        pnet, rnet, onet: caffemodel
        """
        logger.debug("Initializing: %s: (model_path: '%s', allow_growth: %s, threshold: %s)",
                     self.__class__.__name__, model_path, allow_growth, threshold)
        self.threshold = threshold
        self._image_size = None
        self._pnet_scales = None

        self.pnet = PNet(model_path[0], allow_growth)
        self.rnet = RNet(model_path[1], allow_growth)
        self.onet = ONet(model_path[2], allow_growth)
        logger.debug("Initialized: %s", self.__class__.__name__)

    def detect_faces(self, batch):
        """Detect faces in a batch of images, and returns an array of bounding boxes and 
        landmarks points for each image in the batch.

        Parameters
        ----------
        batch: :class:`numpy.ndarry`
            Array of shape 

        Returns
        -------
        boxes: :class:`numpy.ndarry`
            Array of shape 
        points: :class:`numpy.ndarry`
            Array of shape 
        """
        image_shape = batch.shape[1:3]
        if self._image_size != image_shape:
            self._image_size = image_shape
            self._pnet_scales = calculate_scales(image_shape)

        bounding_squares = self.detect_pnet(batch, image_shape)
        bounding_squares = self.detect_rnet(batch, bounding_squares, image_shape)
        bounding_squares = self.detect_onet(batch, bounding_squares, image_shape)

        boxes = [square[:5] if square.size != 0 else np.empty((0, 5)) for square in bounding_squares]
        points = [square[5:]  if square.size != 0 else np.empty((0, 5)) for square in bounding_squares]
        return boxes, points

    def detect_pnet(self, images, orig_image_shape):
        """ first stage - fast proposal network (pnet) to obtain face candidates """
        rectangles = [[] for _ in range(images.shape[0])]
        
        for scale in self._pnet_scales:
            scale_factor = images.shape[1] / scale
            batch = np.stack([cv2.resize(image, (scale, scale), interpolation=cv2.INTER_AREA)
                              for image in images])
            # optimized_batch_size = min(128, int(2**18/scale/scale))
            optimized_batch_size = 256
            classifers, regressions = self.pnet.predict(batch, batch_size=optimized_batch_size)
            longest_side = max(2, *classifers.shape[1:3])
            stride = (2.0 * longest_side - 1.0) / (longest_side - 1.0)
            for img_number, (classifer, regression) in enumerate(zip(classifers[..., 1], regressions)):
                boxes = detect_face_12net(classifer,
                                          regression,
                                          stride,
                                          scale_factor,
                                          orig_image_shape,
                                          self.threshold[0])
                rectangles[img_number].extend(boxes)
        ret = [nms(np.stack(box_list), 0.7, 'iou') if box_list else np.empty((0, 5)) for box_list in rectangles]
        #rectangles = centered_square(rectangles, orig_image_shape)
        return ret

    def detect_rnet(self, image_batch, rectangle_batch, orig_image_shape):
        """ second stage - refinement of face candidates with rnet """
        squares_list = []
        for rectangles, image in zip(rectangle_batch, image_batch):
            if rectangles.shape[0] == 0:
                squares_list.append(np.empty((0, 5)))
                continue
            predict_batch = []
            for rectangle in rectangles:
                if rectangle.size != 0:
                    int_rect = np.rint(rectangle[0:4]).astype(np.uint32)
                    crop_img = image[int_rect[1]:int_rect[3], int_rect[0]:int_rect[2]]
                    scale_img = cv2.resize(crop_img, (24, 24))
                    predict_batch.append(scale_img)
            predict_batch = np.array(predict_batch)
            classifer, regression = self.rnet.predict(predict_batch, batch_size=256)
            pre_nms_rectangles = process_results(classifer[:, 1],
                                                 regression,
                                                 None,
                                                 np.array(rectangles),
                                                 self.threshold[1],
                                                 2)
            rectangles = nms(pre_nms_rectangles, 0.7, 'iou')
            squares = centered_square(rectangles, orig_image_shape)
            squares_list.append(squares)
        return squares_list

    def detect_onet(self, image_batch, rectangle_batch, orig_image_shape):
        """ third stage - further refinement and facial landmarks positions with onet """
        squares_list = []
        for rectangles, image in zip(rectangle_batch, image_batch):
            if rectangles.shape[0] == 0:
                squares_list.append(np.empty((0, 5)))
                continue
            predict_batch = []
            for rectangle in rectangles:
                int_rect = np.rint(rectangle[0:4]).astype(np.uint32)
                crop_img = image[int_rect[1]:int_rect[3], int_rect[0]:int_rect[2]]
                scale_img = cv2.resize(crop_img, (48, 48))
                predict_batch.append(scale_img)
            predict_batch = np.array(predict_batch)
            classifer, regression, landmarks = self.onet.predict(predict_batch, batch_size=256)
            pre_nms_rectangles = process_results(classifer[:, 1],
                                                 regression,
                                                 landmarks,
                                                 np.array(rectangles),
                                                 self.threshold[2],
                                                 3)
            rectangles = nms(pre_nms_rectangles, 0.7, 'iom')
            squares = centered_square(rectangles, orig_image_shape)
            squares_list.append(squares)
        return squares_list


def sub_pixel_resize(self, src_image, dst_image_size_x, dst_image_size_y, coordinates):
    """ third stage - further refinement and facial landmarks positions with onet """
    # current non sub-pixel method
    int_rect = np.rint(coordinates).astype(np.uint32)
    cropped_img = src_image[int_rect[1]:int_rect[3], int_rect[0]:int_rect[2]]
    scaled_img = cv2.resize(cropped_img,
                            (dst_image_size_x, dst_image_size_y),
                            interpolation=cv2.INTER_AREA)

    # opencv sub-pixel method - fast
    x_map, y_map = np.meshgrid(np.linspace(coordinates[0], coordinates[2], dst_image_size_x),
                               np.linspace(coordinates[1], coordinates[3], dst_image_size_y))
    x_map, y_map = x_map.astype(np.float32), y_map.astype(np.float32)
    scaled_img = cv2.remap(isrc_image, x_map, y_map, interpolation=cv2.INTER_AREA)

    # scipy sub-pixel method - accurate
    x_map, y_map = np.meshgrid(np.linspace(coordinates[0], coordinates[2], dst_image_size_x),
                               np.linspace(coordinates[1], coordinates[3], dst_image_size_y))
    # x_map, y_map = x_map.astype(np.float32), y_map.astype(np.float32)
    scaled_img = ndimage.map_coordinates(src_image, [x_map, y_map], order=1)
    return scaled_img

def detect_face_12net(classifer, regression, stride, scale, orig_image_shape, threshold):
    """ Detect face position and calibrate bounding box on 12net feature map(matrix version)
    Input:
        classification_probability : softmax feature map for face classify
        roi                 : feature map for regression
        stride              : bounding box XXXXXXXXXXXXXXXXXXX
        scale               : current input image scale in multi-scales
        orig_image_shape    : tuple of image's original height and width
        threshold           : 0.6 can have 99% recall rate
    """
    var_y, var_x = np.where(classifer >= threshold)
    coordinates = np.nonzero(classifer >= threshold)
    scores3 = classifer[coordinates]
    scores2 = classifer[classifer >= threshold]
    scores = np.array([classifer[var_y, var_x]]).T

    boundingbox = np.array([var_x, var_y])
    print("\n same?: ", np.allclose(scores,scores2), scores.shape, scores2.shape, scores3.shape, boundingbox.shape)
    bb1 = stride * boundingbox * scale
    bb2 = bb1 + 11.0 * scale
    boundingbox = np.fix(np.concatenate((bb1, bb2), axis=0))
    offset = regression[var_x, var_y] * 12.0 * scale
    boundingbox = boundingbox.T + offset
    rectangles = np.concatenate((boundingbox, scores), axis=1)
    
    rectangles = nms(rectangles, 0.5, "iou")
    squares = centered_square(rectangles, orig_image_shape)
    return squares

def process_results(classifer, regression, landmarks, rectangles, threshold, stage_number):
    """ Process R-Net and O-Net's output

    Parameters
    ----------
    classifer: :class:`numpy.ndarry`
        classification grid with a probability that the upper left vertice is present
    regression: :class:`numpy.ndarry`
        regression with a predicted offset to the prior stage's results
    landmarks: :class:`numpy.ndarry`
        regression with the predicted coordinates of the 5 facial landmarks
    rectangles: :class:`numpy.ndarry`
        previous stage's output
    threshold: tuple of floats
        tuple containing the confidence threshold to select rectangles
    stage_number: int
        qualifier to select landmark logic chain

    Returns
    -------
    rectangles: :class:`numpy.ndarry`
        Array of shape Nx5 or Nx15 containing the rectangles vertices and face landmarks
    """
    confident_coordinates = classifer >= threshold
    rectangle_confidence = classifer[confident_coordinates]

    x_coordinates = rectangles[confident_coordinates, 0:3:2]
    y_coordinates = rectangles[confident_coordinates, 1:4:2]
    net_width = x_coordinates[:, 1:2] - x_coordinates[:, 0:1]
    net_height = y_coordinates[:, 1:2] - y_coordinates[:, 0:1]

    x_offsets = regression[confident_coordinates, 0:3:2] * net_width
    y_offsets = regression[confident_coordinates, 1:4:2] * net_height
    x_coordinates += x_offsets
    y_coordinates += y_offsets

    rectangles = np.stack([x_coordinates[:, 0], y_coordinates[:, 0],
                           x_coordinates[:, 1], y_coordinates[:, 1],
                           rectangle_confidence], axis=1)
    if stage_number == 3:
        x_landmarks = (landmarks[confident_coordinates, 0:5] * net_width + x_coordinates[:, 0:1])
        y_landmarks = (landmarks[confident_coordinates, 5:10] * net_height + y_coordinates[:, 1:2])
        rectangles = np.concatenate([rectangles,
                                     x_landmarks[:, 0:1], y_landmarks[:, 0:1],
                                     x_landmarks[:, 1:2], y_landmarks[:, 1:2],
                                     x_landmarks[:, 2:3], y_landmarks[:, 2:3],
                                     x_landmarks[:, 3:4], y_landmarks[:, 3:4],
                                     x_landmarks[:, 4:5], y_landmarks[:, 4:5]], axis=1)
    return rectangles

def nms(boxes, threshold, method='iou'):
    """ Perform Non-Maximum Suppression """
    retained_box_indices = list()

    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    ranked_indices = boxes[:, 4].argsort()[::-1]
    while ranked_indices.size > 0:
        best = ranked_indices[0]
        rest = ranked_indices[1:]

        max_of_xy = np.maximum(boxes[best, :2], boxes[rest, :2])
        min_of_xy = np.minimum(boxes[best, 2:4], boxes[rest, 2:4])
        width_height = np.maximum(0, min_of_xy - max_of_xy + 1)
        intersection_areas = width_height[:, 0] * width_height[:, 1]
        if method == 'iou':
            iou = intersection_areas / (areas[best] + areas[rest] - intersection_areas)
        else:
            iou = intersection_areas / np.minimum(areas[best], areas[rest])

        overlapping_boxes = (iou > threshold).nonzero()[0]
        if len(overlapping_boxes) != 0:
            overlap_set = ranked_indices[overlapping_boxes + 1]
            vote = np.average(boxes[overlap_set, :4], axis=0, weights=boxes[overlap_set, 4])
            boxes[best, :4] = vote
        retained_box_indices.append(best)

        remaining_boxes = (iou <= threshold).nonzero()[0]
        ranked_indices = ranked_indices[remaining_boxes + 1]
    return boxes[retained_box_indices]

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

def centered_square(rectangles, orig_image_shape):
    """ Convert axis-parallel rectangles into axis-parallel squares centered at each rectangle's
    center. Include error checking code to offset square if it goes outisde the image's size

    Parameters
    ----------
    rectangles: :class:`numpy.ndarry`
        Array of shape Nx5, with
            the first axis contains the number of bounding box candidates
            the second axis is comprised of 5 items
                x-coordinate of left side of the rectangle
                y-coordinate of top side of the rectangle
                x-coordinate of right side of the rectangle
                y-coordinate of bottom side of the rectangle
                NN model's confidence of the bounding box rectangle's accuracy
    orig_image_shape: tuple of ints
        tuple containing the original image's height and width in pixels

    Returns
    -------
    squares: :class:`numpy.ndarry`
        Array of shape Nx5 with the same parameters as rectangles
    """
    half_length = np.maximum(rectangles[:, 2] - rectangles[:, 0],
                             rectangles[:, 3] - rectangles[:, 1], dtype=np.float32) * 0.5
    center_x = np.mean(rectangles[:, 0:3:2], dtype=np.float32, axis=1)
    center_y = np.mean(rectangles[:, 1:4:2], dtype=np.float32, axis=1)

    # correct if square would go out of bounds of the image's size
    center_x = np.minimum(np.maximum(half_length, center_x), orig_image_shape[1] - half_length)
    center_y = np.minimum(np.maximum(half_length, center_y), orig_image_shape[0] - half_length)

    # move the center of the rectangle to the corrected center and calculate vertices
    squares = np.stack([center_x, center_y, center_x, center_y, rectangles[:, 4]], axis=1)
    offsets = np.stack([-half_length, -half_length, half_length, half_length], axis=1)
    squares[:, 0:4] += offsets
    return squares
