#!/usr/bin/env python3
""" MTCNN Face detection plugin """

from __future__ import absolute_import, division, print_function

import cv2
from keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D, Permute, PReLU

import numpy as np

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
        kwargs = {"minsize": self.config["minsize"],
                  "threshold": threshold,
                  "factor": self.config["scalefactor"]}

        if kwargs["minsize"] < 10:
            valid = False
        elif not all(0.0 < threshold <= 1.0 for threshold in kwargs['threshold']):
            valid = False
        elif not 0.0 < kwargs['factor'] < 1.0:
            valid = False

        if not valid:
            kwargs = {"minsize": 20,  # minimum size of face
                      "threshold": [0.6, 0.7, 0.7],  # three steps threshold
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
# Code adapted from: https://github.com/xiangrufan/keras-mtcnn
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


class PNet(KSession):
    """ Keras PNet model for MTCNN """
    def __init__(self, model_path, allow_growth):
        super().__init__("MTCNN-PNet", model_path, allow_growth=allow_growth)
        self.define_model(self.model_definition)
        self.load_model_weights()

    @staticmethod
    def model_definition():
        """ Keras PNetwork for MTCNN """
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


class RNet(KSession):
    """ Keras RNet model for MTCNN """
    def __init__(self, model_path, allow_growth):
        super().__init__("MTCNN-RNet", model_path, allow_growth=allow_growth)
        self.define_model(self.model_definition)
        self.load_model_weights()

    @staticmethod
    def model_definition():
        """ Keras RNetwork for MTCNN """
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


class ONet(KSession):
    """ Keras ONet model for MTCNN """
    def __init__(self, model_path, allow_growth):
        super().__init__("MTCNN-ONet", model_path, allow_growth=allow_growth)
        self.define_model(self.model_definition)
        self.load_model_weights()

    @staticmethod
    def model_definition():
        """ Keras ONetwork for MTCNN """
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
    """ MTCNN Detector for face alignment """
    # TODO Batching for rnet and onet

    def __init__(self, model_path, allow_growth, minsize, threshold, factor):
        """
        minsize: minimum faces' size
        threshold: threshold=[th1, th2, th3], th1-3 are three steps threshold
        factor: the factor used to create a scaling pyramid of face sizes to
                detect in the image.
        pnet, rnet, onet: caffemodel
        """
        logger.debug("Initializing: %s: (model_path: '%s', allow_growth: %s, minsize: %s, "
                     "threshold: %s, factor: %s)", self.__class__.__name__, model_path,
                     allow_growth, minsize, threshold, factor)
        self.minsize = minsize
        self.threshold = threshold
        self.factor = factor

        self.pnet = PNet(model_path[0], allow_growth)
        self.rnet = RNet(model_path[1], allow_growth)
        self.onet = ONet(model_path[2], allow_growth)
        self._pnet_scales = None
        logger.debug("Initialized: %s", self.__class__.__name__)

    def detect_faces(self, batch):
        """Detects faces in an image, and returns bounding boxes and points for them.
        batch: input batch
        """
        height, width = batch.shape[1:3]
        if self._pnet_scales is None:
            self._pnet_scales = calculate_scales(height, width, self.minsize, self.factor)
        rectangles = self.detect_pnet(batch, height, width)
        rectangles = self.detect_rnet(batch, rectangles, height, width)
        rectangles = self.detect_onet(batch, rectangles, height, width)
        ret_boxes = list()
        ret_points = list()
        for rects in rectangles:
            if rects.size != 0:
                total_boxes = rects[:5]
                points = np.empty(0) # rects[5:].T
            else:
                total_boxes = np.empty((0, 5))
                points = np.empty(0)
            ret_boxes.append(total_boxes)
            ret_points.append(points)
        return ret_boxes, ret_points

    def detect_pnet(self, images, height, width):
        # pylint: disable=too-many-locals
        """ first stage - fast proposal network (pnet) to obtain face candidates """
        rectangles = [[] for _ in range(images.shape[0])]
        method = cv2.INTER_AREA
        for scale in self._pnet_scales:
            batch = np.stack([cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=method)
                              for image in images])
            output = self.pnet.predict(batch)
            cls_probs = np.swapaxes(output[0][..., 1], 1, 2)
            roi_probs = np.swapaxes(output[1], 1, 3)
            longest_side = max(cls_probs.shape[1:3])
            for img_number, (cls_prob, roi_prob) in enumerate(zip(cls_probs, roi_probs)):
                boxes = detect_face_12net(cls_prob,
                                          roi_prob,
                                          longest_side,
                                          1.0 / scale,
                                          width,
                                          height,
                                          self.threshold[0])
                rectangles[img_number].extend(boxes)
        ret = [nms(np.stack(box_list), 0.7, 'iou') for box_list in rectangles]
        return ret

    def detect_rnet(self, image_batch, rectangle_batch, height, width):
        """ second stage - refinement of face candidates with rnet """
        ret = []
        for rectangles, image in zip(rectangle_batch, image_batch):
            if rectangles.size == 0:
                ret.append(np.empty(0))
                continue
            predict_batch = []
            for rectangle in rectangles:
                int_rect = np.rint(rectangle[0:4]).astype(np.uint32)
                crop_img = image[int_rect[1]:int_rect[3], int_rect[0]:int_rect[2]]
                scale_img = cv2.resize(crop_img, (24, 24))
                predict_batch.append(scale_img)
            predict_batch = np.array(predict_batch)
            cls_prob, roi_prob = self.rnet.predict(predict_batch, batch_size=128)
            ret.append(filter_face_24net(cls_prob[:, 1],
                                         roi_prob,
                                         rectangles,
                                         width,
                                         height,
                                         self.threshold[1]))
        return ret

    def detect_onet(self, image_batch, rectangle_batch, height, width):
        """ third stage - further refinement and facial landmarks positions with onet """
        ret = []
        for rectangles, image in zip(rectangle_batch, image_batch):
            if rectangles.size == 0:
                ret.append(np.empty(0))
                continue
            predict_batch = []
            for rectangle in rectangles:
                int_rect = np.rint(rectangle[0:4]).astype(np.uint32)
                crop_img = image[int_rect[1]:int_rect[3], int_rect[0]:int_rect[2]]
                scale_img = cv2.resize(crop_img, (48, 48))
                predict_batch.append(scale_img)
            predict_batch = np.array(predict_batch)
            cls_prob, roi_prob, pts_prob = self.onet.predict(predict_batch, batch_size=128)
            ret.append(filter_face_48net(cls_prob[:, 1],
                                         roi_prob,
                                         rectangles,
                                         width,
                                         height,
                                         self.threshold[2],
                                         pts_prob))
        return ret


def detect_face_12net(classification_probability, roi, out_side, scale, width, height, threshold):
    # pylint: disable=too-many-locals, too-many-arguments
    """ Detect face position and calibrate bounding box on 12net feature map(matrix version)
    Input:
        classification_probability : softmax feature map for face classify
        roi      : feature map for regression
        out_side : feature map's largest size
        scale    : current input image scale in multi-scales
        width    : image's origin width
        height   : image's origin height
        threshold: 0.6 can have 99% recall rate
    """
    stride = (2.0 * out_side - 1.0) / (out_side - 1.0) if out_side != 1 else 0.0
    (var_x, var_y) = np.where(classification_probability >= threshold)
    boundingbox = np.array([var_x, var_y])
    boundingbox = boundingbox.T
    bb1 = stride * boundingbox * scale
    bb2 = np.fix(bb1 + 11.0 * scale)
    bb1 = np.fix(bb1)
    boundingbox = np.concatenate((bb1, bb2), axis=1)
    dx_1 = roi[0][var_x, var_y]
    dx_2 = roi[1][var_x, var_y]
    dx3 = roi[2][var_x, var_y]
    dx4 = roi[3][var_x, var_y]
    score = np.array([classification_probability[var_x, var_y]]).T
    offset = np.array([dx_1, dx_2, dx3, dx4]).T
    
    dx = roi[:, var_x, var_y].T
    boundingbox = boundingbox + offset * 12.0 * scale
    rectangles = np.concatenate((boundingbox, score), axis=1)
    rectangles = centered_square(rectangles, height, width)
    retval = nms(rectangles, 0.5, "iou")
    return retval


def filter_face_24net(classification_probability, roi, rectangles, width, height, threshold):
    # pylint: disable=too-many-locals, too-many-arguments
    """ Filter face position and calibrate bounding box on 12net's output
    Input:
        classification_probability  : softmax feature map for face classify
        roi_prob  : feature map for regression
        rectangles: 12net's predict
        width     : image's origin width
        height    : image's origin height
        threshold : 0.6 can have 97% recall rate
    Output:
        rectangles: possible face positions
    """
    pick = np.where(classification_probability >= threshold)
    rectangles = np.array(rectangles)

    confidence = classification_probability[pick]
    left = rectangles[pick, 0]
    right = rectangles[pick, 2]
    top = rectangles[pick, 1]
    bot = rectangles[pick, 3]
    net_width = right - left
    net_height = bot - top

    left_offset = roi[pick, 0] * net_width
    right_offset = roi[pick, 2] * net_width
    top_offset = roi[pick, 1] * net_height
    bot_offset = roi[pick, 3] * net_height
    left += left_offset
    right += right_offset
    top += top_offset
    bot += bot_offset

    rectangles = np.concatenate((left, top, right, bot, confidence[None, :]), axis=0).T
    rectangles = centered_square(rectangles, height, width)
    retval = nms(rectangles, 0.7, 'iou')
    return retval


def filter_face_48net(classification_probability, roi, rectangles, width, height, threshold, pts):
    # pylint: disable=too-many-locals, too-many-arguments
    """ Filter face position and calibrate bounding box on 24net's output
    Input:
        classification_probability  : cls_prob[1] is face possibility
        roi       : roi offset
        rectangles: 12net's predict, rectangles[i][0:3] is the position, rectangles[i][4] is score
        width     : image's origin width
        height    : image's origin height
        threshold : 0.7 can have 94% recall rate on CelebA-database
        pts       : 5 landmark
    Output:
        rectangles: face positions and landmarks
    """
    pick = np.where(classification_probability >= threshold)
    rectangles = np.array(rectangles)

    confidence = classification_probability[pick]
    left = rectangles[pick, 0]
    right = rectangles[pick, 2]
    top = rectangles[pick, 1]
    bot = rectangles[pick, 3]
    net_width = right - left
    net_height = bot - top

    left_offset = roi[pick, 0] * net_width
    right_offset = roi[pick, 2] * net_width
    top_offset = roi[pick, 1] * net_height
    bot_offset = roi[pick, 3] * net_height
    left += left_offset
    right += right_offset
    top += top_offset
    bot += bot_offset

    left_offset = (pts[pick, 0:5] * net_width[:, :, None] + left[:, :, None]).T
    top_offset = (pts[pick, 5:10] * net_height[:, :, None] + top[:, :, None]).T
    rectangles = np.concatenate((left, top, right, bot, confidence[None, :]), axis=0).T
    """
                                 left_offset[0], top_offset[0], left_offset[1], top_offset[1],
                                 left_offset[2], top_offset[2], left_offset[3], top_offset[3],
                                 left_offset[4], top_offset[4]), axis=0).T
    """
    #rectangles = centered_square(rectangles, height, width)
    retval = nms(rectangles, 0.7, 'iom')
    return retval


def nms(boxes, threshold, method):
    # pylint:disable=too-many-locals
    """ apply NMS(non-maximum suppression) on ROIs in same scale(matrix version)
    Input:
        rectangles: rectangles[i][0:3] is the position, rectangles[i][4] is score
    Output:
        rectangles: same as input
    """
    x_1 = boxes[:, 0]
    y_1 = boxes[:, 1]
    x_2 = boxes[:, 2]
    y_2 = boxes[:, 3]
    var_s = boxes[:, 4]
    area = np.multiply(x_2-x_1+1, y_2-y_1+1)
    s_sort = np.array(var_s.argsort())
    pick = []
    while len(s_sort) > 0:
        # s_sort[-1] have highest prob score, s_sort[0:-1]->others
        xx_1 = np.maximum(x_1[s_sort[-1]], x_1[s_sort[0:-1]])
        yy_1 = np.maximum(y_1[s_sort[-1]], y_1[s_sort[0:-1]])
        xx_2 = np.minimum(x_2[s_sort[-1]], x_2[s_sort[0:-1]])
        yy_2 = np.minimum(y_2[s_sort[-1]], y_2[s_sort[0:-1]])
        width = np.maximum(0.0, xx_2 - xx_1 + 1)
        height = np.maximum(0.0, yy_2 - yy_1 + 1)
        inter = width * height
        if method == 'iom':
            var_o = inter / np.minimum(area[s_sort[-1]], area[s_sort[0:-1]])
        else:
            var_o = inter / (area[s_sort[-1]] + area[s_sort[0:-1]] - inter)
        pick.append(s_sort[-1])
        s_sort = s_sort[np.where(var_o <= threshold)[0]]
    result_rectangle = boxes[pick]
    return result_rectangle


def calculate_scales(height, width, minsize, factor):
    """ Calculate multi-scale
        Input:
            height: Original image height
            width: Original image width
            minsize: Minimum size for a face to be accepted
            factor: Scaling factor
        Output:
            scales  : Multi-scale
    """
    factor_count = 0
    min_layer = np.amin([height, width])
    var_m = 12.0 / minsize
    min_layer *= var_m
    scales = []
    while min_layer >= 12:
        scales += [var_m * np.power(factor, factor_count)]
        min_layer *= factor
        factor_count += 1
    logger.trace(scales)
    return scales

def centered_square(rectangles, height, width):
    """ Convert axis-parallel rectangles into axis-parallel squares centered at each rectangle's
    center

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

    Returns
    -------
    squares: :class:`numpy.ndarry`
        Array of shape Nx5 with the same parameters as rectangles
    """

    center_x = np.mean(rectangles[:, 0:3:2], dtype=np.float32, axis=1)
    center_y = np.mean(rectangles[:, 1:4:2], dtype=np.float32, axis=1)
    half_length = np.maximum(rectangles[:, 2] - rectangles[:, 0],
                             rectangles[:, 3] - rectangles[:, 1], dtype=np.float32) * 0.5
    squares = np.stack([center_x, center_y, center_x, center_y, rectangles[:, 4]], axis=1)
    offsets = np.stack([-half_length, -half_length, half_length, half_length], axis=1)
    squares[:, 0:4] += offsets
    squares[:, 0:4] = np.maximum(0.0, squares[:, 0:4])
    squares[:, 2] = np.minimum(width, squares[:, 2])
    squares[:, 3] = np.minimum(height, squares[:, 3])
    return squares
