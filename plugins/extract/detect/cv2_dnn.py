#!/usr/bin/env python3
""" OpenCV DNN Face detection plugin """

import numpy as np

from ._base import cv2, Detector, logger


class Detect(Detector):
    """ CV2 DNN detector for face recognition """
    def __init__(self, **kwargs):
        git_model_id = 4
        model_filename = ["resnet_ssd_v1.caffemodel", "resnet_ssd_v1.prototxt"]
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.name = "cv2-DNN Detector"
        self.input_size = 300
        self.vram = 0  # CPU Only. Doesn't use VRAM
        self.vram_per_batch = 0
        self.batchsize = 1
        self.confidence = self.config["confidence"] / 100

    def init_model(self):
        """ Initialize CV2 DNN Detector Model"""
        self.model = cv2.dnn.readNetFromCaffe(self.model_path[1], self.model_path[0])
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def process_input(self, batch):
        """ Compile the detection image(s) for prediction """
        batch["feed"] = cv2.dnn.blobFromImages(batch["image"],
                                               scalefactor=1.0,
                                               size=(self.input_size, self.input_size),
                                               mean=[104.0, 117.0, 123.0],
                                               swapRB=False,
                                               crop=False)
        return batch

    def predict(self, batch):
        """ Run model to get predictions """
        self.model.setInput(batch["feed"])
        predictions = self.model.forward()
        batch["prediction"] = self.finalize_predictions(predictions)
        return batch

    def finalize_predictions(self, predictions):
        """ Filter faces based on confidence level """
        predictions = np.swapaxes(predictions, 0, 2)
        pick = np.where(predictions[:, 0, 0, 2] >= self.confidence)
        boxes = predictions[pick, 0, 0, 3:7] * self.input_size
        boxes = np.concatanate((boxes, predictions[pick, 0, 0, 2]), axis=-1)
        logger.debug("Accepting %s faces over confidence %s", pick.shape[0], self.confidence)
        return [boxes]

    def process_output(self, batch):
        """ Compile found faces for output """
        return batch
