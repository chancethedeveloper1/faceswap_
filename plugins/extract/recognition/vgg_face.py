#!/usr/bin python3
""" VGG_Face inference using OpenCV-DNN
Model from: https://www.robots.ox.ac.uk/~vgg/software/vgg_face/

Licensed under Creative Commons Attribution License.
https://creativecommons.org/licenses/by-nc/4.0/
"""

import cv2
import numpy as np
from lib.model.session import KSession
from ._base import Recognizer, logger


class Recognition(Recognizer):
    """ VGG Face feature extraction.
        Input images should be in BGR Order """
    def __init__(self, **kwargs):
        git_model_id = 7
        model_filename = ["vgg_face_v1.prototxt", "vgg_face_v1.caffemodel"]
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.name = "VGG Face"
        self.input_size = 224
        self.vram = 1 # TODO
        self.vram_warnings = 1 # TODO  # at BS 1. OOMs at higher batchsizes
        self.vram_per_batch = 1 # TODO
        # Average image provided in http://www.robots.ox.ac.uk/~vgg/software/vgg_face/
        self.average_img = [129.1863, 104.7624, 93.5940]
        self.threshold=0.4 # 0.3 to 0.6 higher excludes both some real matches and false positives
        self.backend = "CPU" # TODO allow GPU
        self.batchsize = 1
        # self.batchsize = self.config["batch-size"]
        
    def init_model(self):
        """ Initialize CV2 DNN Recognizer Model"""
        logger.debug("Initializing CV2 DNN recognizer model")
        self.model = cv2.dnn.readNetFromCaffe(self.model_path[0], self.model_path[1])
        if backend == "OPENCL":
            logger.info("Using OpenCL backend. You can safely ignore failure messages.")
        cv2_backend = getattr(cv2.dnn, "DNN_TARGET_{}".format(self.backend))
        self.model.setPreferableTarget(cv2_backend)

    def process_input(self, image_batch):
        """ Compile the detected faces for prediction """
        logger.debug("Compiling faces for prediction")
        processed_batch = cv2.dnn.blobFromImages(image_batch,
                                                 scalefactor=1.0,
                                                 size=(self.input_size, self.input_size),
                                                 mean=self.average_img,
                                                 swapRB=False,
                                                 crop=False)
        logger.trace("feed shape: %s", processed_batch.shape)
        return processed_batch

    def predict(self, image_batch):
        """ Return encodings for given image from vgg_face """
        logger.debug("Predicting face encoding")
        self.model.setInput(image_batch)
        predictions = self.model.forward("fc7")[0, :]
        return predictions

    def process_output(self, image_batch):
        """ Compile face encodings for output """
        logger.debug("Processing recognition model output")
        return image_batch

