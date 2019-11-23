#!/usr/bin python3
""" VGG_Face2 inference
Model exported from: https://github.com/WeidiXie/Keras-VGGFace2-ResNet50
which is based on: https://www.robots.ox.ac.uk/~vgg/software/vgg_face/

Licensed under Creative Commons Attribution License.
https://creativecommons.org/licenses/by-nc/4.0/
"""

import numpy as np
from lib.model.session import KSession
from lib.model.layers import L2_normalize
from ._base import Recognizer, logger


class Recognition(Recognizer):
    """ VGG Face feature extraction.
        Input images should be in BGR Order """
    def __init__(self, **kwargs):
        git_model_id = 10
        model_filename = "vggface2_resnet50_v2.h5"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.name = "VGG Face2"
        self.input_size = 224
        self.vram = 1 # TODO
        self.vram_warnings = 1 # TODO  # at BS 1. OOMs at higher batchsizes
        self.vram_per_batch = 1 # TODO
        # Average image provided in https://github.com/ox-vgg/vgg_face2
        self.average_img = np.array([91.4953, 103.8827, 131.0912])
        self.threshold=0.4 # 0.3 to 0.6 higher excludes both some real matches and false positives
        self.batchsize = 1
        # self.batchsize = self.config["batch-size"]

    def init_model(self):
        """ Initialize Keras VGG Face2 Recognizer Model"""
        logger.debug("Initializing Keras VGG Face2 recognizer model")
        # TODO configure Ksession for CPU
        '''
        if backend == "CPU":
            if os.environ.get("KERAS_BACKEND", "") == "plaidml.keras.backend":
                logger.info("Switching to tensorflow backend.")
                os.environ["KERAS_BACKEND"] = "tensorflow"
            with keras.backend.tf.device("/cpu:0"):
        '''
        self.model = KSession(self.name,
                              self.model_path,
                              model_kwargs={"L2_normalize":  L2_normalize})
        self.model.load_model()
        shape = (self.batchsize, self.input_size, self.input_size, 3)
        placeholder = np.zeros(shape, dtype="float32")
        self.model.predict(placeholder)

    def process_input(self, image_batch):
        """ Compile the detected faces for prediction """
        logger.debug("Compiling faces for prediction")
        processed_batch = image_batch - self.average_img
        logger.trace("feed shape: %s", processed_batch.shape)
        return processed_batch

    def predict(self, image_batch):
        """ Run model to get predictions """
        logger.debug("Predicting face encoding")
        predictions = self.model.predict(image_batch)[0, :]
        return predictions

    def process_output(self, image_batch):
        """ Compile face encodings for output """
        logger.debug("Processing recognition model output")
        return image_batch
