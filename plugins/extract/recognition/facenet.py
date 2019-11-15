#!/usr/bin python3
""" Facenet inference
Model exported from: https://github.com/davidsandberg/facenet
which is based on: https://arxiv.org/abs/1503.03832

Licensed under MIT License.
Copyright (c) 2016 David Sandberg

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
 is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from lib.model.session import KSession
from lib.model.layers import L2_normalize
from ._base import Recognizer, logger


class Facenet(Recognizer):
    """ Facenet feature extraction.
        Input images should be in RGB Order """
    def __init__(self, **kwargs):
        git_model_id = 12
        model_filename = "Facenet_inception_vggface2_v1.h5"
        super().__init__(git_model_id=git_model_id, model_filename=model_filename, **kwargs)
        self.name = "Facenet"
        self.input_size = 160
        self.vram = 1 # TODO
        self.vram_warnings = 1 # TODO  # at BS 1. OOMs at higher batchsizes
        self.vram_per_batch = 1 # TODO
        self.threshold=0.4 # 0.3 to 0.6 higher excludes both some real matches and false positives
        self.batchsize = self.config["batch-size"]
        self.colorformat = "RGB"

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
        self.model = KSession(self.name, self.model_path, model_kwargs=dict())
        self.model.load_model()
        shape = (self.batchsize, self.input_size, self.input_size, 3)
        placeholder = np.zeros(shape, dtype="float32")
        self.model.predict(placeholder)

    def process_input(self, batch):
        """ Compile the detected faces for prediction """
        logger.debug("Compiling faces for prediction")
        input_ = np.array([face.feed_face[..., :3]
                           for face in batch["detected_faces"]], dtype="float32")
        batch["feed"] = (input_ - 127.5) / 128.0
        logger.trace("feed shape: %s", batch["feed"].shape)
        return batch

    def predict(self, batch):
        """ Run model to get predictions """
        logger.debug("Predicting face encoding")
        predictions = self.model.predict(batch["feed"])
        return predictions

    def process_output(self, batch):
        """ Compile face encodings for output """
        logger.debug("Processing recognition model output")
        return batch
