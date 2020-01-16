#!/usr/bin/env python3
""" Manual Balance colour adjustment plugin for faceswap.py converter """

import numpy as np
from ._base import Adjustment


class Color(Adjustment):
    """ Color distribution adjustment """

    def process(self, old_face, new_face, mask):
        """
        Shift the color distribution of the swapped facial crop by an user input amount.

        Parameters:
        -------
        old_face : Numpy array, shape (n_images, height, width, n_channels), float32
            Facial crop of the original subject
        new_face : Numpy array, shape (n_images, height, width, n_channels), float32
            Facial crop of the swapped output from the neural network
        mask : Numpy array, shape (n_images, height, width, n_channels), float32
            Segmentation mask of the facial crop of the original subject

        Returns:
        -------
        new_face_shifted : Numpy array, shape (n_images, height, width, n_channels), float32
            Facial crop of the swapped output with a shifted color distribution
        """
        adjustment = np.array([self.config["balance_1"],
                               self.config["balance_2"],
                               self.config["balance_3"]])[None, None, None, :]
        pos_mask = np.nonzero(new_face >= 0.0)
        neg_mask = np.nonzero(new_face < 0.0)
        new_face[pos_mask] = ((1.0 - new_face) * adjustment + new_face)[pos_mask]
        new_face[neg_mask] = (new_face * (1.0 + adjustment))[neg_mask]
        new_face_shifted = self._adjust_contrast(new_face)
        return new_face_shifted

    def _adjust_contrast(self, image):
        """ Adjust image's contrast and brightness. """
        if not self.config["contrast"] and not self.config["brightness"]:
            return image

        contrast = max(-126.0, self.config["contrast"] * 127.0)
        brightness = max(-126.0, self.config["brightness"] * 127.0)
        adj_brightness = brightness - contrast
        adj_contrast = contrast / 127.0 + 1.0
        image *= adj_contrast
        image += adj_brightness
        return image
