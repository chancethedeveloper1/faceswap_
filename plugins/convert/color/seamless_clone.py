#!/usr/bin/env python3
""" Seamless clone adjustment plugin for faceswap.py converter """

import cv2
import numpy as np
from ._base import Adjustment


class Color(Adjustment):
    """ Poisson Blending of colors along mask boundary """

    @staticmethod
    def process(old_face, new_face, raw_mask):
        """
        Seamlessly merge the swapped facial crop into the original facial crop using Poisson
        Blending. The colors along the segmentation mask boundary will be shifted to produce
        a continious transition.

        Parameters:
        -------
        old_face : Numpy array, shape (height, width, n_channels), float32
            Facial crop of the original subject
        new_face : Numpy array, shape (height, width, n_channels), float32
            Facial crop of the swapped output from the neural network
        raw_mask : Numpy array, shape (height, width, n_channels), float32
            Segmentation mask of the facial crop of the original subject

        Returns:
        -------
        new_face_shifted : Numpy array, shape (height, width, n_channels), float32
            Facial crop of the swapped output with a shifted color distribution
        """
        height, width, _ = old_face.shape
        height = height // 2
        width = width // 2

        y_indices, x_indices, _ = np.nonzero(raw_mask)
        y_crop = slice(np.min(y_indices), np.max(y_indices))
        x_crop = slice(np.min(x_indices), np.max(x_indices))
        y_center = int(np.rint((np.max(y_indices) + np.min(y_indices)) / 2 + height))
        x_center = int(np.rint((np.max(x_indices) + np.min(x_indices)) / 2 + width))

        insertion = np.rint(new_face[y_crop, x_crop] * 255.0).astype("uint8")
        insertion_mask = np.rint(raw_mask[y_crop, x_crop] * 255.0).astype("uint8")
        insertion_mask[insertion_mask != 0] = 255
        prior = np.rint(np.pad(old_face * 255.0,
                               ((height, height), (width, width), (0, 0)),
                               'constant')).astype("uint8")

        new_face_shifted = cv2.seamlessClone(insertion,
                                    prior,
                                    insertion_mask,
                                    (x_center, y_center),
                                    cv2.NORMAL_CLONE)
        new_face_shifted = new_face_shifted[height:-height, width:-width].astype("float32") / 255.0
        return new_face_shifted
