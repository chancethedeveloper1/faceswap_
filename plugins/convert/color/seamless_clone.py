#!/usr/bin/env python3
""" Seamless clone adjustment plugin for faceswap.py converter """

import cv2
import numpy as np
from ._base import Adjustment


class Color(Adjustment):
    """ Poisson Blending of colors along mask boundary """

    def process(self, old_face, new_face, mask):
        """
        Seamlessly merge the swapped facial crop into the original facial crop using Poisson
        Blending. The colors along the segmentation mask boundary will be shifted to produce
        a continious transition.

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
        new_face_shifted = np.empty_like(new_face)
        for index, (old_img, new_img, img_mask) in enumerate(zip(old_face, new_face, mask)):
            new_face_shifted[index] = self._seamless_clone(old_img, new_img, img_mask)
        return new_face_shifted

    @staticmethod
    def _seamless_clone(old_face, new_face, mask):
        """  Clone each image seperately """
        height, width, _ = old_face.shape
        y_indices, x_indices, _ = np.nonzero(mask)
        y_crop = slice(np.min(y_indices), np.max(y_indices))
        x_crop = slice(np.min(x_indices), np.max(x_indices))
        y_center = int(np.rint((np.max(y_indices) + np.min(y_indices) + height) / 2.0))
        x_center = int(np.rint((np.max(x_indices) + np.min(x_indices) + width) / 2.0))

        insertion_image = np.rint(new_face[y_crop, x_crop] * 255.0).astype("uint8")
        insertion_mask = np.rint(mask[y_crop, x_crop] * 255.0).astype("uint8")
        insertion_mask[insertion_mask != 0] = 255
        height = height // 2
        width = width // 2
        background_image = np.rint(np.pad(old_face * 255.0,
                                          ((height, height), (width, width), (0, 0)),
                                          'constant')).astype("uint8")

        padded_image = cv2.seamlessClone(insertion_image,
                                         background_image,
                                         insertion_mask,
                                         (x_center, y_center),
                                         cv2.NORMAL_CLONE)
        new_face_shifted = padded_image[height:-height, width:-width].astype("float32") / 255.0
        return new_face_shifted
