#!/usr/bin/env python3
""" Color Transfer adjustment color matching adjustment plugin for faceswap.py converter """

import numpy as np
from ._base import Adjustment


class Color(Adjustment):
    """ Color distribution matching
    Builds on "Color Transfer between Images" paper by Reinhard et al., 2001.

    Additionally, each image's statistics will be weighted by the facial segmentation mask.
    Statisical matching is available in several colorspaces and clipping is employed to
    prevent overflow.
    """

    @staticmethod
    def process(old_face, new_face, raw_mask):
        """
        Match the 1st and 2nd moments of the color distribution from the original facial crop
        by adjusting the distribution in the swapped facial crop.

        Parameters:
        -------
        old_face : Numpy array, shape (n_images, height, width, n_channels), float32
            Facial crop of the original subject
        new_face : Numpy array, shape (n_images, height, width, n_channels), float32
            Facial crop of the swapped output from the neural network
        raw_mask : Numpy array, shape (n_images, height, width, n_channels), float32
            Segmentation mask of the facial crop of the original subject

        Returns:
        -------
        new_face_shifted : Numpy array, shape (n_images, height, width, n_channels), float32
            Facial crop of the swapped output with a shifted color distribution
        """
        old_mean = np.average(old_face, axis=(1, 2), weights=raw_mask)
        new_mean = np.average(new_face, axis=(1, 2), weights=raw_mask)
        old_std = np.sqrt(np.average((old_face - old_mean)**2, axis=(1, 2), weights=raw_mask))
        new_std = np.sqrt(np.average((new_face - new_mean)**2, axis=(1, 2), weights=raw_mask))

        # there is no "preserve paper" as Reinhard's math always uses (old_face_std / new_face_std)
        # must have been confusion due to the terminology of "source" and "target" in the paper
        new_face_shifted = (new_face - new_mean) * (old_std / new_std) + (old_mean)
        return new_face_shifted
