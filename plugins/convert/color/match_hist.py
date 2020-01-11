#!/usr/bin/env python3
""" Match histogram colour adjustment plugin for faceswap.py converter """

import numpy as np
from ._base import Adjustment


class Color(Adjustment):
    """ Match the histogram of the color intensity of each image channel """

    def process(self, old_face, new_face, raw_mask):
        """
        Calculate the histogram of each color channel in the original facial crop and the
        swapped facial crop. Shift the histogram of the swap to align with the histogram of the
        original.

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
        mask_indices = np.nonzero(raw_mask.squeeze())
        threshold = self.config["threshold"] / 100.0
        new_face_shifted = np.empty_like(new_face)
        for color_channel in [0,1,2]:
            new_face_shifted [:, :, c] = self._hist_match(old_face[:, :, color_channel],
                                                          new_face[:, :, color_channel],
                                                          mask_indices,
                                                          threshold)
        return new_face_shifted

    @staticmethod
    def _hist_match(old_channel, new_channel, mask_indices, threshold):
        """  Construct the histogram of the color intensity of a channel
             for the swap and the original. Match the histogram of the original
             by interpolation.
        """
        if mask_indices[0].size == 0:
            return new_channel

        old_masked = old_channel[mask_indices]
        new_masked = new_channel[mask_indices]
        _, bin_idx, s_counts = np.unique(new_masked, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(old_masked, return_counts=True)
        s_quants = np.cumsum(s_counts, dtype='float32')
        t_quants = np.cumsum(t_counts, dtype='float32')
        s_quants = threshold * s_quants / s_quants[-1]
        t_quants /= t_quants[-1]
        interp_s_values = np.interp(s_quants, t_quants, t_values)
        new_channel[mask_indices] = interp_s_values[bin_idx]
        return new_channel
