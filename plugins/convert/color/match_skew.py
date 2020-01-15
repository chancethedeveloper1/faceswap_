#!/usr/bin/env python3
""" Power transform colour adjustment plugin for faceswap.py converter """

import numpy as np
import scipy.optimize as opt
from ._base import Adjustment


class Color(Adjustment):
    """ Color distribution shifting using an optimized power transform """

    def process(self, old_face, new_face, raw_mask):
        """
        Remap swapped_images to the source_images' color distribution accounting for the skewed
        color distribution
        # Currently designed for batches of images and no masks
        Parameters:
        -------
        old_face : Numpy array, shape (n_images, height, width, n_channels), float32
            Facial crop of the original subject whose color distributions are to be matched
        new_face : Numpy array, shape (n_images, height, width, n_channels), float32
            Facial crop of the swapped output from the neural network
        raw_mask : Numpy array, shape (n_images, height, width, n_channels), float32
            Segmentation mask of the facial crop of the original subject

        Returns:
        -------
        new_face_shifted : Numpy array, shape (height, width, n_channels), float32
            Facial crop of the swapped output with a shifted color distribution
        """
        channels = range(new_face.shape[-1])
        old_faces = np.concatenate([old_face[..., channel] for channel in channels], axis=0)
        new_faces = np.concatenate([new_face[..., channel] for channel in channels], axis=0)
        masks = np.concatenate([raw_mask[..., channel] for channel in channels], axis=0)
        new_face_shifted = new_faces.copy()
        for index, (old, new, mask) in enumerate(zip(old_faces, new_faces, masks)):
            select = (mask >= 0.5)
            source_lambdas = self._yeo_johnson_optimize(old[select])
            swapped_lambdas = self._yeo_johnson_optimize(new[select])
            origs = self._yeo_johnson_transform(old[select], source_lambdas)
            swaps = self._yeo_johnson_transform(new[select], swapped_lambdas)
            norm_swaps = (swaps - np.mean(swaps)) / np.std(swaps)
            shifted_swaps = norm_swaps * np.std(origs) + np.mean(origs)
            new_face_shifted[index][select] = self._yeo_johnson_inverse_transform(shifted_swaps,
                                                                                  source_lambdas)
        new_face_shifted = np.stack(new_face_shifted, axis=-1)[None, ...]
        return new_face_shifted

    def _yeo_johnson_optimize(self, images):
        """Estimate the parameter lambda of the Yeo-Johnson transform for each feature. Each
        image color channel is an independent feature. The lambda parameter for minimizing
        skewness is estimated on each feature independently using maximum likelihood.

        Parameters
        ----------
        images : Numpy array, shape (n_images, height, width, n_channels)
            The data to be transformed using a power transformation.

        Returns
        -------
        lmbdas : Numpy array, shape (n_images, n_channels)
            The optimized lambdas to normalize the image data
        """
        half_n_pixels = images.shape[0] * -0.5
        term_images = np.sum(np.sign(images) * np.log1p(np.abs(images)))

        def _neg_log_likelihood(lmbdas):
            """Return the negative log likelihood of the image data as a
            function of the lambda values."""
            transformed = self._yeo_johnson_transform(images, lmbdas)
            neg_log_likelihood = half_n_pixels * np.log(np.var(transformed))
            neg_log_likelihood += (lmbdas - 1.0) * term_images
            return -neg_log_likelihood

        # lower_bound = np.full_like(term_images, -2.0)
        # upper_bound = np.full_like(term_images, 2.0)
        # initial_guess = np.full_like(term_images, 0.5)
        # bounds = opt.Bounds(lower_bound, upper_bound)
        optimized = opt.brent(_neg_log_likelihood, brack=(-3.0, 3.0))
        # lmbdas = optimized.x
        # return lmbdas
        return optimized

    @staticmethod
    def _yeo_johnson_transform(images, lmbda):
        """Return transformed input images using the Yeo-Johnson transform with
        parameter lambdas.

        Parameters
        ----------
        images : Numpy array, shape (n_images, height, width, n_channels)
            The data to be transformed using a power transformation.
        lmbdas : Numpy array, shape (n_images, n_channels)
            The optimized lambdas to normalize the image data

        Returns
        -------
        normed_images : Numpy array, shape (n_images, height, width, n_channels)
            The transformed data.
        """
        normed_images = np.empty_like(images)
        epsilon = np.finfo(np.float32).eps
        # lmbdas = lmbda[None, None, :] * np.ones_like(images)
        lmbdas = lmbda * np.ones_like(images)

        positive_masks = (images >= 0.0)
        zero_lambdas = (np.abs(lmbdas) < epsilon)
        two_lambdas = (np.abs(lmbdas - 2.0) > epsilon)
        mask_a = positive_masks & zero_lambdas
        mask_b = positive_masks & ~zero_lambdas
        mask_c = ~positive_masks & ~two_lambdas
        mask_d = ~positive_masks & two_lambdas

        normed_images[mask_a] = np.log1p(images[mask_a])
        normed_images[mask_c] = -np.log1p(-images[mask_c])
        normed_images[mask_b] = (np.power(images[mask_b] + 1.0, lmbda) - 1.0) / lmbda
        normed_images[mask_d] = -(np.power(-images[mask_d] + 1.0, 2. - lmbda) - 1.0) / (2. - lmbda)
        return normed_images

    @staticmethod
    def _yeo_johnson_inverse_transform(normed_images, lmbda):
        """Return inverse-transformed input images using the Yeo-Johnson inverse
        transform with parameter lambdas.

        Parameters
        ----------
        normed_images : Numpy array, shape (n_images, height, width, n_channels)
            The transformed data.
        lmbdas : Numpy array, shape (n_images, n_channels)
            The optimized lambdas to normalize the image data

        Returns
        -------
        images : Numpy array, shape (n_images, height, width, n_channels)
            The data inverted back to the original distribution
        """
        images = np.empty_like(normed_images)
        epsilon = np.finfo(np.float32).eps
        # lmbdas = lmbda[None, None, :] * np.ones_like(images)
        lmbdas = lmbda * np.ones_like(normed_images)

        positive_masks = (normed_images >= 0.0)
        zero_lambdas = (np.abs(lmbdas) < epsilon)
        two_lambdas = (np.abs(lmbdas - 2.0) > epsilon)
        mask_a = positive_masks & zero_lambdas
        mask_b = positive_masks & ~zero_lambdas
        mask_c = ~positive_masks & ~two_lambdas
        mask_d = ~positive_masks & two_lambdas

        stable = (-2.0 + lmbda) * normed_images[mask_d] + 1.0
        images[mask_a] = np.expm1(normed_images[mask_a])
        images[mask_c] = -np.expm1(-normed_images[mask_c])
        images[mask_b] = np.power(normed_images[mask_b] * lmbda + 1.0, 1.0 / lmbda) - 1.0
        images[mask_d] = 1.0 - np.power(np.abs(stable), 1.0 / (2.0 - lmbda)) * np.sign(stable)
        return images
