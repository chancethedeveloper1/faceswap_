#!/usr/bin/env python3
""" Power transform colour adjustment plugin for faceswap.py converter """

import numpy as np
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
        old_face : Numpy array, shape (height, width, n_channels), float32
            Facial crop of the original subject whose color distributions are to be matched
        new_face : Numpy array, shape (height, width, n_channels), float32
            Facial crop of the swapped output from the neural network
        raw_mask : Numpy array, shape (height, width, n_channels), float32
            Segmentation mask of the facial crop of the original subject

        Returns:
        -------
        new_face_shifted : Numpy array, shape (height, width, n_channels), float32
            Facial crop of the swapped output with a shifted color distribution
        """
        old_face = old_face[None, :, :, :]
        old_face = new_face[None, :, :, :]
        source_lambdas = self._yeo_johnson_optimize(old_face)
        swapped_lambdas = self._yeo_johnson_optimize(new_face)

        normalized_swaps = self._yeo_johnson_transform(new_face, swapped_lambdas)
        new_face_shifted = self._yeo_johnson_inverse_transform(normalized_swaps, source_lambdas)

        return new_face_shifted

    @staticmethod
    def _yeo_johnson_optimize(images):
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
        half_n_pixels = images.shape[1] * images.shape[2] * 0.5
        n_images, n_channels = images.shape[0], images.shape[3]
        sign_images = np.sign(images)
        log_abs_images = np.log1p(np.abs(images))
        term_images = np.sum(sign_images * log_abs_images, axis=(1,2))

        def _neg_log_likelihood(lmbdas, image_data):
            """Return the negative log likelihood of the image data as a
            function of the lambda values."""
            lmbdas_shaped = lmbdas.reshape(n_images, n_channels)
            transformed = self._yeo_johnson_transform(image_data, lmbdas_shaped)
            neg_log_likelihood = np.log(np.var(transformed, axis=(1,2)))
            neg_log_likelihood *= half_n_pixels
            neg_log_likelihood += (term_images * (1.0 - lmbdas_shaped))
            total_neg_log_likelihood = np.sum(neg_log_likelihood)
            return total_neg_log_likelihood

        lower_bound = np.full_like(term_images, -2.0)
        upper_bound = np.full_like(term_images, 2.0
        initial_guess = np.full_like(term_images, 0.1)
        bounds = scipy.optimize.Bounds(lower_bound, upper_bound)
        optimized = scipy.optimize.minimize(_neg_log_likelihood,
                                            initial_guess,
                                            args=(images,),
                                            method='L-BFGS-B',
                                            bounds=bounds)
        lmbdas = optimized.x.reshape(n_images, n_channels)
        return lmbdas

    @staticmethod
    def _yeo_johnson_transform(images, lmbdas):
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
        epsilon = np.finfo(np.float64).eps
        lmbdas = lmbdas[:, None, None, :]

        positive_masks = (images >= 0.0)
        zero_lambdas = (np.abs(lmbdas) < epsilon)
        two_lambdas = (np.abs(lmbdas - 2.0) < epsilon)
        mask_a = positive_masks & zero_lambdas
        mask_b = positive_masks & ~zero_lambdas
        mask_c = ~positive_masks & two_lambdas
        mask_d = ~positive_masks & ~two_lambdas
 
        normed_images[mask_a] = np.log1p(images[mask_a])
        normed_images[mask_c] = -np.log1p(-images[mask_c])
        normed_images[mask_b] = (np.power(images[mask_b] + 1.0, lmbdas) - 1.0) / lmbdas
        normed_images[mask_d] = -(np.power(-images[mask_d] + 1.0, 2.0 - lmbdas) - 1.0) / (2.0 - lmbdas)

        normed_images -= np.mean(normed_images, axis=(1,2))
        normed_images /= np.std(normed_images, axis=(1,2))

        return normed_images

    @staticmethod
    def _yeo_johnson_inverse_transform(normed_images, lmbdas):
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
        epsilon = np.finfo(np.float64).eps
        lmbdas = lmbdas[:, None, None, :]

        normed_images *= np.std(normed_images, axis=(1,2))
        normed_images += np.mean(normed_images, axis=(1,2))

        positive_masks = (normed_images >= 0.0)
        zero_lambdas = (np.abs(lmbdas) < epsilon)
        two_lambdas = (np.abs(lmbdas - 2.0) < epsilon)
        mask_a = positive_masks & zero_lambdas
        mask_b = positive_masks & ~zero_lambdas
        mask_c = ~positive_masks & two_lambdas
        mask_d = ~positive_masks & ~two_lambdas
 
        images[mask_a] = np.expm1(normed_images[mask_a])
        images[mask_c] = -np.expm1(-normed_images[mask_c])
        images[mask_b] = np.power(x[mask_b * lmbdas + 1.0, 1.0 / lmbdas) - 1.0
        images[mask_d] = 1.0 - np.power((-2.0 + lmbdas) * x[mask_d] + 1.0, 1.0 / (2.0 - lmbdas))

        return images
