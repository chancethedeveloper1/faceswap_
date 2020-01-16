#!/usr/bin/env python3
""" Parent class for color Adjustments for faceswap.py converter """

import logging
import numpy as np
import cv2
from matplotlib import pyplot as plt
from lib.image import batch_convert_color
from plugins.convert._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_config(plugin_name, configfile=None):
    """ Return the config for the requested model """
    return Config(plugin_name, configfile=configfile).config_dict


class Adjustment():
    """ Parent class for adjustments """
    def __init__(self, configfile=None, config=None):
        logger.debug("Initializing %s: (configfile: %s, config: %s)",
                     self.__class__.__name__, configfile, config)
        self.config = self.set_config(configfile, config)
        logger.debug("config: %s", self.config)
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_config(self, configfile, config):
        """ Set the config to either global config or passed in config """
        section = ".".join(self.__module__.split(".")[-2:])
        if config is None:
            retval = get_config(section, configfile)
        else:
            config.section = section
            retval = config.config_dict
            config.section = None
        logger.debug("Config: %s", retval)
        return retval

    def process(self, old_face, new_face, color_method):
        """ Override for specific color adjustment process """
        raise NotImplementedError

    def run(self, old_face, new_face, color_method):
        """ Perform selected adjustment on facial crops """
        logger.trace("Performing color adjustment")
        old_face_color = self.convert_colorspace(old_face[None, :, :, :3], color_method, to_bgr=False)
        new_face_color = self.convert_colorspace(new_face[None, :, :, :3], color_method, to_bgr=False)
        mask = np.concatenate((new_face[:, :, -1:], new_face[:, :, -1:], new_face[:, :, -1:]), axis=-1)[None, ...]

        new_face_shifted = self.process(old_face_color, new_face_color, mask)
        new_face_shifted = self.convert_colorspace(new_face_shifted, color_method, to_bgr=True)
        new_face_shifted = self.clip_image(new_face_shifted)
        new_face[:, :, :3] = new_face_shifted[0]

        # self._histogram_compare(old_face, new_face, new_face_shifted, mask)
        logger.trace("Performed color adjustment")
        return new_face

    def convert_colorspace(self, new_face, color_method, to_bgr=False):
        """ Convert colorspace based on mode or back to BGR """
        mode = self.config.get(color_method + "-colorspace", "Lab")
        colorspace = "YCrCb" if mode == "YCrCb" else mode.upper()
        conversion = "{}2BGR".format(colorspace) if to_bgr else "BGR2{}".format(colorspace)
        image = batch_convert_color(new_face, conversion)
        return image

    def clip_image(self, image):
        """
        Perform either clipping or min-max scaling to a NumPy array

        Parameters:
        -------
        image : Numpy array, shape (height, width, n_channels), float32
            Image data potentially ranging beyond 0.0 to 1.0

        Returns:
        -------
        image : Numpy array, shape (height, width, n_channels), float32
            Image data within the 0.0 to 1.0 range
        """
        mode = self.config.get("clip", "clip")
        if mode == "clip":
            np.clip(image, 0.0, 1.0, out=image)
        elif mode == "scale":
            image_min = np.amin(image, axis=(1, 2))
            image_max = np.amax(image, axis=(1, 2))
            image_min_clipped = np.maximum(image_min, [0.0])
            image_max_clipped = np.minimum(image_max, [1.0])

            scale_mask = (image_min < image_min_clipped | image_max > image_max_clipped)
            clip_range = image_max_clipped[scale_mask] - image_min_clipped[scale_mask]
            img_range = image_max[scale_mask] - image_min[scale_mask]
            img_adjust = image[scale_mask] - image_min[scale_mask]
            image[scale_mask] = clip_range * img_adjust / img_range + image_min_clipped[scale_mask]
        elif mode == "none":
            logger.trace("No overflow adjustment. Typically only used for raw data output.")
            # TODO parse entire convert code for clips. Preferable to use single scale/clip at end.
            # instead of the clip here in color and elsewhere...
        else:
            logger.trace("Insert Faceswap Error code here")
        return image

    @staticmethod
    def _histogram_compare(old_face, new_face, new_face_shifted, mask):
        """
        Perform either clipping or min-max scaling to a NumPy array

        Parameters:
        -------
        old_face : Numpy array, shape (n_images, height, width, n_channels), float32
            Facial crop of the original subject
        new_face : Numpy array, shape (n_images, height, width, n_channels), float32
            Facial crop of the swapped output from the neural network
        """
        images = [old_face, new_face, new_face_shifted]
        names = ['old_face', 'new_face', 'new_face_shifted']
        markers = ["None","None","*"]
        linestyles = ["solid","dotted","None"]
        colors = ["blue", "green", "red"]
        channels = range(old_face.shape[-1])
        pixel_number = old_face.shape[0] * old_face.shape[1]
        plt.figure()
        plt.title("Color Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Prob. of Pixels")
         
        mask = np.rint(mask).astype("uint8")
        for (image, marker, linestyle, name) in zip(images, markers, linestyles, names):
            for (color, channel) in zip(colors, channels):
                hist = cv2.calcHist([image], channels=[channel], mask=mask, histSize=[256], ranges=[0.0, 1.0], accumulate=False)
                hist /= pixel_number
                plt.plot(hist, color = color, marker=marker, markersize=6, markeredgecolor=color, linestyle = linestyle, label=name)
        plt.xlim([0, 256])
        plt.legend()
        plt.savefig('compare.png')
