#!/usr/bin/env python3
""" Default configurations for convert """

import logging
import os
import sys

from importlib import import_module

from lib.config import FaceswapConfig
from lib.utils import full_path_split

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Config(FaceswapConfig):
    """ Config File for Convert """

    def set_defaults(self):
        """ Set the default values for config """
        logger.debug("Setting defaults")
        self.set_globals()
        current_dir = os.path.dirname(__file__)
        for dirpath, _, filenames in os.walk(current_dir):
            default_files = [fname for fname in filenames if fname.endswith("_defaults.py")]
            if not default_files:
                continue
            base_path = os.path.dirname(os.path.realpath(sys.argv[0]))
            import_path = ".".join(full_path_split(dirpath.replace(base_path, ""))[1:])
            plugin_type = import_path.split(".")[-1]
            for filename in default_files:
                self.load_module(filename, import_path, plugin_type)

    def set_globals(self):
        """ Set the global options for convert """
        logger.debug("Setting global config")
        section = "global"
        self.add_section(title=section, info="Options for the entire conversion toolchain")
        self.add_item(
            section=section, title="overflow correction", datatype=str, default="clip",
            min_max=None, rounding=None, fixed=True, gui_radio=True,
            choices=["clip", "scale", "none"],
            info="Post-processing will commonly adjust channel intensity values outside of the "
                 "bounds of the respective color space. The above methodolody will be used to "
                 "correct these overflow values."
                 "\n\tClip: Overflow values are clamped to the min or max allowed bounds."
                 "\n\tScale: The entire color channel is scaled to ensure the min or max values "
                 "do not overflow past their allowed bounds."
                 "\n\tNone: No correction of channel values is performed. Common image display "
                 "methods will likely interptret these overflow values incorrectly.")
        self.add_item(
            section=section, title="colorspace", datatype=str, default="LAB",
            min_max=None, rounding=None, fixed=True, gui_radio=True,
            choices=["RGB", "LAB", "HSV", "YCrCb"],
            info="Transform neural network output into the above colorspace before "
                 "performing post-processing corrections."
                 "\n\t RGB: Red, Green, Blue. An additive colorspace consisting of a linear "
                 "combination of Red, Green, and Blue primaries. The color information is "
                 "separated into the three respective channels but brightness information "
                 "will be collectively encoded amongst the three correlated channels."
                 "\n\t LAB: Lightness, a, b"
                 " ... Lightness - Brightness intensity"
                 " ... A - Color range from green to red"
                 " ... B - Color range from blue to yellow"
                 " ... A transformation which is designed to approximate human vision. Aspires "
                 "to acheive perceptual uniformity as channel intensity values change. Uses a "
                 "single channel to describe brightness (L), making it very intuitive to specify "
                 "color brightness."
                 "\n\t HSV: Hue, Saturation, Value."
                 " ... Hue - Dominant wavelength or similiarity to a perceived color."
                 " ... Saturation - colorfulness of a shade relative to its own brightness."
                 " ... Value - brightness relative to the brightness of a similarly illuminated "
                 "white."
                 " ... A transformation which compromises between effectiveness for segmentation "
                 "and computational complexity. Uses a single channel to describe color (H), "
                 "making it very intuitive to specify color hue."
                 "\n\t YCrCb: Luma, Chroma red-differnce, Chroma blue-difference."
                 " ... Y - Luma component (weighted RGB without gamma correction)."
                 " ... Cr - Difference between the R channel and the Luma magnitude."
                 " ... Cb - Difference between the B channel and the Luma magnitude."
                 " ... ITU-R BT.601 conversion protocol without gamma correction. "
                 "A transformation commonly used in color video which seperates luminance and "
                 "chrominance components. Uses a single channel to describe luma (Y), "
                 "making it very intuitive to specify color brightness.")

    def load_module(self, filename, module_path, plugin_type):
        """ Load the defaults module and add defaults """
        logger.debug("Adding defaults: (filename: %s, module_path: %s, plugin_type: %s",
                     filename, module_path, plugin_type)
        module = os.path.splitext(filename)[0]
        section = ".".join((plugin_type, module.replace("_defaults", "")))
        logger.debug("Importing defaults module: %s.%s", module_path, module)
        mod = import_module("{}.{}".format(module_path, module))
        self.add_section(title=section, info=mod._HELPTEXT)  # pylint:disable=protected-access
        for key, val in mod._DEFAULTS.items():  # pylint:disable=protected-access
            self.add_item(section=section, title=key, **val)
        logger.debug("Added defaults: %s", section)
