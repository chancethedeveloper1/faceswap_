#!/usr/bin/env python3
"""
    The default options for the faceswap Match_Skew Color plugin.

    Defaults files should be named <plugin_name>_defaults.py
    Any items placed into this file will automatically get added to the relevant config .ini files
    within the faceswap/config folder.

    The following variables should be defined:
        _HELPTEXT: A string describing what this plugin does
        _DEFAULTS: A dictionary containing the options, defaults and meta information. The
                   dictionary should be defined as:
                       {<option_name>: {<metadata>}}

                   <option_name> should always be lower text.
                   <metadata> dictionary requirements are listed below.

    The following keys are expected for the _DEFAULTS <metadata> dict:
        datatype:  [required] A python type class. This limits the type of data that can be
                   provided in the .ini file and ensures that the value is returned in the
                   correct type to faceswap. Valid datatypes are: <class 'int'>, <class 'float'>,
                   <class 'str'>, <class 'bool'>.
        default:   [required] The default value for this option.
        info:      [required] A string describing what this option does.
        choices:   [optional] If this option's datatype is of <class 'str'> then valid
                   selections can be defined here. This validates the option and also enables
                   a combobox / radio option in the GUI.
        gui_radio: [optional] If <choices> are defined, this indicates that the GUI should use
                   radio buttons rather than a combobox to display this option.
        min_max:   [partial] For <class 'int'> and <class 'float'> datatypes this is required
                   otherwise it is ignored. Should be a tuple of min and max accepted values.
                   This is used for controlling the GUI slider range. Values are not enforced.
        rounding:  [partial] For <class 'int'> and <class 'float'> datatypes this is
                   required otherwise it is ignored. Used for the GUI slider. For floats, this
                   is the number of decimal places to display. For ints this is the step size.
        fixed:     [optional] [train only]. Training configurations are fixed when the model is
                   created, and then reloaded from the state file. Marking an item as fixed=False
                   indicates that this value can be changed for existing models, and will override
                   the value saved in the state file with the updated value in config. If not
                   provided this will default to True.
"""


_HELPTEXT = "Options for matching the mean, variance, and skew of the pixel intensity of the "
            "original face"


_DEFAULTS = {
    "colorspace": {
        "default": "LAB",
        "choices": ["RGB", "LAB", "HSV", "YCrCb"],
        "info": "Transform neural network output into the above colorspace before "
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
                "making it very intuitive to specify color brightness.",
        "datatype": str,
        "rounding": None,
        "min_max": None,
        "gui_radio": True,
        "fixed": True,
    },
    "overflow correction": {
        "default": "clip",
        "choices": ["clip", "scale", "none"],
        "info": "Post-processing will commonly adjust channel intensity values outside of the "
                "bounds of the respective color space. The above methodolody will be used to "
                "correct these overflow values."
                "\n\tClip: Overflow values are clamped to the min or max allowed bounds."
                "\n\tScale: The entire color channel is scaled to ensure the min or max values "
                "do not overflow past their allowed bounds."
                "\n\tNone: No correction of channel values is performed. Common image display "
                "methods will likely interptret these overflow values incorrectly.",
        "datatype": str,
        "rounding": None,
        "min_max": None,
        "gui_radio": True,
        "fixed": True,
    },
}
