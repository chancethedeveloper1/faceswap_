#!/usr/bin/env python3
""" Dataset Loader """

from ._base import TrainingDataGenerator


class Loader(TrainingDataGenerator):
    """ Dataset is currently identical to Base """
    def __init__(self, *args, **kwargs):  # pylint:disable=useless-super-delegation
        super().__init__(*args, **kwargs)

    # move functions unique to Dataset Image Loading to this sub-class

