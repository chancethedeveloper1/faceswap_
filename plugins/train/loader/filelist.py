#!/usr/bin/env python3
""" FileList Loader """

from ._base import TrainingDataGenerator


class Loader(TrainingDataGenerator):
    """ FileList is currently identical to Base """
    def __init__(self, *args, **kwargs):  # pylint:disable=useless-super-delegation
        super().__init__(*args, **kwargs)

    # move functions unique to File List Image Loading to this sub-class
