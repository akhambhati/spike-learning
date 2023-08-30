#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" utils.py
Description: Learn network representations of spiking events from intracranial EEG.
"""
__author__ = "Ankit N. Khambhati"
__copyright__ = "Copyright 2021, Ankit N. Khambhati"
__credits__ = ["Ankit N. Khambhati"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = "Ankit N. Khambhati"
__email__ = ""
__status__ = "Prototype"


import hashlib
import json


def dict_hash(dictionary):
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()
