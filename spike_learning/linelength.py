#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" linelength.py
Description: Calculate line-length feature from intracranial EEG.
"""
__author__ = "Ankit N. Khambhati"
__copyright__ = "Copyright 2021, Ankit N. Khambhati"
__credits__ = ["Ankit N. Khambhati"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = "Ankit N. Khambhati"
__email__ = ""
__status__ = "Prototype"


import numpy as np


def LineLength(pp_data):
    pp_LL = {}
    for key in pp_data:
        try:
            pp_LL[key] = pp_data[key].copy()
        except:
            pp_LL[key] = pp_data[key]

    pp_LL['data'] = np.diff(pp_LL['data'], axis=0)**2
    pp_LL['timestamp vector'] = pp_LL['timestamp vector'][1:]

    return pp_LL
