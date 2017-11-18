# 
# Author    : Manuel Bernal Llinares
# Project   : ml-playground
# Timestamp : 12-11-2017 8:57
# ---
# © 2017 Manuel Bernal Llinares <mbdebian@gmail.com>
# All rights reserved.
# 

"""
This is the playground for the practical part of the book
"""

import pylab
import numpy
import pandas


# Helpers
# Categorical-to-numerical function from chapter 2 changed to automatically add column names
def cat_to_num(data):
    categories = np.unique(data)
    features = {}
    for cat in categories:
        binary = (data == cat)
        features["{}_{}".format(data.name, cat)] = binary.astype("int")
    return pandas.DataFrame(features)


# TODO - This first part will work on TED Talks dataset
