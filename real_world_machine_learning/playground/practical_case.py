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

def roc_curve(true_labels, predicted_probs, n_points=100, pos_class=1):
    # Initialization
    # Reference line (this one is easy)
    thr = pylab.linspace(0, 1, n_points)
    # I guess this is about true possitive rate
    tpr = pylab.zeros(n_points)
    # I guess this is about the false possitive rate
    fpr = pylab.zeros(n_points)
    # What the fuck is this, why the sample code is so fucking obscure?
    # Possitive and negative vectors?
    pos = true_labels == pos_class
    neg = np.logical_not(pos)
    # Count possitives and negatives
    n_pos = np.count_nonzero(pos)
    n_neg = np.count_nonzero(neg)
    # Calculate tpr and fpr for every position of the curve
    for index, value in enumerate(thr):
        tpr[index] = np.count_nonzero(np.logical_and(predicted_probs >= value, pos)) / n_pos
        fpr[index] = np.count_nonzero(np.logical_and(predicted_probs >= value, neg)) / n_neg
    return fpr, tpr, thr

# TODO - This first part will work on TED Talks dataset
