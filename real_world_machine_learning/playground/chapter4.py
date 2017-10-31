# 
# Author    : Manuel Bernal Llinares
# Project   : ml-playground
# Timestamp : 26-10-2017 9:50
# ---
# © 2017 Manuel Bernal Llinares <mbdebian@gmail.com>
# All rights reserved.
# 

"""
Scratchpad / playground for the 4th chapter on the book
"""

import math
import time
import pylab
import random
import numpy as np

# Seed pseudo-random number generator
random.seed(time.time())

print("+++> Create a random 100x5 Matrix")
features = pylab.rand(100, 5)
print("+++> Create the prediction target")
target = pylab.rand(100) > 0.5

# First we're going to try the holdout method
print("[{} Holdout Method {}]".format("-" * 20, "-" * 20))
n = features.shape[0]
n_train = math.floor(0.7 * n)
# Randomize index
# Note: sometimes you want to retain the order in the dataset and skip this step, e.g. in the case of time-based
# datasets where you want to test on 'later' instances
print("+++> Create a random permutation of the dataset")
idx = np.random.permutation(n)
# Split the index
print("+++> Split the dataset (train / test)")
idx_train = idx[:n_train]
idx_test = idx[n_train:]
# Break your data into training and testing subsets
print("+++> Break your data into training and testing subsets")
features_train = features[idx_train,:]
target_train = target[idx_train]
features_test = features[idx_test,:]
target_test = target[idx_test]
# Log the aspect of the data
print("---> Train dataset shape {}".format(features_train.shape))
print("---> Test dataset shape {}".format(features_test.shape))
print("---> Target data for the training dataset, shape {}".format(target_train.shape))
print("---> Target data for the test dataset, shape {}".format(target_test.shape))
# ---- And... that was it, according to the example
print("-" * 80)

# K-fold cross-validation
# Number of items in the dataset
n = features.shape[0]
# Number of folds
k_folds = 10
# Un-initialized array with as many elements as elements in the dataset
preds_kfold = np.empty(n)
folds = np.random.randint(0, k_folds, size=n)

for idx in np.arange(k_folds):
    pass