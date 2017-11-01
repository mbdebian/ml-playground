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

# Constants
padding = 40
total_padding = padding * 2

# Seed pseudo-random number generator
random.seed(time.time())

print("+++> Create a random 100x5 Matrix")
features = pylab.rand(100, 5)
print("+++> Create the prediction target")
target = pylab.rand(100) > 0.5

# First we're going to try the holdout method
print("[{} Holdout Method {}]".format("-" * padding, "-" * padding))
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
features_train = features[idx_train, :]
target_train = target[idx_train]
features_test = features[idx_test, :]
target_test = target[idx_test]
# Log the aspect of the data
print("---> Train dataset shape {}".format(features_train.shape))
print("---> Test dataset shape {}".format(features_test.shape))
print("---> Target data for the training dataset, shape {}".format(target_train.shape))
print("---> Target data for the test dataset, shape {}".format(target_test.shape))
# ---- And... that was it, according to the example
# print("-" * total_padding)

# K-fold cross-validation
# Number of items in the dataset
n = features.shape[0]
# Number of folds
k_folds = 10
# Un-initialized array with as many elements as elements in the dataset
preds_kfold = np.empty(n)
folds = np.random.randint(0, k_folds, size=n)

for idx in np.arange(k_folds):
    # For each fold, break your data into training and testing subsets
    features_train = features[folds != idx, :]
    target_train = target[folds != idx]
    features_test = features[folds == idx, :]
    # I don't really need to compute the target_test, I can do it on the fly later
    target_test = target[folds == idx]
    # Print the indices in each fold, for inspection
    print("Fold for index #{}: {}".format(idx, np.nonzero(folds == idx)[0]))
    # This part is about the model
    # Build and predict for CV fold (to be filled out)
    # model = train(features_train, target_train)
    # preds_kfold[folds == idx] = predict(model, features_test)
# Measure model accuracy
# accuracy = evaluate_acc(preds_kfold, target)
print("[{} ============= {}]\n\n".format("-" * padding, "-" * padding))

# The ROC Curve
print("[{} The ROC Curve {}]".format("-" * padding, "-" * padding))


def roc_curve(true_labels, predicted_probe, n_points=100, pos_class=1):
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
    




print("[{} ============= {}]\n\n".format("-" * padding, "-" * padding))
