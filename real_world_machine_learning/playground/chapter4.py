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
import pandas
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Constants
padding = 40
total_padding = padding * 2

# Seed pseudo-random number generator
random.seed(time.time())


# Helpers

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


def area_under_the_curve(true_labels, predicted_labels, pos_class=1):
    fpr, tpr, thr = roc_curve(true_labels, predicted_labels, pos_class=pos_class)
    area = -pylab.trapz(tpr, x=fpr)
    return area


def root_mean_square_error(true_values, predicted_values):
    n = len(true_values)
    residuals = 0
    for i in range(n):
        residuals += (true_values[i] - predicted_values[i]) ** 2
    return np.sqrt(residuals / n)


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
# Randomly generated predictions should give us a diagonal ROC curve
preds = pylab.rand(len(target))
fpr, tpr, thr = roc_curve(target, preds, pos_class=True)
# pylab.plot(pylab.linspace(0, 1))
pylab.plot(fpr, tpr)
# Let's calculate the Area Under the Curve, for the ROC curve
print("+++> Area Under the Curve: {}".format(area_under_the_curve(target, preds, pos_class=True)))
print("[{} ============= {}]\n\n".format("-" * padding, "-" * padding))

# Multi-class Classification
print("[{} Multi-class Classification {}]".format("-" * padding, "-" * padding))
print("+++> Load the MNIST dataset")
mnist_dataset = pandas.read_csv("../book_code/data/mnist_small.csv")
# Hold-out method
print("+++> Holdout method for the dataset")
mnist_dataset_train = mnist_dataset[:int(0.8 * len(mnist_dataset))]
mnist_dataset_test = mnist_dataset[int(0.8 * len(mnist_dataset)):]
print("+++> Fit a Random Forest Classifier")
randomforest_classifier = RandomForestClassifier()
randomforest_classifier.fit(mnist_dataset_train.drop('label', axis=1), mnist_dataset_train['label'])
print("+++> Calculate the model predictions on the test holdout")
randomforest_classifier_predictions = randomforest_classifier.predict(mnist_dataset_test.drop('label', axis=1))
print("---> Sample predictions: {}".format(randomforest_classifier_predictions[:10]))
print("+++> Compute the Confusion Matrix")
randomforest_mnist_confusion_matrix = confusion_matrix(mnist_dataset_test['label'], randomforest_classifier_predictions)
pylab.matshow(randomforest_mnist_confusion_matrix, cmap='Greys')
pylab.colorbar()
pylab.savefig("figures/figure-4.19.eps", format='eps')
print("[{} ========================== {}]\n\n".format("-" * padding, "-" * padding))

# Root Mean Square Error
print("[{} Root Mean Square Error {}]".format("-" * padding, "-" * padding))
print("[{} ====================== {}]\n\n".format("-" * padding, "-" * padding))

# Show all plots
pylab.show()
