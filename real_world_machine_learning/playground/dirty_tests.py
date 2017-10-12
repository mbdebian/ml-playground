# 
# Author    : Manuel Bernal Llinares
# Project   : ml-playground
# Timestamp : 13-09-2017 19:28
# ---
# © 2017 Manuel Bernal Llinares <mbdebian@gmail.com>
# All rights reserved.
# 

"""
This is a script for quick and dirty tests on machine learning concepts

(Working with code from Chapter 2 in the book)
"""

import numpy as np

# Converting categorical data to numerical features
cat_data = np.array(['male', 'female', 'male', 'male', 'female', 'male', 'female', 'female'])


def category_to_numerical(data):
    categories = np.unique(data)
    features = []
    for category in categories:
        binary = (data == category)
        features.append((category, binary.astype("int")))
    return features


print("----> Categorical data: {}\n\tCategories: {}\n\tNumerical conversion: {}"
      .format(cat_data,
              ",".join(np.unique(cat_data)),
              category_to_numerical(cat_data)))

# Simple feature engineering of the Titanic data
cabin_data = np.array(["C65", "", "E36", "C54", "B57 B59 B63 B66"])


def feature_engineering_cabin_features(data):
    features = []
    for cabin in data:
        cabins = cabin.split(" ")
        n_cabins = len(cabins)
        try:
            cabin_char = cabins[0][0]
        except IndexError:
            cabin_char = "X"
            n_cabins = 0
        # The rest is the cabin number
        try:
            cabin_num = int(cabins[0][1:])
        except:
            cabin_num = -1
        # Add 3 features for each passenger
        features.append([cabin_char, cabin_num, n_cabins])
    return features


print("---> Feature Engineering of '{}' into\n\t-> {}"
      .format(cabin_data,
              ",".join(feature_engineering_cabin_features(cabin_data))))
