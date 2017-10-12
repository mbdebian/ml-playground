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

from numpy import unique

# Converting categorical data to numerical features
cat_data = ['male', 'female', 'male', 'male', 'female', 'male', 'female', 'female']


def category_to_numerical(data):
    categories = unique(data)
    features = []
    for category in categories:
        binary = (data == category)
        features.append(binary.astype(int))
    return features


print("----> Categorical data: {}\n\tCategories: {}\n\tNumerical conversion: {}"
      .format(cat_data,
              ",".join(unique(cat_data)),
              category_to_numerical(cat_data)))
