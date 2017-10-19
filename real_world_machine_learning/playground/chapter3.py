# 
# Author    : Manuel Bernal Llinares
# Project   : ml-playground
# Timestamp : 18-10-2017 16:11
# ---
# © 2017 Manuel Bernal Llinares <mbdebian@gmail.com>
# All rights reserved.
# 

"""
Scratchpad for the chapter 3 from the book
"""

import pandas
import numpy as np
import math


# Helper functions
# Categorical-to-numerical function from chapter 2 changed to automatically add column names
def cat_to_num(data):
    categories = np.unique(data)
    features = {}
    for cat in categories:
        binary = (data == cat)
        features["{}_{}".format(data.name, cat)] = binary.astype("int")
    return pandas.DataFrame(features)


def prepare_data(data):
    """
    Takes a dataframe of raw data and returns ML model features
    :param data: dataframe of raw data
    :return: ML model features
    """
    # Initially, we build a model only no the available numerical values
    features = data.drop(["PassengerId", "Survived", "Fare", "Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1)
    # Setting missing age values to -1
    features["Age"] = data["Age"].fillna(-1)
    # Adding the sqrt of the fare feature
    features["sqrt_Fare"] = math.sqrt(data["Fare"])
    # Adding gender categorical value
    features = features.join(cat_to_num(data["Sex"]))
    # Adding Embarked categorical value
    features = features.join(cat_to_num(data["Embarked"]))
    # ML model features are now ready
    return features


# Read the Titanic sample data
sample_data_titanic = pandas.read_csv("../book_code/data/titanic.csv")
print("---> Sample data - Titanic, #{} entries".format(len(sample_data_titanic)))
print("... Sample ...\n"
      "{}\n"
      "... END of Sample ...".format(sample_data_titanic[:5]))

# We make a 80/20% train/test split of the data
data_train = sample_data_titanic[:int(0.8 * len(sample_data_titanic))]
data_test = sample_data_titanic[int(0.8 * len(sample_data_titanic))]

# ML training model
ml_training_model = prepare_data(data_train)
print("---> ML Training model\n"
      "... SAMPLE ...\n"
      "{}\n"
      "... END of SAMPLE".format(ml_training_model[:5]))
