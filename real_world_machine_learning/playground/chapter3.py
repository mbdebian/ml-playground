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


# Helper functions
# Categorical-to-numerical function from chapter 2 changed to automatically add column names
def cat_to_num(data):
    categories = np.unique(data)
    features = {}
    for cat in categories:
        binary = (data == cat)
        features["{}={}".format(data.name, cat)] = binary.astype("int")
    return pandas.DataFrame(features)


# Read the Titanic sample data
sample_data_titanic = pandas.read_csv("../book_code/data/titanic.csv")
print("---> Sample data - Titanic, #{} entries".format(len(sample_data_titanic)))
print("... Sample ...\n"
      "{}\n"
      "... END of Sample ...".format(sample_data_titanic[:5]))

# We make a 80/20% train/test split of the data
data_train = sample_data_titanic[:int(0.8 * len(sample_data_titanic))]
data_test = sample_data_titanic[int(0.8 * len(sample_data_titanic))]