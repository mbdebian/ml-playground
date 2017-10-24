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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


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
    features_drop_list = ["PassengerId", "Survived", "Fare", "Name", "Sex", "Ticket", "Cabin", "Embarked"]
    print("+++> Drop features: {}".format(features_drop_list))
    features = data.drop(features_drop_list, axis=1)
    # Setting missing age values to -1
    print("+++> Fix missing 'Age' values, filling them with '-1'")
    features["Age"] = data["Age"].fillna(-1)
    # Adding the sqrt of the fare feature
    print("+++> Change 'Fare' for its square root value")
    features["sqrt_Fare"] = np.sqrt(data["Fare"])
    # Adding gender categorical value
    print("+++> Convert 'Sex' categorical data")
    features = features.join(cat_to_num(data["Sex"]))
    # Adding Embarked categorical value
    print("+++> Convert 'Embarked' categorical data")
    features = features.join(cat_to_num(data["Embarked"]))
    # ML model features are now ready
    return features


# Read the Titanic sample data
# The data is English localized, but pandas will fail trying to convert numbers into float because this system is
# Spanish localized
sample_data_titanic = pandas.read_csv("../book_code/data/titanic.csv")
print("---> Sample data - Titanic, #{} entries".format(len(sample_data_titanic)))
print("... Sample ...\n"
      "{}\n"
      "... END of Sample ...".format(sample_data_titanic[:5]))
# Data fix (for numpy 'unique') - We know there is missing 'Embarked' data that, when run through 'unique', is
# interpreted as 'float', but 'Embark' is 'str', so we're gonna change that
sample_data_titanic['Embarked'].fillna("missing", inplace=True)
# We make a 80/20% train/test split of the data
data_train = sample_data_titanic[:int(0.8 * len(sample_data_titanic))]
data_test = sample_data_titanic[int(0.8 * len(sample_data_titanic)):]
# ---> Logistic Regression Model <---
# ML training model
ml_training_model = prepare_data(data_train)
print("---> ML Training model\n"
      "... SAMPLE ...\n"
      "{}\n"
      "... END of SAMPLE".format(ml_training_model[:5]))
model = LogisticRegression()
model.fit(ml_training_model, data_train['Survived'])
# Make predictions
model_predictions_on_test_data = model.predict(prepare_data(data_test))
print("[--- Logistic Regression ---]")
print("---> Model predictions on test data:\n{}".format(model_predictions_on_test_data))
# Compute the accuracy of the model on the test data
model_score_on_test_data = model.score(prepare_data(data_test), data_test['Survived'])
print("---> Model Accuracy on test data:\n{}".format(model_score_on_test_data))

# ---> Non-linear model with Support Vector Machines <---
print("[--- SVC ---]")
model_svc = SVC()
model_svc.fit(ml_training_model, data_train['Survived'])
model_svc_score_on_test_data = model_svc.score(prepare_data(data_test), data_test['Survived'])
print("---> Model Accuracy on test data:\n{}".format(model_svc_score_on_test_data))

# ---> Classification with multiple classes: hand-written digits <---
print("[--- MNIST Small Dataset (KNN Classifier) ---]")
mnist_dataset = pandas.read_csv("../book_code/data/mnist_small.csv")
mnist_train = mnist_dataset[:int(0.8 * len(mnist_dataset))]
mnist_test = mnist_dataset[int(0.8 * len(mnist_dataset)):]
# Instantiate the classifier
knn = KNeighborsClassifier(n_neighbors=10)
# Train the classifier by dropping the 'label' column (which is the classification target)
#knn.fit(mnist_train.drop('label', axis=1), mnist_train['label'])
# Predictions
#knn_mnist_predictions = knn.predict(mnist_test.drop('label', axis=1))
#print("KNN ---> {}".format(knn_mnist_predictions))
