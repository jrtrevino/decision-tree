from typing import overload
import pandas as pd
import math
from Classifier import gen_confusion_matrix
from validation import sample_df

# normalizes data in numerical columns


def csv_to_df(path):
    df = pd.read_csv(path, header=0)  # , skiprows=[2])
    attributes = df.iloc[0].to_dict()
    attributes['class_label'] = df.iloc[1][0]
    df = df.drop([0, 1])
    # replace values of attributes dict with possible values
    for key in attributes.copy():
        if key == 'class_label':
            continue
        # check if value isn't numerical
        if int(attributes[key]) != 0:
            val = df[key].unique()
        else:
            val = "num"
            df[key] = pd.to_numeric(df[key], errors='coerce')
            max_value = df[key].max()
            min_value = df[key].min()
            df[key] = (df[key] - min_value) / (max_value - min_value)
        attributes[key] = val
    return df, attributes

# calculates distance for numerical attributes between row_x and row_y


def euclid_distance(row_x, row_y, attributes):
    distance_num = distance_cat = 0
    # for storing the number of cat attributes and matching values
    number_cat = number_cat_match = 0
    for attribute in attributes:
        if attribute == "class_label" or attribute == attributes['class_label']:
            continue
        if type(attributes[attribute]) == str and attributes[attribute] == "num":
            distance_num += math.sqrt((row_x[attribute] - row_y[attribute])**2)
        else:
            number_cat += 1
            if row_x[attribute] == row_y[attribute]:
                number_cat_match += 1
    distance_cat = ((number_cat - number_cat_match) / number_cat)
    return distance_num + distance_cat

# determines the classification of df_row using dataset df with KNN.
# Data should be a row from a holdout set using the .iloc feature of pandas.
# Df should be our data set, and k should be the number of points we want
# to check distance to.


def knn(data, df, attributes, k):
    distances = []
    for index, row in df.iterrows():
        if index == data.name:
            continue
        distances.append(
            {'row-index': index, 'distance': euclid_distance(data, row, attributes)})
    distances = sorted(distances, key=lambda x: x['distance'])
    closest_df = df.loc[[row_index['row-index']
                         for row_index in distances[:k]]]
    predictions = closest_df[attributes['class_label']
                             ].value_counts().keys().tolist()
    return predictions[0]


def cross_validation(df, attributes, k):
    matrix, indices = gen_confusion_matrix(df, attributes)
    num_folds = 10
    overall_error = overall_accuracy = total = 0
    folds = sample_df(num_folds, df)
    for i in range(num_folds):
        print("Fold number {}".format(i))
        holdout = folds[i]
        training_set = pd.concat([folds[index]
                                  for index in range(num_folds) if index != i])
        for y in range(len(holdout)):
            pred = knn(holdout.iloc[y], training_set, attributes, k)
            if pred != holdout.iloc[i][attributes['class_label']]:
                overall_error += 1
            else:
                overall_accuracy += 1
            total += 1
            matrix[indices[holdout.iloc[i][attributes['class_label']]]
                   ][indices[pred]] += 1
    print("Generating confusion matrix for 10 folds")
    print(matrix)
    print(f"Accuracy: {overall_accuracy * 100 / total:.2f}%")
    print(f"Error: {overall_error * 100 / total:.2f}%")
    # for i in range(len(df)):
    #pred = knn(df.iloc[i], df, attributes, k)
    # print("Prediction: {} Actual: {}".format(
    # pred, row[attributes['class_label']]))


def wrapper(k):
    df, attributes = csv_to_df('./data/heart.csv')
    cross_validation(df, attributes, k)


wrapper(10)
