import induceC45 as c45
import random
from Classifier import gen_confusion_matrix
from Classifier import bfs
from validation import sample_df
from pandas import concat

# handles to flow of our code


def gen_random_forest(df, attr, data_path, num_attributes, num_data_points, num_trees):
    forest = []
    for i in range(num_trees):
        tree_attr = random_attributes(num_attributes, attr.copy())
        tree_df = random_sample(df, num_data_points)
        tree = c45.c45(tree_df, tree_attr, 0.1, True, data_path)
        forest.append(tree)
    return forest

# Selects n random attributes from an attributes dictionary


def random_attributes(num_attributes, attributes):
    class_label = attributes.pop('class_label', None)
    keys = list(attributes.keys())
    random_attr = {}
    while True:
        random_key = random.choice(keys)
        if random_key not in random_attr:
            random_attr[random_key] = attributes[random_key]
        if len(random_attr) == num_attributes:
            break
    random_attr['class_label'] = class_label
    return random_attr

# returns a dataframe with n random samples


def random_sample(dataframe, num_samples):
    return dataframe.sample(n=num_samples)

# uses 10 fold cross validation to determine our forest accuracy


def validate_forest(dataframe, attributes, data_path, num_attributes, num_data_points, num_trees):
    num_folds = 10
    matrix, indices = gen_confusion_matrix(dataframe, attributes)
    # overall_matrix = [row[:] for row in matrix]
    overall_accuracy = overall_error = total = 0
    df_partitions = sample_df(num_folds, dataframe)
    # begin evaluation
    for i in range(len(df_partitions)):
        holdout = df_partitions[i]
        training_set = concat([df_partitions[index]
                               for index in range(len(df_partitions)) if index != i])
        forest = gen_random_forest(
            training_set, attributes, data_path, num_attributes, num_data_points, num_trees)
        # classifying data in the holdout set for this fold
        for i in range(len(holdout)):
            pred = forest_prediction(holdout.iloc[i], forest)
            if pred != holdout.iloc[i][attributes['class_label']]:
                overall_error += 1
            else:
                overall_accuracy += 1
            total += 1
            matrix[indices[holdout.iloc[i][attributes['class_label']]]
                   ][indices[pred]] += 1
    # matrix now contains the entire
    print("Generating confusion matrix for 10 folds")
    print(matrix)
    print(f"Accuracy: {overall_accuracy * 100 / total:.2f}%")
    print(f"Error: {overall_error * 100 / total:.2f}%")


def forest_prediction(data, forest):
    predictions = {}
    for tree in forest:
        pred = bfs(data, tree)
        if pred not in predictions:
            predictions[pred] = 0
        predictions[pred] += 1
    predictions = sorted(predictions.items(), key=lambda pred: pred[1])
    return predictions[0][0]


def wrapper(data_path, num_attributes, num_data_points, num_trees):
    df, attr = c45.csv_to_df(data_path)
    validate_forest(df, attr, data_path, num_attributes,
                    num_data_points, num_trees)


wrapper('./data/iris.data.csv', 3, 100, 10)
