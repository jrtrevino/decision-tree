import sys
from pandas import concat
from Classifier import classify_df
from Classifier import gen_confusion_matrix
from induceC45 import c45
from induceC45 import csv_to_df as to_df


def wrapper(n_slices, data_path):
    # step 1: partition dataset
    df, attr = to_df(data_path)
    df_partitions = sample_df(
        n_slices, df) if n_slices >= 0 else None
    # gen confusion matrix for full dataset
    print("Generating Confusion Matrix")
    matrix, indices = gen_confusion_matrix(df, attr)
    overall_matrix = [row[:] for row in matrix]
    overall_accuracy = overall_error = 0
    # step 2: for each partition i, generate tree
    if df_partitions is not None:
        for i in range(len(df_partitions)):
            holdout = df_partitions[i]
            training_set = concat([df_partitions[index]
                                   for index in range(len(df_partitions)) if index != i])
            tree = c45(training_set, attr, 0.3, True, data_path)
            # step 2-i: calculate accuracy for each tree from partition i using holdout set
            results = classify_df(holdout, attr, tree, matrix, indices)
            overall_accuracy += results['Accuracy']
            overall_error += results['Error']
            for i in range(len(results['Matrix'])):
                overall_matrix[i] = [a+b for a,
                                     b in zip(results['Matrix'][i], overall_matrix[i])]
    else:
        for i in range(len(df)):
            training_df = df.copy()
            holdout = df.sample(n=1)
            training_df = training_df.drop(holdout.index[0])
            tree = c45(training_df, attr, 0.45, True, data_path)
            # step 2-i: calculate accuracy for each tree from partition i using holdout set
            results = classify_df(holdout, attr, tree, matrix, indices)
            overall_accuracy += results['Accuracy']
            overall_error += results['Error']
            for i in range(len(results['Matrix'])):
                overall_matrix[i] = [a+b for a,
                                     b in zip(results['Matrix'][i], overall_matrix[i])]
    # step 3: using accuracies generated from steps 2-i to 2-n_slices, calc overall acc
    print("Overall Accuracy: {:.2f}%".format(
        overall_accuracy / (n_slices if n_slices >= 0 else len(df) - 1)))
    print("Overall Error Rate: {:.2f}%".format(
        overall_error / (n_slices if n_slices >= 0 else len(df) - 1)))
    print("Overall confusion Matrix: {}".format(overall_matrix))


def all_but_one(df):
    print(None)

# This should be used if n_slices > 1.
# Use all_but_one for n_slices = -1.


def sample_df(n_slices, dataframe):
    df_array = []
    for i in range(n_slices):
        sample = dataframe.sample(frac=1/n_slices)
        df_array.append(sample)
    # print(len(df_array))
    return df_array


# wrapper(4, './agaricus-lepiota.csv')
if __name__ == "__main__":
    args = sys.argv[1:]
    if not len(args) >= 2:
        print("Usage: python3 Evaluation.py <dataset.csv> <n> [restrictions]")
        print("Where n is an int equalling -1 or greater than 1.")
        sys.exit(-1)
    elif ".csv" not in args[0]:
        print("Please provide a dataset as your first argument")
        sys.exit(-1)
    try:
        int(args[1])
    except:
        print("Argument 2 should be an integer")
        sys.exit(-1)
    print("Beginning Evaluation!")
    wrapper(int(args[1]), args[0])
