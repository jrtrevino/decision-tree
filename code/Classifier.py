import pandas as pd
import json
import sys

# same concept as classify_wrapper, but takes in the df, attributes, and tree directly


def classify_df(df, attr, tree, matrix, indices):
    num_correct = num_incorrect = accuracy = 0
    # print("Training set detected. Class label: {}".format(
    # attr['class_label']))
    #print("Creating confusion matrix for this dataset\n")
    #matrix, indices = gen_confusion_matrix(df, attr)
    for index, row in df.iterrows():
        pred = bfs(row, tree)
        if pred == row[attr['class_label']]:
            num_correct += 1
        else:
            num_incorrect += 1
        # print(pred, pred in indices)
        #print(indices[row[attr['class_label']]], indices[pred])
        matrix[indices[row[attr['class_label']]]][indices[pred]] += 1
        # matrix[row][col] implies matrix[actual][pred]
        # print("Row index: {}, predicted label: {}, actual: {}".format(
        # index, pred, row[attr['class_label']]))
    metrics = {
        'Classified': num_correct + num_incorrect,
        'Correct': num_correct,
        'Incorrect': num_incorrect,
        'Accuracy': num_correct / (num_correct + num_incorrect) * 100,
        'Error': num_incorrect / (num_correct + num_incorrect) * 100,
        'Indices': indices,
        'Matrix': matrix}
    display_metrics(metrics)
    return metrics

    # classifies a dataset using a c45 induced classification tree in a JSON file.


def classify_wrapper(data_path, tree_path):
    try:
        f = open(tree_path, "r")
        tree = json.load(f)
    except json.JSONDecodeError as e:
        print("json.loads failed for {}".format(tree_path))
        print("Reason: {}".format(e))
        sys.exit(-1)

    attr, df, training = csv_to_df(data_path)
    if not training:
        for index, row in df.iterrows():
            print("Row index: {}, predicted label: {}".format(
                index, bfs(row, tree)))
    else:
        # training set detected
        num_correct = num_incorrect = accuracy = 0
        print("Training set detected. Class label: {}".format(
            attr['class_label']))
        print("Creating confusion matrix for this dataset\n")
        matrix, indices = gen_confusion_matrix(df, attr)
        for index, row in df.iterrows():
            pred = bfs(row, tree)
            if pred == row[attr['class_label']]:
                num_correct += 1
            else:
                num_incorrect += 1
            matrix[indices[row[attr['class_label']]]][indices[pred]] += 1
            # matrix[row][col] implies matrix[actual][pred]
            # print("Row index: {}, predicted label: {}, actual: {}".format(
            # index, pred, row[attr['class_label']]))
        metrics = {
            'Classified': num_correct + num_incorrect,
            'Correct': num_correct,
            'Incorrect': num_incorrect,
            'Accuracy': num_correct / (num_correct + num_incorrect) * 100,
            'Error': num_incorrect / (num_correct + num_incorrect) * 100,
            'Indices': indices,
            'Matrix': matrix}
        display_metrics(metrics)
        return metrics


def display_metrics(metrics):
    print("\nTotal classified: {}".format(metrics['Classified']))
    print("Total correctly classified: {}".format(metrics['Correct']))
    print("Total incorrectly classified: {}".format(metrics['Incorrect']))
    print("Accuracy: {:.2f}%".format(metrics['Accuracy']))
    print("Error rate: {:.2f}%".format(metrics['Error']))
    print("\nPrinting Confusion Matrix")
    print("Matrix headers: Pred(column) vs Actual(row)")
    print(
        "Header values/indices for rows & columns: {}\n".format(metrics['Indices']))
    print(metrics['Matrix'], "\n")


def csv_to_df(path):
    attributes = {'class_label': None}
    drop_indices = [0]
    index_shift = 3
    training_set = False
    # check if training set
    with open(path, "r") as file:
        data = file.readlines()
    if len(data[2].strip()) > 0:
        attributes['class_label'] = data[2].strip()
        drop_indices = [0, 1]
        index_shift -= 1
        training_set = True
    # convert to pandas df
    df = pd.read_csv(path, header=0)
    attributes.update(df.iloc[0].to_dict())
    # drop unnecessary rows
    df = df.drop(drop_indices)
    df.index += index_shift
    return attributes, df, training_set


def bfs(row, tree):
    if 'leaf' in tree:
        # we have reached our decision
        return tree['leaf']['decision']
    else:
        # grab the next attribute in the tree
        attr = tree['node']['var']
        # grab the row's value for that attribute
        attr_val = row[attr]
        # print(tree['node']['edges'])
        for obj in tree['node']['edges']:
            # search through edges and match the attribute
            if attr_val == obj['edge']['value']:
                next_tree = obj['edge']
                return bfs(row, next_tree)


def gen_confusion_matrix(dataframe, attr):
    matrix = []
    indices = {}
    index = 0
    dom_class = dataframe.groupby(attr['class_label']).groups
    for val in dom_class:
        indices[val] = index
        index += 1
    for dom_val in indices:
        matrix.append([0] * len(indices))
    return matrix, indices


def main_wrapper():
    args = sys.argv[1:]
    if len(args) != 2:
        print("Usage: python3 Classifier <CSVFile> <JSONFile>")
    elif ".csv" not in args[0] or ".json" not in args[1]:
        print("Please ensure you provide both a CSV and JSON file.")
        print("Usage: python3 Classifier <CSVFile> <JSONFile>")
    else:
        classification_metadata = classify_wrapper(args[0], args[1])
        return classification_metadata


if __name__ == "__main__":
    main_wrapper()
