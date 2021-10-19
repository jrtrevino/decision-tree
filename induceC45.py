import pandas as pd
import math  # for log calculation
import json
import sys
# Requires a path to a TRAINING set, which includes the class label in the dataset.
# Structure of our dataset CSV file should be as follows:
#   Line 1 -> Attributes of our dataset, including out class category
#   Line 2 -> Domain of each attribute
#   Line 3 -> Our class attribute
#   Line 4+ -> Our data
# Returns a pandas dataframe of the dataset, and a dictonary containing our attributes.
# This dictionary also contains a 'class_label' key for easy access for later.


def csv_to_df(path):
    df = pd.read_csv(path, header=0)  # , skiprows=[2])
    attributes = df.iloc[0].to_dict()
    attributes['class_label'] = df.iloc[1][0]
    df = df.drop([0, 1])
    return df, attributes

# Requires a pandas dataframe and a dictionary object containing our
# attributes.
# Returns an array containing:
#   0 -> The most frequent class label (str)
#   1 -> The probability of choosing this label within the dataset.
# This is a helper function used by c45() when our information gain
# computations are not high enough for our threshold.


def find_most_frequent_label(dataframe, attributes):
    label = value = attribute = None
    values = dataframe[attributes['class_label']].value_counts()
    keys = values.index.tolist()
    for i in range(len(values)):
        if value is None or values[i] > value:
            label = keys[i]
            value = values[i]
    return [label, value/len(dataframe)]

# Determines which attribute to split on. Requires a pandas
# dataframe, a dictionary of attributes, and a float value containing
# our information gain threshold.
# Returns the name of the attribute to split on, or None if
# our information gain calculation was insufficient based on the
# threshold.


def select_splitting_attribute(dataframe, attributes, threshold):
    attributes_gain = []
    class_count = dataframe.groupby(attributes['class_label']).size()
    df_entropy = sum([-(x / len(dataframe)) * math.log2(x / len(dataframe))
                      for x in class_count])
    for attribute in dataframe.columns:
        # skip our class label -> it's not an attribute
        if attribute == attributes['class_label']:
            continue
        # for grabbing subset size for later
        attr_df = dataframe.groupby([attribute]).groups
        # group by labels in an attribute by the class label
        attr_class_df = dataframe.groupby(
            [attribute, attributes['class_label']]).groups
        attr_entropy = {}
        attr_subset_size = {}
        # will contain the gain for the attribute to append to attributes_gain array
        attribute_gain = {attribute: None}
        for key in attr_class_df:
            # key -> index for grouped attributes
            # print(attr_df[key[0]].tolist())#  <- dataset length
            if key[0] not in attr_subset_size:
                attr_subset_size[key[0]] = len(
                    attr_df[key[0]].tolist())
            if key[0] not in attr_entropy:
                attr_entropy[key[0]] = 0
            pr = len(attr_class_df[key]) / len(attr_df[key[0]])
            attr_entropy[key[0]] += (math.log2(pr) * (-pr))
        # print(attribute, attr_subset_size)
       #  print(attr_entropy, sum([size for size in attr_subset_size.values()]))
        # attribute gain calculation
        attribute_gain[attribute] = sum(
            [(attr_subset_size[key] / sum(attr_subset_size.values())) *
             attr_entropy[key] for key in attr_entropy])
        attributes_gain.append(
            {'attribute': attribute, 'gain': df_entropy - attribute_gain[attribute]})
    # first item will have highest gain
    attributes_gain = sorted(
        attributes_gain, key=lambda d: d['gain'], reverse=True)
    # print(attributes_gain)
    return attributes_gain[0]['attribute'] if attributes_gain[0]['gain'] > threshold else None

# The heart of our classification.
# Returns a classification tree as a dictionary.
# Requires a pandas dataframe, dictionary containing our attributes,
# and a threshold value.
# The parent and file path arguments are to place metadata in the
# tree when it is initialized.


def c45(dataframe, attributes, threshold, parent=False, file=None):
    node_label = None
    tree = {}
    if parent:
        # just to add our metadata for the tree
        tree['dataset'] = file
        # tree['node'] = {}
    # check termination conditions 1 (if) & 2 (elif)
    if len(dataframe.groupby(attributes['class_label'])) == 1:
        # checks if all of the class labels are the same
        node_label = list(dataframe.groupby(
            [attributes['class_label']]).groups)[0]
        # print('All the same class label: {} for class {}'.format(node_label, attributes['class_label']))
        tree['leaf'] = {'decision': node_label, 'p': 1.0}

    elif len(attributes) == 1:
        # we choose 1 instead of 0 in condition check because we have the class
        # label inside of our attributes dictionary, but this wont ever get removed.
        print('Find most frequent label')
        node_label = find_most_frequent_label(dataframe, attributes)
        tree['leaf'] = {'decision': node_label[0], 'p': node_label[1]}
    else:
        # determine splitting attribute
        splitting_attribute = select_splitting_attribute(
            dataframe, attributes, threshold)
        # print(splitting_attribute)
        if not splitting_attribute:
            node_label = find_most_frequent_label(dataframe, attributes)
            # print('No good splitting attribute, making leaf: {}'.format(node_label[0]))
            tree['leaf'] = {'decision': node_label[0], 'p': node_label[1]}
        else:
            node_label = splitting_attribute
            # filter attribute from attributes
            attributes.pop(node_label, None)
            # print(node_label)
            tree['node'] = {'var': node_label, 'edges': []}
            # grab all values for attribute in dataframe
            attr_df = dataframe.groupby([node_label]).groups
            for key in attr_df:
                filtered_df = dataframe.loc[dataframe[node_label] == key]
                # print(len(filtered_df))
                # print(filtered_df)
                # print("Creating Edge: {}, Node: {}".format(key, node_label))
                tree['node']['edges'].append(
                    {'edge': key,
                     'value':  (c45(filtered_df, attributes, threshold))})
    return tree

# Our wrapper to create a c45 decision tree. Requires a path to our dataset and
# an option restrictions file containing attributes to ignore during our classification.


def induce_c45(data_path, restrictions_path=None):
    df, attributes = csv_to_df(data_path)
    tree = c45(df, attributes, 0.10, True, data_path)
    print(json.dumps(tree, indent=2))
    return tree


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 1:
        print("Usage: python3 induceC45.py <{}> [<{}>]".format(
            "dataset", "restrictions"))
    elif ".csv" not in args[0]:
        print("Please provide a csv file as the first argument.")
    else:
        induce_c45(args[0], args[1] if len(args) > 1 else None)
