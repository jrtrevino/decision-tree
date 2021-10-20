# C4.5 Decision Tree Classification
A set of files that generate a classification tree, evaluate datapoints, and validate our tree using 
cross-validation methods.

## Lab members
Joey Trevino, jrtrevin@calpoly.edu
Casey Koiwai, kkoiwai@calpoly.edu

### Notes
This program does not support the restrictions file implementation. All three programs have a 
'wrapper' function that gets activated when ran from the command line. See running instructions below.
Our validation program will print out the overall metrics for our classification tree.

## Running

### C45
To run induceC45, you must provide a dataset. We use a default threshold value of 0.30 to use when
calculating information gain.

```
python3 induceC45.py <dataset.csv>
```

This program produces a JSON formatted dictionary of the tree. You can redirect standard out into
a JSON file, which will be used in our next program, Classifer.py.

### Classifier
To run Classifier.py, you must provide a CSV file of data and a classification tree.
```
python3 Classifier.py <dataset.csv> <classifier.json>
```

If a training set is provided, the program auto detects this and will print the predicted and 
actual class label. If a test set is provided, only the predicted label is displayed.

### Validation
Lastly, we have the program validation.py. This will take a dataset, and run cross validation against
a generated classification tree. To run this program, run:

```
python3 validation.py <dataset.csv> <num_folds>
```
Num_folds is an integer such that -1 <= n. Please note that if the number of folds is greater than the 
size of the dataset, issues may occur.

If num_folds is -1, all-but-one cross validation occurs. Any other number generates num_fold partitions
and uses one of the folds as a holdout set.

#### Submission
We have posted the results from validation.py using 10 folds on all three datasets. The decision tree
uses an information gain of 0.30 as a threshold for classification. The output includes:
    1. Overall Accuracy (%)
    2. Overall Error Rate (%)
    3. Overall Confusion Matrix

