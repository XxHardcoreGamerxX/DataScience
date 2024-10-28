import sys
import numpy as np
import pandas as pd
"""
File reader

Parameters:
- file_path: Input file to be read

Returns:
The input file after having being read
"""
def file_reader(file_path):
    return pd.read_csv(file_path, delimiter="\t")
"""
Function that writes to file

Parameters:
- Data: the data to be written in the output file
- File_path: The file that will store the output
"""
def file_writer(data, file_path):
    data.to_csv(file_path, sep="\t", index=False)
"""
Node class to determine the characteristics of the nodes in our tree
"""
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value
"""
DecisionTreeClassifier class that will contain the main logic and helper functions for classifying our data
"""
class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=5):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
    """
    Function that constructs the tree based on the data

    Parameters:
    - data: Data from a given input file
    - current_depth: The depth level that the tree is currently constructing the tree on

    Returns:
    - The tree built using the C4.5 algorithm where we utilise gain ration for selecting attributes
    """
    def build_tree(self, data, current_depth=0):
        num_samples, num_features = data.shape

        if num_samples < self.min_samples_split or current_depth >= self.max_depth:
            leaf_value = data.iloc[:, -1].mode()[0]
            return Node(value=leaf_value)
        
        best_gain_ratio = 0
        best_feature = None
        best_threshold = None
        best_subsets = None

        for feature_index in range(num_features - 1):
            feature_values = data.iloc[:, feature_index]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                left_subset = data[data.iloc[:, feature_index] <= threshold]
                right_subset = data[data.iloc[:, feature_index] > threshold]
                if len(left_subset) < self.min_samples_split or len(right_subset) < self.min_samples_split:
                    continue

                current_gain_ratio = self.gain_ratio(data.iloc[:, -1], left_subset.iloc[:, -1], right_subset.iloc[:, -1])
                if current_gain_ratio > best_gain_ratio:
                    best_gain_ratio = current_gain_ratio
                    best_feature = feature_index
                    best_threshold = threshold
                    best_subsets = (left_subset, right_subset)

        if best_feature is None:
            leaf_value = data.iloc[:, -1].mode()[0]
            return Node(value=leaf_value)

        left_tree = self.build_tree(best_subsets[0], current_depth + 1)
        right_tree = self.build_tree(best_subsets[1], current_depth + 1)

        return Node(feature_index=best_feature, threshold=best_threshold, left=left_tree, right=right_tree, info_gain=best_gain_ratio)
    """
    Function that calculates entropy

    Parameters:
    - labels: the labels of the given data

    Returns:
    - The entropy that has been calculated

    """
    def entropy(self, labels):
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        num_labels = len(labels)
        entropy = 0

        for count in label_counts:
            probability = count / num_labels
            entropy -= probability * np.log2(probability)

        return entropy
    """
    Function that calculates the info gain

    Parameters:
    - Parent: parent node
    - left: left child of parent node
    - right: right child of parent node

    Returns:
    - calculated info gain
    """
    def info_gain(self, parent, left, right):
        info = self.entropy(parent)
        left_entropy = self.entropy(left)
        right_entropy = self.entropy(right)
        info_A = (len(left)/len(parent))*left_entropy + (len(right)/len(parent)) * right_entropy
        info_gain = info - info_A
        return info_gain
    """ 
    Function that calculates the split info

    Parameters:
    - parent: parent node
    - subsets: partition of the parent dataset

    Returns:
    - The calculated split info
    """
    def split_info(self, parent, subsets):
        num_total = len(parent)
        split_info = 0
        
        for subset in subsets:
            proportion = len(subset) / num_total
            split_info -= proportion * np.log2(proportion)
        
        return split_info
    
    """
    Function that calculates the gain ratio

    Parameters:
    - Parent: parent node
    - left: left child of parent node
    - right: right child of parent node

    Returns:
    The calculated gain ratio
    """
    def gain_ratio(self, parent, left, right):
        info_gain = self.info_gain(parent, left, right)
        info_A = self.split_info(parent, [left, right])
        gain_ratio = info_gain / info_A if info_A != 0 else 0
        return gain_ratio

    def predict(self, X):
        """
        Predicts the target values for the input data

        Parameters:
        - X: The input data containing features.

        Returns:
        - A list of predicted target values corresponding to each row in X
        """
        predictions = []
        for _, row in X.iterrows():
            node = self.root
            while node.value is None:
                feature_value = row.iloc[node.feature_index]
                if feature_value <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.value)
        return predictions
    """
    Function that is responsiple for fitting the classifier to  the data

    Parameters:
    - X: A feature matrix consisting of n_samples and n_features that represent training samples
    - Y: The target values of the shape n_samples that represnet the target labels of the training samples
    """
    def fit(self, X, Y):
        data = pd.concat([X, Y], axis=1)
        self.root = self.build_tree(data)


if __name__ == "__main__":
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    output_filename = sys.argv[3]

    train_data = file_reader(train_filename)
    test_data = file_reader(test_filename)

    tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
    tree.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1])
    predictions = tree.predict(test_data)

    result_data = test_data.copy()
    result_data['predicted'] = predictions
    file_writer(result_data, output_filename)