"""
    datasets.py

# Description
This file implements the dataset loading for the Meta-iCVI demonstration file.

This includes the following UCI datasets:
1. Wine - https://archive.ics.uci.edu/dataset/109/wine
2. Leaf - https://archive.ics.uci.edu/dataset/288/leaf
3. Cervical Cancer Behavior Risk - https://archive.ics.uci.edu/dataset/537/cervical+cancer+behavior+risk
4. Iris - https://archive.ics.uci.edu/dataset/53/iris
"""

# -----------------------------------------------------------------------------
# DEPENDENCIES
# -----------------------------------------------------------------------------

import numpy as np
from sklearn.datasets import make_blobs


# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

DATASETS = [
    'wine',
    'leaf',
    'cervical',
    'iris'
]

# -----------------------------------------------------------------------------
# CLASSES
# -----------------------------------------------------------------------------


class Dataset:

    def __init__(self, name):
        if name == 'wine':
            data = np.vstack([np.array(line.split(',')) for line in open('data/wine.data').readlines() if line])
            self.data = data[:, 1:].astype(float)
            self.labels = data[:, 0].astype(int)
            max_values = np.max(self.data, axis=0)
            min_values = np.min(self.data, axis=0)
            self.data = (self.data - min_values) / (max_values - min_values)
        if name == 'leaf':
            data = np.vstack([np.array(line.split(',')) for line in open('data/leaf/leaf.csv').readlines() if line])
            self.data = data[:, 1:].astype(float)
            self.labels = data[:, 0].astype(int)
            max_values = np.max(self.data, axis=0)
            min_values = np.min(self.data, axis=0)
            self.data = (self.data - min_values) / (max_values - min_values)
        if name == 'cervical':
            data = np.vstack([np.array(line.split(',')) for line in open('data/sobar-72.csv').readlines() if line])
            self.data = data[:, :-1].astype(float)
            self.labels = data[:, -1].astype(int)
            max_values = np.max(self.data, axis=0)
            min_values = np.min(self.data, axis=0)
            self.data = (self.data - min_values) / (max_values - min_values)
        if name == 'mushroom':
            data = np.vstack([np.array(line.split(',')) for line in open('data/agaricus-lepiota.data').readlines() if line])
            for ci in range(data.shape[1]):
                categories = np.unique(data[:, ci])
                for ri in range(data.shape[0]):
                    data[ri, ci] = np.where(categories == data[ri, ci])[0][0]
            print(data[0, :])
            self.data = data[:, 1:].astype(float)
            self.labels = data[:, 0].astype(int)
            max_values = np.max(self.data, axis=0)
            min_values = np.min(self.data, axis=0)
            self.data = (self.data - min_values) / (max_values - min_values)
            rem = max_values == min_values
            self.data = self.data[:, np.logical_not(rem)]
        if name == 'iris':
            data = np.vstack([np.array(line.split(',')) for line in open('data/iris.data').readlines() if line])
            print(data.shape)
            print(data[0])
            self.data = data[:, :-1].astype(float)
            self.labels = data[:, -1]
            labels = np.unique(self.labels)
            print(labels)
            self.labels[self.labels == labels[0]] = 1
            self.labels[self.labels == labels[1]] = 2
            self.labels[self.labels == labels[2]] = 3
            self.labels[self.labels == labels[3]] = 3
            self.labels = self.labels.astype(int)

            max_values = np.max(self.data, axis=0)
            min_values = np.min(self.data, axis=0)
            self.data = (self.data - min_values) / (max_values - min_values)
        if name == 'blobs':
            self.data, self.labels = make_blobs(n_samples=400, centers=4, n_features=10, random_state=0)
            max_values = np.max(self.data, axis=0)
            min_values = np.min(self.data, axis=0)
            self.data = (self.data - min_values) / (max_values - min_values)
