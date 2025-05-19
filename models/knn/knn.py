import pandas as pd
import numpy as np

# This code implements an the K-Nearest-Neighbor algorithm
class KNearestNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.data = None
        self.labels = None

    def fit(self, data, labels):
        self.data = data
        self.labels = labels

    def predict(self, new_data):
        distances = np.linalg.norm(self.data - new_data, axis=1)
        nearest_indices = np.argsort(distances)[:self.n_neighbors]
        nearest_labels = self.labels[nearest_indices]
        return np.bincount(nearest_labels).argmax()