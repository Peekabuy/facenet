from sklearn.neighbors import LSHForest
import numpy as np
from scipy.stats import mode


class LSH_KNN:
    def __init__(self, weights='uniform', **kwargs):
        self.n_neighbors = kwargs['n_neighbors']
        self.lsh = LSHForest(**kwargs)
        self.weights = weights

    def fit(self, X, y):
        self.y = y
        self.lsh.fit(X)

    def predict_top_n(self, X, n):
        _, indices = self.lsh.kneighbors(X, n_neighbors=self.n_neighbors)
        votes = np.zeros((len(X), n))
        for i in range(len(indices)):
            votes[i] = np.bincount([self.y[j] for j in indices[i]]).argsort()[-n:][::-1]
        return votes.astype(int)

    def predict_proba(self, X):
        _, neighbor_indices = self.lsh.kneighbors(X, n_neighbors=self.n_neighbors)
        proba = np.zeros((len(X), np.amax(self.y) + 1))
        for test_point in range(len(neighbor_indices)):
            if self.weights == 'uniform':
                weights = np.ones(len(neighbor_indices[test_point]))
            elif self.weights == 'distance':
                weights = [1 / self.dist(X[test_point], self.y[j]) for j in neighbor_indices[test_point]]
            weighted_class_counts = np.bincount([self.y[j] for j in neighbor_indices[test_point]], weights=weights)
            proba[test_point] = np.true_divide(weighted_class_counts, np.sum(weighted_class_counts))
        return proba

    def predict(self, X):
        _, neighbor_indices = self.lsh.kneighbors(X, n_neighbors=self.n_neighbors)
        result = np.zeros(len(X))
        for test_point in range(len(neighbor_indices)):
            if self.weights == 'uniform':
                weights = np.ones(len(neighbor_indices[test_point]))
            elif self.weights == 'distance':
                weights = [1 / self.dist(X[test_point], self.y[j]) for j in neighbor_indices[test_point]]
            weighted_class_counts = np.bincount([self.y[j] for j in neighbor_indices[test_point]], weights=weights)
            result[test_point] = np.argmax(weighted_class_counts)
        return result.astype(int)

    def dist(self, a, b):
        return np.linalg.norm(a - b)
