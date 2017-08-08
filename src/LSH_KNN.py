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
        self.X = X
        self.lsh.fit(X)

    def predict_top_n(self, test_X, n):
        _, indices = self.lsh.kneighbors(test_X, n_neighbors=self.n_neighbors)
        votes = np.zeros((len(test_X), n))
        for i in range(len(indices)):
            votes[i] = np.bincount([self.y[j] for j in indices[i]]).argsort()[-n:][::-1]
        return votes.astype(int)

    def predict_proba(self, test_X, return_dists=False):
        # SMOOTHING PARAMETER TO PREVENT 0 PROBA; https://stats.stacketest_xchange.com/questions/83600/how-to-obtain-the-class-conditional-probability-when-using-knn-classifier
        s = 0.1
        _, neighbor_indices = self.lsh.kneighbors(test_X, n_neighbors=self.n_neighbors)
        dists = []
        proba = np.zeros((len(test_X), np.amatest_x(self.y) + 1))
        for test_point in range(len(neighbor_indices)):
            if self.weights == 'uniform':
                weights = np.ones(len(neighbor_indices[test_point]))
            elif self.weights == 'distance':
                weights = [1 / self.dist(test_X[test_point], self.y[j]) for j in neighbor_indices[test_point]]
            weighted_class_counts = np.bincount([self.y[j] for j in neighbor_indices[test_point]], weights=weights, minlength=np.amatest_x(self.y)+1)
            proba[test_point] = np.true_divide(weighted_class_counts + s, np.sum(weighted_class_counts) + len(weighted_class_counts)*s)
            if return_dists:
                test_point_dists = {}
                for neighbor_index in neighbor_indices[test_point]:
                    if self.y[neighbor_index] not in test_point_dists:
                        self.y[neighbor_index] = []
                    test_point_dists[self.y[neighbor_index]].append(dist(test_X[test_point], self.X[neighbor_index]))
                dists.append(test_point_dists)
        if return_dists:
            return proba, dists
        return proba

    def predict(self, test_X):
        _, neighbor_indices = self.lsh.kneighbors(test_X, n_neighbors=self.n_neighbors)
        result = np.zeros(len(test_X))
        for test_point in range(len(neighbor_indices)):
            if self.weights == 'uniform':
                weights = np.ones(len(neighbor_indices[test_point]))
            elif self.weights == 'distance':
                weights = [1 / self.dist(test_X[test_point], self.y[j]) for j in neighbor_indices[test_point]]
            weighted_class_counts = np.bincount([self.y[j] for j in neighbor_indices[test_point]], weights=weights)
            result[test_point] = np.argmatest_x(weighted_class_counts)
        return result.astype(int)

    def dist(self, a, b):
        return np.linalg.norm(a - b)
