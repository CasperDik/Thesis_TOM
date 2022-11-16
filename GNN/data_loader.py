import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal import StaticGraphTemporalSignal
import pickle

""""
code is based upon the dataloaders of the torch geometric temporal library
"""

class HumanPresenceDataLoader():
    def __init__(self, A, F, normalize: bool = False):
        self.A = A
        self.X = F
        self.norm = normalize

    def data_transformations(self):
        self.X = self.X.transpose((1, 2, 0))    # transpose not needed if in correct format imported
        self.X = self.X.astype(np.float32)

        if self.norm == True:
            self.X = self.normalize_zscore(self.X)

        self.A = torch.from_numpy(self.A)
        self.X = torch.from_numpy(self.X)

    def normalize_zscore(self, X):
        means = np.mean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)
        stds = np.std(X, axis=(0, 2))
        X = X / stds.reshape(1, -1, 1)
        return X

    def get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def get_features_and_target(self, num_t_in:int, num_t_out:int):
        indices = [(i, i + (num_t_in + num_t_out)) for i in range(self.X.shape[2] - (num_t_in + num_t_out) + 1)]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i: i + num_t_in]).numpy())
            target.append((self.X[:, 0, i + num_t_in: j]).numpy())

        self.features = features
        self.targets = target


    def get_dataset(self, num_t_in, num_t_out):
        self.data_transformations()
        self.get_edges_and_weights()
        self.get_features_and_target(num_t_in, num_t_out)
        dataset = StaticGraphTemporalSignal(self.edges, self.edge_weights, self.features, self.targets)

        return dataset

if __name__ == "__main__":
    F = np.load("data/input_matrices/FeatureMatrix.npy")
    A = np.load("data/input_matrices/Adj_Matrix.npy")

    loader = HumanPresenceDataLoader(A, F)
    dataset = loader.get_dataset(num_t_in=12, num_t_out=12)
    print(next(iter(dataset)))

    pickle.dump(dataset, open("data/input_matrices/test_data_for_size.p", "wb"))

    # data = pickle.load(open("dataset.p", "rb"))
    # print(type(data))
    # print(next(iter(data)))

