import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal import StaticGraphTemporalSignal
import sys


""""
code is based upon the dataloaders of the torch geometric temporal library
"""

"""
to install necessary libraries and dependencies:

check/change python and cuda version:
python -c "import torch; print(torch.__version__)"

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cpu.html
pip install torch-geometric
pip install torch_geometric_temporal
"""

class HumanPresenceDataLoader():
    def __init__(self):
        self.load_dataset()

    def load_dataset(self):
        try:
            # todo: filename and location as input
            self.A = np.load("Adj_Matrix.npy")
            self.X = np.load("FeatureMatrix.npy")

        except FileNotFoundError:
            sys.exit("File not found")

    def data_transformations(self, normalize: bool = False):
        self.X = self.X.transpose((1, 2, 0))    # transpose not needed if in correct format imported
        self.X = self.X.astype(np.float32)

        if normalize == True:
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
        self.data_transformations(normalize=True)
        self.get_edges_and_weights()
        self.get_features_and_target(num_t_in, num_t_out)
        dataset = StaticGraphTemporalSignal(self.edges, self.edge_weights, self.features, self.targets)

        return dataset

if __name__ == "__main__":
    loader = HumanPresenceDataLoader()
    dataset = loader.get_dataset(num_t_in=60, num_t_out=10)
    print(next(iter(dataset)))
