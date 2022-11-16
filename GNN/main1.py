import pickle
import numpy as np

from GNN.data_loader import HumanPresenceDataLoader
from GNN.generate_input_matrices import getfiles, feature_matrix, adj_matrix, plot_adj_matrix
from GNN.reduce_graph import reduce_graph, back_to_occupancy_grid



################### Feature Matrix ###################
# extract data and F feature matrix
path = r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis TOM\GNN\data\logs22-04-22"
files = getfiles(path)  # can take long --> pickle? or pickle at later stage

F = feature_matrix(files)
print(F.shape)

################### Adjacency Matrix ###################
# create adjacency matrix
N_nodes = len(F[0])
A = adj_matrix(N_nodes)
print(A.shape)

################### Reduce Graph ###################
# reduce A and F
F, A, idx = reduce_graph(F, A)

print(F.shape)
print(A.shape)
np.save("data/input_matrices/FeatureMatrix_Reduced.npy", F)
np.save("data/input_matrices/Adj_Matrix_Reduced.npy", A)
np.save("data/input_matrices/idx.npy", idx)

################### Load Data for Model ###################
# load dataset into correct format
print(F.shape)

loader = HumanPresenceDataLoader(A, F, normalize=False)
dataset = loader.get_dataset(num_t_in=12, num_t_out=12)
print(next(iter(dataset)))

pickle.dump(dataset, open("data/datasets/test.p", "wb"))

# run model using GPU in google colab
# see A3T-GCN.ipynb