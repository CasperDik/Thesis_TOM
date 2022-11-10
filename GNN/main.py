import pickle
import numpy as np
from sklearn.cluster import DBSCAN

from GNN.data_loader import HumanPresenceDataLoader
from GNN.generate_input_matrices import getfiles, feature_matrix, adj_matrix, plot_adj_matrix
from GNN.get_traj import extract_trajectories
from GNN.reduce_graph import reduce_graph, back_to_occupancy_grid
from GNN.run_model import reshape_data, run_A3T_GNN
from GNN.traj_clustering import change_format, plot_all_traj, traj_segmentation, distance_matrix, plot_cluster



################### Feature Matrix ###################
# extract data and F feature matrix
path = r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis TOM\GNN\data\logs22-04-22"
files = getfiles(path)

F = feature_matrix(files)
print(F.shape)
np.save("data/input_matrices/FeatureMatrix.npy", F)

###################Engineer Types###################
# trajectories and clustering

# todo: trajactories and identifying engineer type/work sequence

# loop over number of engineers
# and for each occupancy grid create A, reduce F and A, load and store dataset
# then also run model for each dataset but on colab not here

################### Adjacency Matrix ###################
# create adjacency matrix
N_nodes = len(F[0])
A = adj_matrix(N_nodes)
print(A.shape)
plot_adj_matrix(A)
np.save("data/input_matrices/Adj_Matrix.npy", A)

################### Reduce Graph ###################
# reduce A and F
F = np.load("data/input_matrices/FeatureMatrix.npy")
A = np.load("data/input_matrices/Adj_Matrix.npy")

F, A, idx = reduce_graph(F, A)

print(F.shape)
print(A.shape)
# plot_adj_matrix(A)
np.save("data/input_matrices/FeatureMatrix_Reduced.npy", F)
np.save("data/input_matrices/Adj_Matrix_Reduced.npy", A)
np.save("data/input_matrices/idx.npy", idx)

################### Load Data for Model ###################
# load dataset into correct format
F = np.load("data/input_matrices/FeatureMatrix_Reduced.npy")[2000:5000,:,:]
A = np.load("data/input_matrices/Adj_Matrix_Reduced.npy")
print(F.shape)
loader = HumanPresenceDataLoader(A, F, normalize=False)

dataset = loader.get_dataset(num_t_in=12, num_t_out=12)
print(next(iter(dataset)))

pickle.dump(dataset, open("data/datasets/test_data_smaller.p", "wb"))

################### Run Model ###################
# better to run it in colab using binary file of the dataset

batch_size = 16
# test_loader, train_loader, static_edge_index = reshape_data(dataset, batch_size)

# run_A3T_GNN(test_loader, train_loader, static_edge_index, idx)

