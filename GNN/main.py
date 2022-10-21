import pickle
import numpy as np
from sklearn.cluster import DBSCAN

from GNN.loader_data_model import HumanPresenceDataLoader
from GNN.generate_input_matrices import getfiles, feature_matrix, adj_matrix
from GNN.get_traj import extract_trajectories
from GNN.reduce_graph import reduce_graph
from GNN.run_model import reshape_data, run_A3T_GNN
from GNN.traj_clustering import change_format, plot_all_traj, traj_segmentation, distance_matrix, plot_cluster



####################################################################
# extract data and F feature matrix
path = "/GNN/data/logs22-04-22"
files = getfiles(path)

F = feature_matrix(files)       # todo: extract all node features
print(F.shape)
# np.save("data/input_matrices/FeatureMatrix.npy", F)

####################################################################
# get trajectories
dataset = F[:, :, 0]
extract_trajectories(dataset)
# todo: return trajectory or filename+location and then import

trajectories = None

# cluster trajectories
traj_lst = change_format(trajectories)
plot_all_traj(traj_lst)
traj_lst = traj_segmentation(traj_lst)
D = distance_matrix(traj_lst)

# 120 and 4 worked decent, (120, 6) also
mdl = DBSCAN(eps=120, min_samples=4)
cluster_lst = mdl.fit_predict(D)

print(cluster_lst)
print(len(cluster_lst))
# np.save("cluster_lst.npy", cluster_lst)

# cluster_lst = cluster_lst[cluster_lst == -1]
plot_cluster(traj_lst, cluster_lst)

####################################################################
# get occupancy grid from trajectories
# todo: incorporate clustering results in new feature matrix/occuppancy grids
# name it F and same format

####################################################################
# loop over number of engineers
# and for each occupancy grid create A, reduce F and A, load and store dataset
# then also run model for each dataset but on colab not here

####################################################################
# create adjacency matrices
N_nodes = len(F[0])
A = adj_matrix(N_nodes)
print(A.shape)
# plot_adj_matrix(A)
# np.save("data/input_matrices/Adj_Matrix.npy", A)

####################################################################
# reduce A and F
F, A = reduce_graph(F, A)

print(F.shape)
print(A.shape)
# plot_adj_matrix(A)
# np.save("data/input_matrices/FeatureMatrix_Reduced.npy", F)
# np.save("data/input_matrices/Adj_Matrix_Reduced.npy", A)

####################################################################
# load dataset into correct format
# todo: F and A as inputs
loader = HumanPresenceDataLoader()
dataset = loader.get_dataset(num_t_in=12, num_t_out=12)
print(next(iter(dataset)))

# pickle.dump(dataset, open("data/input_matrices/test_data_for_size.p", "wb"))

####################################################################
# run model
# better to run it in colab using binary file of the dataset

# batch_size = 16
# test_loader, train_loader, static_edge_index = reshape_data(dataset, batch_size)
#
# run_A3T_GNN(test_loader, train_loader, static_edge_index)
