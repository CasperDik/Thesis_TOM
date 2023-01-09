import pickle
import numpy as np

from GNN.data_loader import HumanPresenceDataLoader
from GNN.generate_input_matrices import getfiles, feature_matrix, adj_matrix, plot_adj_matrix
from GNN.reduce_graph import reduce_graph, back_to_occupancy_grid

name = "sim2_all_p_30min_5_30"

################### Feature Matrix ###################
# extract data and F feature matrix
path_logs1 = r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis TOM\GNN\data\logs_22-04-22"
path_logs2 = r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis TOM\GNN\data\logs_04-11-22"

files_log1 = getfiles(path_logs1)
files_log2 = getfiles(path_logs2)
files = files_log1 + files_log2

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
# np.save("data/input_matrices/FeatureMatrix_Reduced_sim7_25p_50cm.npy", F)
# np.save("data/input_matrices/Adj_Matrix_Reduced_sim7_25p_50cm.npy", A)
np.save("data/test_input_settings_datasets/idx_" + name + ".npy", idx)


################### Load Data for Model ###################
# load dataset into correct format
print(F.shape)

loader = HumanPresenceDataLoader(A, F, normalize=True)
dataset = loader.get_dataset(num_t_in=5, num_t_out=30)
print(next(iter(dataset[0])))

pickle.dump(dataset, open("data/test_input_settings_datasets/dataset_norm_" + name + ".p", "wb"))

# run model using GPU in google colab
# see A3T-GCN.ipynb
