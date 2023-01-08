import pickle
import numpy as np

from GNN.data_loader import HumanPresenceDataLoader
from GNN.generate_input_matrices import getfiles, feature_matrix, adj_matrix

path_logs1 = r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis TOM\GNN\data\logs_22-04-22"
path_logs2 = r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis TOM\GNN\data\logs_04-11-22"

# todo: change getfiles function --> multiple simulations, only fraction of each dataset(steps < (60*60 + 60*15) <--15 minutes after 1 hour)
files_log1 = getfiles(path_logs1)
files_log2 = getfiles(path_logs2)
files = files_log1 + files_log2
print(len(files))

F = feature_matrix(files)
print(F.shape)

N_nodes = len(F[0])
A = adj_matrix(N_nodes)
print(A.shape)

np.save("data/input_matrices/FeatureMatrix_allsim.npy", F)
np.save("data/input_matrices/Adj_Matrix_allsim.npy", A)

# todo: not reduce grid --> different settings makes no sense
# todo: might lead to problems with plotting

loader = HumanPresenceDataLoader(A, F, normalize=True)
dataset = loader.get_dataset(num_t_in=2, num_t_out=5)
print(next(iter(dataset[0])))

pickle.dump(dataset, open("data/datasets/dataset_norm_allsim.p", "wb"))