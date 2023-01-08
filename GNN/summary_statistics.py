import numpy as np

##### summary statistics #####

### before omitting nodes###
F = np.load("data/input_matrices/FeatureMatrix.npy")   # feature matrix   [7200,10000, 4]
A = np.load("data/input_matrices/Adj_Matrix.npy")      # adjacency matrix [10000, 10000]

# number of cells
total_nr_cells = F.shape[1]
print("Total number of cells: ", total_nr_cells)

# cells with no occupancy during simulation
cells_no_occ = np.count_nonzero(np.sum(F[:, :, 0], axis=0) == 0)
print("Total cells without occupancy: ", cells_no_occ)

# cells with <5 occupancy during simulation
cells_less_5 = len(np.where(np.sum(F[:, :, 0], axis=0) < 5)[0])
print("Total cells with <5 occupancy: ", cells_less_5)

# average occupancy per cell
avg_occ = F[:, :, 0].mean()
print("Average occupancy per cell: ", avg_occ)

# average occupancy per cell, excluding zeros
avg_occ_ex0 = F[np.where(F[:, :, 0] > 0)][:,0].mean()
print("average occupancy per cell, excluding zeros: ", avg_occ_ex0)

# max humans in 1 cell
max_occ = F[:, :, 0].max()
print("Maximum number of humans in 1 cell: ", max_occ)

### after omitting nodes ###
F_reduced = np.load("data/input_matrices/FeatureMatrix_Reduced.npy")
A_reduced = np.load("data/input_matrices/Adj_Matrix_Reduced.npy")

# nodes omitted
print("Nodes omitted: ", 10000-F_reduced.shape[1])

# total nodes in graph
print("total nodes: ", F_reduced.shape[1])

# total number of edges
print("Total number of edges: ", A_reduced.sum())

# average edges per node
print("average edges per node: ", A_reduced.sum()/F_reduced.shape[1])

