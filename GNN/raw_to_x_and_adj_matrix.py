"""
transform to .txt files to .npy in the format that can be used in the dataloader

desired format:
A, adj matrix --> NxN matrix with each entry the edge value, N=number of Nodes

X, feature matrix --> TxNxF, For all T times, for all N nodes the array F with features

"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os


def getfiles(path):
    folder_names = []
    for entry_name in os.listdir(path):
        entry_path = os.path.join(path, entry_name)
        if os.path.isdir(entry_path):
            folder_names.append(entry_name)

    filepaths = []
    for folder in folder_names:
        if folder[-5:] == "100cm" and folder[:11] == "simulation1":
            for file in os.scandir(path+ "/" + folder):
                if file.is_file() and file.path[-4:] == ".txt":
                    filepaths.append(file.path)

    return filepaths


def load_features(f):
    # delete hashtag from file, otherwise will stop reading after #
    delete_hashtag(f)

    # load data
    d = np.genfromtxt(f, delimiter=[1, 20], dtype=[("f0", np.uint8), ("f1", object)])

    # =0 if no obstacle (thus no wall, coffee, ws etc), otherwise 1
    obstacle = np.where(d["f1"] == b"\n", 0, 1)

    # stack human presence and obstacle
    d = np.stack((d["f0"], obstacle), axis=1)

    return d


def delete_hashtag(f):
    with open(f, "rb") as input_file:
        s = input_file.read()
        input_file.close()
        s = s.replace(b"#", b"")

    with open(f, "wb") as output_file:
        output_file.write(s)


def feature_matrix(files: list):
    f0 = load_features(files[0])
    f1 = load_features(files[1])
    features = np.stack((f0, f1))

    for f in files[2:]:
        print(f)
        d = load_features(f)
        features = np.vstack((features, d[None, :, :]))
    print(features.shape)

    return features


def adj_matrix(N_Nodes: int):
    # get the adjacency matrix as defined in the thesis
    # in GENERAL connect a node n to itself, next node below, node to the right, node to the right above and below diagonally
    # but some exception! --> indicated in the comments below

    # only works if adjacency matrix is a square matrix of NxN

    dim = int(np.sqrt(N_Nodes))
    adj_mat = np.zeros((N_Nodes, N_Nodes))
    for n in range(N_Nodes):
        adj_mat[n, n] = 1

        # right
        if not (n > (N_Nodes-dim-1)):     # if at last column, no connection to the right
            adj_mat[n, n+dim] = 1

        # below
        if not (n % dim == (dim-1)):     # if at bottom row, no connection to next node below
            adj_mat[n, n + 1] = 1

        # diagonal right below
        if not (n > (N_Nodes-dim-1)) | (n % dim == (dim-1)):      # if at the last column or at the bottom row, no connection to the diagonal right below
            adj_mat[n, n+dim+1] = 1

        # diagonal right up
        if not (n == 0) | (n % dim == 0) | (n > (N_Nodes-dim-1)):       # if at top row or at the last column, no diagonal connection upper right
            adj_mat[n, n+dim-1] = 1

    # as the edges are undirected mirror the upper triangle to the lower triangle
    adj_mat = np.triu(adj_mat) + np.tril(adj_mat.T, -1)

    return adj_mat


def plot_adj_matrix(A):
    # confirms that adjacency matrix is correctly made

    Graph = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    nx.draw_networkx(Graph)
    plt.show()


if __name__ == "__main__":
    path = "C:/Users/caspe/OneDrive/Documenten/MSc TOM/Thesis TOM/GNN/Data/logs22-04-22"
    files = getfiles(path)

    # files = ["Data/logs22-04-22/simulation1-50p-50cm/heatmap_08H00m01s.txt", "Data/logs22-04-22/simulation1-50p-50cm/heatmap_08H00m02s.txt", "Data/logs22-04-22/simulation1-50p-50cm/heatmap_08H00m03s.txt", "Data/logs22-04-22/simulation1-50p-50cm/heatmap_08H00m04s.txt"]

    F = feature_matrix(files)
    N_nodes = len(F[0])
    A = adj_matrix(N_nodes)
    # plot_adj_matrix(A)

    print(F.shape)
    print(A.shape)

    np.save("FeatureMatrix.npy", F)
    np.save("Adj_Matrix.npy", A)

