import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import os

def getfiles(path):
    """based upon code originally designed by Laurens Kuiper"""
    folder_names = []
    for entry_name in os.listdir(path):
        entry_path = os.path.join(path, entry_name)
        if os.path.isdir(entry_path):
            folder_names.append(entry_name)

    filepaths = []
    for folder in folder_names:
        # todo: different files?
        if folder[-5:] == "100cm" and folder[:11] == "simulation2":
        #if folder == "simulation2-25p-100cm":      # 1 specific simulation
            step = 0
            for file in os.scandir(path + "/" + folder):
                if file.is_file() and file.path[-4:] == ".txt":
                    step += 1
                    #if int(file.path[-7:-5]) % 10 == 0: #control time interval
                    if step > 60*60 and step < (60*60 + 30*60):    # only logs after hour in simulation
                        filepaths.append(file.path)

    return filepaths


def load_features(f):
    # delete hashtag from file, otherwise will stop reading after #
    delete_hashtag(f)

    # load data
    d = np.genfromtxt(f, delimiter=[1, 20], dtype=[("f0", np.uint8), ("f1", object)])
    x = d["f1"].astype("U")

    w = np.where(np.char.find(x, "Wall") > 0, 1, 0)
    c = np.where(np.char.find(x, "coffee") > 0, 1, 0)
    ws = np.where(np.char.find(x, "WS") > 0, 1, 0)

    # first column is human presence, second wall, third coffee, fourth workstation
    d = np.stack((d["f0"], w), axis=1)
    d = np.concatenate((d, c[:, None]), axis=1)
    d = np.concatenate((d, ws[:, None]), axis=1)

    return d

def delete_hashtag(f):
    with open(f, "rb") as input_file:
        s = input_file.read()
        input_file.close()
        s = s.replace(b"#", b"")

    with open(f, "wb") as output_file:
        output_file.write(s)

def feature_matrix(files: list):
    f0 = load_features(files[0]).astype('uint8')
    f1 = load_features(files[1]).astype('uint8')
    features = np.stack((f0, f1))
    for f in files[2:]:
        print(f)
        d = load_features(f).astype('uint8')
        features = np.vstack((features, d[None, :, :]))
    print(features.shape)
    return features

def adj_matrix(N_Nodes: int):
    # get the adjacency matrix as defined in the methodology
    # in GENERAL connect a node n to itself, next node below, node to the right, node to the right above and below diagonally
    # i.e. first degree neighbours
    # but some exception! --> indicated in the comments below (when those do not connections do not exist)

    # only works if grid is a square matrix of NxN
    # only correct if adjacency matrix is symmetric --> undirected graph

    dim = int(np.sqrt(N_Nodes))
    adj_mat = np.zeros((N_Nodes, N_Nodes), dtype="uint8")

    for n in range(N_Nodes):
        adj_mat[n, n] = 1   # connect each node to itself

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
    matplotlib.use('TkAgg')

    # confirms that adjacency matrix is correctly made --> check with 3x3 grid otherwise hard to determine
    Graph = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    nx.draw_networkx(Graph)
    plt.show()


if __name__ == "__main__":
    path = r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis TOM\GNN\data\logs22-04-22"
    files = getfiles(path)

    # files = ["Data/logs22-04-22/simulation1-50p-50cm/heatmap_08H00m01s.txt", "Data/logs22-04-22/simulation1-50p-50cm/heatmap_08H00m02s.txt", "Data/logs22-04-22/simulation1-50p-50cm/heatmap_08H00m03s.txt", "Data/logs22-04-22/simulation1-50p-50cm/heatmap_08H00m04s.txt"]

    F = feature_matrix(files)
    print(F.shape)
    N_nodes = len(F[0])
    A = adj_matrix(N_nodes)
    # plot_adj_matrix(A)

    print(F.shape)
    print(A.shape)
    #
    # np.save("data/input_matrices/FeatureMatrix.npy", F)
    # np.save("data/input_matrices/Adj_Matrix.npy", A)

