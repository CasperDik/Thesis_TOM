import numpy as np
import matplotlib.pyplot as plt
from CMap2D import gridshow, CMap2D

def inspect_zero_values(F):
    F_prime = F[:, :, 0]
    heatmap = np.sum(F_prime, axis=0)
    print("Maximum # humans in 2 hours: ", max(heatmap))
    print("Number of cells never visited: ", np.count_nonzero(heatmap == 0))
    print("Number of cells visited less than 5 times: ", len(np.where(heatmap < 5)[0]))

    heatmap_min5 = np.where(heatmap>5)[0]
    plt.hist(heatmap_min5, bins=50)
    plt.show()

    heatmap_occ = np.array(np.array_split(heatmap, 100))
    grid1 = CMap2D()
    grid1.from_array(heatmap_occ, origin=(0,0), resolution=1.)

    fig, ax1 = plt.subplots(1, 1)
    plt.sca(ax1)
    gridshow(grid1.occupancy())
    plt.show()


def reduce_graph(F, A):
    """drop nodes and adjacent edges if always 0"""
    F_prime = F[:, :, 0]
    heatmap = np.sum(F_prime, axis=0)

    # need list with all zeros idx
    idx = np.where(heatmap==0)
    # delete from A and F
    A = np.delete(A, idx, 0)
    A = np.delete(A, idx, 1)
    F = np.delete(F, idx, axis=1)

    return F, A

if __name__ == "__main__":
    F = np.load("data/input_matrices/FeatureMatrix.npy")
    A = np.load("data/input_matrices/Adj_Matrix.npy")

    F, A = reduce_graph(F, A)

    np.save("data/input_matrices/test_A.npy", A)
    np.save("data/input_matrices/test_F.npy", F)
    # A about 5 times smaller, F half the size, entire dataset about half the size
