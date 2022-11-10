import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import directed_hausdorff
from sklearn.cluster import DBSCAN
import math


def change_format(data):
    traj_lst = []
    for id in data["id"].unique():
        row = data.loc[data["id"] == id, " pos row"].to_numpy()
        col = data.loc[data["id"] == id, " pos col"].to_numpy()
        traj_lst.append(np.dstack((row, col))[0])
    return traj_lst


def plot_all_traj(traj_lst):
    for traj in traj_lst:
        plt.plot(traj[:, 0], traj[:, 1])
    plt.show()


def traj_segmentation(traj_lst):
    degree_threshold = 5
    for traj_index, traj in enumerate(traj_lst):

        hold_index_lst = []
        previous_azimuth = 1000

        for point_index, point in enumerate(traj[:-1]):
            next_point = traj[point_index + 1]
            diff_vector = next_point - point
            azimuth = (math.degrees(math.atan2(*diff_vector)) + 360) % 360

            if abs(azimuth - previous_azimuth) > degree_threshold:
                hold_index_lst.append(point_index)
                previous_azimuth = azimuth
        hold_index_lst.append(traj.shape[0] - 1)  # Last point of trajectory is always added

        traj_lst[traj_index] = traj[hold_index_lst, :]
    return traj_lst

def hausdorff(u, v):
    d = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    return d

def distance_matrix(traj_lst):
    traj_count = len(traj_lst)
    D = np.zeros((traj_count, traj_count))

    # This may take a while
    for i in range(traj_count):
        for j in range(i + 1, traj_count):
            distance = hausdorff(traj_lst[i], traj_lst[j])
            D[i, j] = distance
            D[j, i] = distance
    return D


def plot_cluster(traj_lst, cluster_lst):
    '''
    Plots given trajectories with a color that is specific for every trajectory's own cluster index.
    Outlier trajectories which are specified with -1 in `cluster_lst` are plotted dashed with black color
    '''
    sns.set()
    plt.rcParams['figure.figsize'] = (12, 12)

    color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_lst.extend(['firebrick', 'olive', 'indigo', 'khaki', 'teal', 'saddlebrown',
                      'skyblue', 'coral', 'darkorange', 'lime', 'darkorchid', 'dimgray'])

    cluster_count = np.max(cluster_lst) + 1

    for traj, cluster in zip(traj_lst, cluster_lst):

        if cluster == -1:
            # Means it it a noisy trajectory, paint it black
            plt.plot(traj[:, 0], traj[:, 1], c='k', linestyle='dashed')

        else:
            plt.plot(traj[:, 0], traj[:, 1], c=color_lst[cluster % len(color_lst)])
    plt.show()

def plot_all_traj_indiv(traj_lst):
    while True:
        for traj in traj_lst:
            plt.plot(traj[:, 0], traj[:, 1])
            plt.show()

            ans = input("y for next trajectory, n to break the loop")
            if (ans.lower() == 'n'):
                return


if __name__=="__main__":
    path = r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis_code_laurens_inez\Thesis-Inez-20220713T084610Z-001\Thesis-Inez\position_result.csv"
    data = pd.read_csv(path)

    traj_lst = change_format(data)
    #plot_all_traj(traj_lst)

    # plot traj 1 by 1
    plot_all_traj_indiv(traj_lst)
    #
    # traj_lst = traj_segmentation(traj_lst)
    # D = distance_matrix(traj_lst)
    #
    # #120 and 4 worked decent, (120, 6) also
    # # try low eps and high min sample
    # mdl = DBSCAN(eps=120, min_samples=4)
    # cluster_lst = mdl.fit_predict(D)
    #
    # print(cluster_lst)
    # print(len(cluster_lst))
    # # np.save("cluster_lst.npy", cluster_lst)
    #
    #
    # # cluster_lst = cluster_lst[cluster_lst == -1]
    #
    # plot_cluster(traj_lst, cluster_lst)
    #
    # # has paper with references
    # #https://medium.com/@tarammullin/dbscan-parameter-estimation-ff8330e3a3bd
