import numpy as np
import os
import pandas as pd
from itertools import chain
import time


def getfiles():
    # Retrieve all directories in current folder
    all_files = {}
    directories_in_curdir = list(filter(os.path.isdir, os.listdir(os.curdir)))
    for dir in directories_in_curdir:
        all_files[dir] = []
        # Already done
        # if dir != 'simulation1-50p-100cm' and dir != 'simulation1-100p-100cm':
        # Retrieve all files in those directories
        for filename in os.scandir(dir):
            if filename.is_file():
                if filename.path[-4:] == '.txt':
                    # Gets every file's path and append it to a dictionary in which the key is the directory
                    all_files[dir].append(filename.path)

    return all_files

def main(fls: list, res: int):
    # construct 3d array
    f = fls[0]
    data = np.loadtxt(f, dtype=int, usecols=0)
    data = np.array(np.array_split(data, res))
    f = fls[1]
    x = np.loadtxt(f, dtype=int, usecols=0)
    x = np.array(np.array_split(x, res))
    data = np.stack((data, x))
    print("3d array")

    # stack all other files on the 3d array
    for f in fls[2:]:
        x = np.loadtxt(f, dtype=int, usecols=0)
        x = np.array(np.array_split(x, res))
        data = np.vstack((data, x[None, :, :]))
        print(data.shape)

    return data

if __name__ == "__main__":
    t0 = time.time()

    fls = getfiles()

    # # remove all files with resolution 100
    # remove = [k for k in fls if k[-5:] == "100cm"]
    # for k in remove: del fls[k]
    #
    # # get all paths to 1 large list
    # fls = list(fls.values())
    # fls = list(chain.from_iterable(fls))
    # data = main(fls, res=200)
    #
    # # store as .npy
    # np.save("data_50cm", data)
    #
    # t1 = time.time()
    # print("Total time to execute: ", t1-t0)
    #
    # # x = np.load("inputdata.npy")

