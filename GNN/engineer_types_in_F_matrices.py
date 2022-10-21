import pandas as pd

path = r"C:\Users\caspe\OneDrive\Documenten\MSc TOM\Thesis TOM\GNN\10212022_012528.csv"
data = pd.read_csv(path)

print(data["id"].unique())
data.groupby("id")
# print smallest t for each id


# pseudo code
# have csv file with trajectories
# have a list with type of each trajectory
# output n arrays with occupancies grid based on the n engineer types

# step 1:
# split trajectory file based on type

# step 2:
# from trajectory to occupancy grid
# reverse the trajectory creation

# assuming t of each trajectory is kept
# loop over t
# for each t place back on the occupancy with the recorded row pos and col pos
# col and row pos can be in the middle of grid cell --> 5.5 --> take this into account --> =1 in both
#
# how do we include node features
# place t's back in the (T, 10000, 2) array --> first clear it with zeros

