import numpy as np

def find_frontiers(occupancy, coverage):
    frontiers = []
    H, W = occupancy.shape

    for i in range(1, H-1):
        for j in range(1, W-1):

            if occupancy[i,j] != 0:     # skip unknown + occupied
                continue

            if coverage[i,j] == 1:      # skip visited
                continue

            neigh = occupancy[i-1:i+2, j-1:j+2]

            if np.any(neigh == -1):     # if ANY cell around is unknown region
                frontiers.append((j, i))

    return frontiers
