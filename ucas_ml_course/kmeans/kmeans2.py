# A1(4,2,5), A2(10,5,2), A3(5,8,7),
# B1(1,1,1), B2(2,3,2), B3(3,6,9),
# C1(11,9,2),C2(1,4,6), C3(9,1,7), C4(5,6,7)

import torch
import numpy as np
from sklearn.cluster import KMeans


def main():
    x = torch.tensor([
        [4, 2, 5],
        [10, 5, 2],
        [5, 8, 7],
        [1, 1, 1],
        [2, 3, 2],
        [3, 6, 9],
        [11, 9, 2],
        [1, 4, 6],
        [9, 1, 7],
        [5, 6, 7]
    ], dtype=torch.float)
    x = x.numpy()

    centers = torch.tensor([
        [4, 2, 5],
        [1, 1, 1],
        [11, 9, 2]
    ], dtype=torch.float)
    centers = centers.numpy()

    k = 3
    model = KMeans(n_clusters=k, init=centers, n_init=1)
    model.fit(x)

    print('clusters: ', model.labels_)
    print('centers: ', model.cluster_centers_)

    # clusters:  [0 2 0 1 1 0 2 0 0 0]
    # centers:  [[ 4.5        4.5        6.8333335]
    #  [ 1.5        2.         1.5      ]
    #  [10.5        7.         2.       ]]


if __name__ == '__main__':
    main()
