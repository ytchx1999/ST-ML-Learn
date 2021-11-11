# A1(4,2,5), A2(10,5,2), A3(5,8,7),
# B1(1,1,1), B2(2,3,2), B3(3,6,9),
# C1(11,9,2),C2(1,4,6), C3(9,1,7), C4(5,6,7)

import torch


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

    print(x.shape)

    k = 3
    clusters = torch.zeros((x.shape[0],), dtype=torch.long)
    centers = torch.tensor([
        [4, 2, 5],
        [1, 1, 1],
        [11, 9, 2]
    ], dtype=torch.float)
    print(clusters)

    iteration = 0
    while True:
        print('')
        print('iteration: ', iteration)

        # dist
        dist = torch.zeros((centers.shape[0], x.shape[0]))
        for i in range(centers.shape[0]):
            for j in range(x.shape[0]):
                dist[i][j] = torch.dist(centers[i], x[j])
        print('dist: ', dist)

        clusters_new = torch.argmin(dist, dim=0)
        print('clusters:     ', clusters)
        print('clusters_new: ', clusters_new)
        if torch.equal(clusters, clusters_new):
            break
        else:
            clusters = clusters_new.clone()
            cnt = torch.zeros((centers.shape[0]))
            centers_new = torch.zeros_like(centers)

            # new centers
            for j in range(x.shape[0]):
                idx = clusters[j]
                cnt[idx] += 1
                centers_new[idx] += x[j]
            for i in range(centers_new.shape[0]):
                centers_new[i] /= cnt[i]

            centers = centers_new.clone()
            print('centers: ', centers)

        iteration += 1

    # iteration:  0
    # dist:  tensor([[ 0.0000,  7.3485,  6.4031,  5.0990,  3.7417,  5.7446, 10.3441,  3.7417,
    #           5.4772,  4.5826],
    #         [ 5.0990,  9.8995, 10.0499,  0.0000,  2.4495,  9.6437, 12.8452,  5.8310,
    #          10.0000,  8.7750],
    #         [10.3441,  4.1231,  7.8740, 12.8452, 10.8167, 11.0454,  0.0000, 11.8743,
    #           9.6437,  8.3666]])
    # clusters:      tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # clusters_new:  tensor([0, 2, 0, 1, 1, 0, 2, 0, 0, 0])
    # centers:  tensor([[ 4.5000,  4.5000,  6.8333],
    #         [ 1.5000,  2.0000,  1.5000],
    #         [10.5000,  7.0000,  2.0000]])
    #
    # iteration:  1
    # dist:  tensor([[ 3.1402,  7.3390,  3.5395,  7.6503,  5.6446,  3.0322,  9.2661,  3.6324,
    #           5.7033,  1.5899],
    #         [ 4.3012,  9.0277,  8.8600,  1.2247,  1.2247,  8.6313, 11.8110,  4.9497,
    #           9.3541,  7.6485],
    #         [ 8.7321,  2.0616,  7.5000, 11.2805,  9.3941, 10.3078,  2.0616, 10.7355,
    #           7.9530,  7.5000]])
    # clusters:      tensor([0, 2, 0, 1, 1, 0, 2, 0, 0, 0])
    # clusters_new:  tensor([0, 2, 0, 1, 1, 0, 2, 0, 0, 0])


if __name__ == '__main__':
    main()
