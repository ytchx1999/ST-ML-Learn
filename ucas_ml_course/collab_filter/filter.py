#        Product 1 Product 2 Product 3 Product 4
# User 1	1	1	5	3
# User 2	3	？	5	4
# User 3	1	3	1	1
# User 4	4	3	2	1
# User 5	2	2	2	4

import torch


def main():
    pred_user = 2
    pred_product = 2
    neigh_k = 3

    user_idx = pred_user - 1  # index
    product_idx = pred_product - 1  # index
    ups = torch.tensor([
        [1, 1, 5, 3],
        [3, 0, 5, 4],
        [1, 3, 1, 1],
        [4, 3, 2, 1],
        [2, 2, 2, 4]
    ], dtype=torch.float)

    idx = [0, 2, 3]
    x = ups[:, idx]

    # user-user cosine similarity
    cos_sim = torch.zeros((x.shape[0],))
    for i in range(x.shape[0]):
        cos_sim[i] = torch.dot(x[i], x[user_idx]) / (torch.norm(x[i], p=2) * torch.norm(x[user_idx], p=2))
    print('Cosine similarity of user 2: ', cos_sim)

    sim_val, sim_idx = torch.topk(cos_sim, k=(neigh_k + 1))
    sim_idx = sim_idx[1:]
    print(f'Top 3 similar users of user 2: User_{sim_idx[0] + 1}, User_{sim_idx[1] + 1}, User_{sim_idx[2] + 1}.')
    # print(sim_idx)

    # average rating
    print('')
    avg_r = torch.mean(ups, dim=1)
    print('average rating: ', avg_r)

    # compute rating
    sum_up = 0
    sum_down = 0
    for i in sim_idx:
        sum_up += cos_sim[i] * (ups[i][product_idx] - avg_r[i])
        sum_down += cos_sim[i]
    rate = avg_r[user_idx] + (sum_up / sum_down)
    print('User 2’s rating for Product 2: ', rate.item())

    # Cosine similarity of user 2:  tensor([0.9562, 1.0000, 0.9798, 0.8024, 0.9238])
    # Top 3 similar users of user 2: User_3, User_1, User_5.
    #
    # average rating:  tensor([2.5000, 3.0000, 1.5000, 2.5000, 2.5000])
    # User 2’s rating for Product 2:  2.850874423980713


if __name__ == '__main__':
    main()
