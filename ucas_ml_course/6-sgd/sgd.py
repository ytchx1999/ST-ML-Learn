import torch

# [1, 2, 5, 4],
# [2, 5, 1, 2]
x = torch.tensor([
    [1, 2],
    [2, 5],
    [5, 1],
    [4, 2]
], dtype=torch.float)

y = torch.tensor([
    [19],
    [26],
    [19],
    [20]
], dtype=torch.float)

lr = 0.001

w = torch.ones((2, 1))

for e in range(100):
    for i in range(x.shape[0]):
        batch_loss = torch.dist(torch.matmul(x[i].reshape(1, -1), w).view(-1), y[i])
        print(batch_loss.item())
        for j in range(x.shape[1]):
            w[j] -= 2 * lr * (torch.matmul(x[i].reshape(1, -1), w).view(-1) - y[i]) * x[i][j]

print(w)
# tensor([[2.9181],
#         [4.4780]])

