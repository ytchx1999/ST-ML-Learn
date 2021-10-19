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
    delta = torch.zeros(w.shape)
    loss = 0
    for i in range(x.shape[0]):
        loss += torch.dist(torch.matmul(x[i].reshape(1, -1), w).view(-1), y[i])
        for j in range(x.shape[1]):
            delta[j] += (torch.matmul(x[i].reshape(1, -1), w).view(-1) - y[i]) * x[i][j]
    print(loss.item() / x.shape[0])

    for j in range(x.shape[1]):
        w[j] -= 2 * lr * delta[j]

print(w)
# tensor([[2.9199],
#         [4.4995]])

