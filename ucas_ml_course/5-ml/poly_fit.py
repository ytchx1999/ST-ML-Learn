import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

class PolyFit(nn.Module):
    def __init__(self, k):
        super(PolyFit, self).__init__()
        self.w = nn.Linear(k, 1)

    def forward(self, x):
        return self.w(x)


def main():
    k = 3
    x = torch.linspace(0, math.pi, 10).reshape(-1, 1)
    y = torch.sin(x) + 0.02 * torch.randn(x.size())
    print(x.shape)
    print(y.shape)
    # plt.plot(x, y)
    # plt.show()

    x_k = []
    for i in range(k):
        x_k.append(x ** (i + 1))
    x_k = torch.cat(x_k, dim=1)
    print(x_k.shape)

    model = PolyFit(k)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for e in range(300):
        out = model(x_k)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())

    w = model.w.weight.data.detach()
    print(w.shape)
    y_pred = torch.matmul(x_k, w.transpose(0, 1)).reshape(-1, 1)
    print(y_pred.shape)

    plt.plot(x, y)
    plt.plot(x, y_pred)
    plt.show()


if __name__ == '__main__':
    main()
