# 2、编写两类正态分布模式的贝叶斯分类程序。（可选例题或以下模式集进行测试设，这里P(ω1)= P(ω2)=1/2）
#
#  ω1：{(0 0)T, (2 0)T, (2 2)T, (0 2)T}
#
#  ω2：{(4 4)T, (6 4)T, (6 6)T, (4 6)T}

import torch
import numpy as np
import matplotlib.pyplot as plt


def main():
    x1 = torch.tensor([
        [0, 2, 2, 0],
        [0, 0, 2, 2]
    ], dtype=torch.float)
    x2 = torch.tensor([
        [4, 6, 6, 4],
        [4, 4, 6, 6]
    ], dtype=torch.float)

    m1 = torch.mean(x1, dim=1, keepdim=True)
    print(m1)
    # print(m1.shape)
    m2 = torch.mean(x2, dim=1, keepdim=True)
    print(m2)
    # print(m2.shape)

    c1 = torch.matmul((x1 - m1), (x1 - m1).transpose(0, 1)) / x1.shape[1]
    print(c1)
    c2 = torch.matmul((x2 - m2), (x2 - m2).transpose(0, 1)) / x2.shape[1]
    print(c2)

    c = c1.clone()
    c_inv = torch.inverse(c)
    print(c_inv)

    k = torch.matmul((m1 - m2).transpose(0, 1), c_inv).squeeze()

    b = -0.5 * torch.matmul(torch.matmul(m1.transpose(0, 1), c_inv), m1) + 0.5 * torch.matmul(
        torch.matmul(m2.transpose(0, 1), c_inv), m2)

    d_x = ''
    for i in range(k.shape[0]):
        d_x += f'{k[i]} * x_{i + 1} '
    d_x += '+ ' + str(b.item())
    print('d(x)=', d_x)

    # d(x)= -4.0 * x_1 -4.0 * x_2 + 24.0

    x = np.linspace(-1, 10, 100)
    y = (k[0].item() * x + b.item()) / -k[1].item()
    plt.plot(x, y)

    plt.scatter(x1[0], x1[1])
    plt.scatter(x2[0], x2[1])
    plt.savefig('./img/naive_bayes.png')
    plt.show()


if __name__ == '__main__':
    main()
