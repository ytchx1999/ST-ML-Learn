import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import math


def create_data(a1, u1, s1, a2, u2, s2, n):
    normal1 = torch.normal(u1, s1, size=(int(n * a1),))
    normal2 = torch.normal(u2, s2, size=(int(n * a2),))

    y = torch.cat([normal1, normal2])
    # shuffle
    idx = torch.randperm(y.nelement())
    y = y.view(-1)[idx].view(y.size())

    print("y.shape: ", y.shape)

    return y


class EM:
    def __init__(self):
        self.n = 1000
        self.a1_true = 0.3
        self.u1_true = -2
        self.s1_true = 0.5
        self.a2_true = 0.7
        self.u2_true = 0.5
        self.s2_true = 1

        self.y = create_data(self.a1_true, self.u1_true, self.s1_true,
                             self.a2_true, self.u2_true, self.s2_true, self.n)

        self.a1 = 0.5
        self.u1 = 0
        self.s1 = 1
        self.a2 = 0.5
        self.u2 = 1
        self.s2 = 1

    def gaus(self, u, s):
        phi = (1. / (math.sqrt(2 * math.pi) * s)) * torch.exp(-((self.y - u) ** 2) / (2 * (s ** 2)))
        return phi

    def e_step(self):
        res1 = self.a1 * self.gaus(self.u1, self.s1)
        res2 = self.a2 * self.gaus(self.u2, self.s2)

        gamma1 = res1 / (res1 + res2)
        gamma2 = res2 / (res1 + res2)

        return gamma1, gamma2

    def m_step(self, gamma1, gamma2):
        u1_new = torch.dot(gamma1, self.y) / torch.sum(gamma1)
        u2_new = torch.dot(gamma2, self.y) / torch.sum(gamma2)

        s1_new = torch.sqrt(torch.dot(gamma1, (self.y - self.u1) ** 2) / torch.sum(gamma1))
        s2_new = torch.sqrt(torch.dot(gamma2, (self.y - self.u2) ** 2) / torch.sum(gamma2))

        a1_new = torch.sum(gamma1) / self.n
        a2_new = torch.sum(gamma2) / self.n

        self.a1 = a1_new
        self.u1 = u1_new
        self.s1 = s1_new
        self.a2 = a2_new
        self.u2 = u2_new
        self.s2 = s2_new

    def train(self):
        for epoch in range(500):
            gamma1, gamma2 = self.e_step()
            self.m_step(gamma1, gamma2)

    def get_params(self):
        return self.a1, self.u1, self.s1, self.a2, self.u2, self.s2


def main():
    em = EM()
    print(
        f'True params: a1:{em.a1_true}, u1:{em.u1_true}, s1:{em.s1_true}, a2:{em.a2_true}, u2:{em.u2_true}, s2{em.s2_true}')
    em.train()
    a1, u1, s1, a2, u2, s2 = em.get_params()
    print(
        f'Pred params: a1:{a1.item():.4f}, u1:{u1.item():.4f}, s1:{s1.item():.4f}, a2:{a2.item():.4f}, u2:{u2.item():.4f}, s2{s2.item():.4f}')

    # y.shape:  torch.Size([1000])
    # True params: a1:0.3, u1:-2, s1:0.5, a2:0.7, u2:0.5, s21
    # Pred params: a1:0.2880, u1:-2.0430, s1:0.4646, a2:0.7120, u2:0.4321, s20.9933


if __name__ == '__main__':
    main()
