import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_data(data_dir):
    data = pd.read_csv(data_dir)
    np_data = data.values.astype('float32')

    data_img = np_data[:, 1:]
    data_img /= 255

    data_label = np_data[:, 0]
    pos_mask = (data_label >= 5)
    neg_mask = (data_label < 5)
    data_label[pos_mask] = 1
    data_label[neg_mask] = -1

    print("Done!")

    return torch.from_numpy(data_img), torch.from_numpy(data_label)


class SVM:
    def __init__(self):
        super(SVM, self).__init__()

        self.x, self.y = load_data('../data/mnist_train.csv')
        self.x, self.y = self.x[:1000], self.y[:1000]  # 1000 examples

        self.test_img, self.test_label = load_data('../data/mnist_test.csv')
        self.test_img, self.test_label = self.test_img[:1000], self.test_label[:1000]

        self.n = self.x.shape[0]
        self.alpha = torch.zeros((self.n,))
        self.E = (-self.y)
        self.b = 0
        self.c = 200
        self.sigma = 10
        self.eps = 0.001
        self.k = self.k_i_j()
        self.epochs = 20

    def k_i_j(self):
        k = torch.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i, self.n):
                x = self.x[i] - self.x[j]
                k[i][j] = torch.exp(-torch.dot(x, x) / (2 * (self.sigma ** 2)))
                k[j][i] = k[i][j]
        print('kernel method!')
        return k

    def g_x_i(self, i):
        g = 0
        for j in range(self.n):
            g += self.alpha[j] * self.y[j] * self.k[j][i]
        g += self.b
        return g

    def E_i(self, i):
        return self.g_x_i(i) - self.y[i]

    def is_kkt(self, i):
        if torch.abs(self.alpha[i]) < self.eps and self.y[i] * self.g_x_i(i) >= 1:
            return True
        if -self.eps < self.alpha[i] and self.alpha[i] < (self.c + self.eps) and torch.abs(
                self.y[i] * self.g_x_i(i) - 1) < self.eps:
            return True
        if torch.abs(self.alpha[i] - self.c) < self.eps and self.y[i] * self.g_x_i(i) <= 1:
            return True
        return False

    def get_max_E_j(self, i):
        E_1 = self.E_i(i)
        E_2 = 0
        maxE = -1
        index = 0
        for j in range(self.n):
            E_j = self.E_i(j)
            if torch.abs(E_1 - E_j) > maxE:
                maxE = torch.abs(E_1 - E_j)
                E_2 = E_j
                index = j
        return E_2, index

    def train(self):
        for epoch in range(self.epochs):
            print(f'epoch: {epoch}')
            for i in tqdm(range(self.n)):
                if self.is_kkt(i) == False:
                    E_1 = self.E_i(i)
                    E_2, j = self.get_max_E_j(i)

                    alpha_1_old = self.alpha[i]
                    alpha_2_old = self.alpha[j]
                    y_1 = self.y[i]
                    y_2 = self.y[j]

                    if y_1 != y_2:
                        l = max(0, alpha_2_old - alpha_1_old)
                        h = min(self.c, self.c + alpha_2_old - alpha_1_old)
                    else:
                        l = max(0, alpha_2_old + alpha_1_old - self.c)
                        h = min(self.c, alpha_2_old + alpha_1_old)
                    if l == h:
                        continue

                    k_11 = self.k[i][i]
                    k_12 = self.k[i][j]
                    k_21 = self.k[j][i]
                    k_22 = self.k[j][j]

                    alpha_2_new = alpha_2_old + ((y_2 * (E_1 - E_2)) / (k_11 + k_22 - 2 * k_12))
                    alpha_2_new = max(l, alpha_2_new)
                    alpha_2_new = min(h, alpha_2_new)

                    alpha_1_new = alpha_1_old + y_1 * y_2 * (alpha_2_old - alpha_2_new)

                    b_old = self.b
                    b_1_new = -E_1 - y_1 * k_11 * (alpha_1_new - alpha_1_old) - y_2 * k_21 * (
                            alpha_2_new - alpha_2_old) + b_old
                    b_2_new = -E_2 - y_1 * k_12 * (alpha_1_new - alpha_1_old) - y_2 * k_22 * (
                            alpha_2_new - alpha_2_old) + b_old
                    if 0 < alpha_1_new and alpha_1_new < self.c:
                        b_new = b_1_new
                    elif 0 < alpha_2_new and alpha_2_new < self.c:
                        b_new = b_2_new
                    else:
                        b_new = (b_1_new + b_2_new) / 2

                    self.alpha[i] = alpha_1_new
                    self.alpha[j] = alpha_2_new
                    self.b = b_new
                    self.E[i] = E_1
                    self.E[j] = E_2

    def test_k_i_j(self, i, j):
        x = self.test_img[i] - self.x[j]
        k = torch.exp(-torch.dot(x, x) / (2 * (self.sigma ** 2)))
        return k

    def test(self):
        cnt = 0
        for i in tqdm(range(self.test_img.shape[0])):
            y_pred = 0
            for j in range(self.x.shape[0]):
                k = self.alpha[j] * self.y[j] * self.test_k_i_j(i, j)
                y_pred += k
            y_pred += self.b
            y_pred = torch.sign(y_pred)
            if y_pred == self.test_label[i]:
                cnt += 1
        return float(cnt) / self.test_img.shape[0]


def main():
    # train_img, train_label = load_data('../data/mnist_train.csv')
    # print(train_img.shape, train_label.shape)
    # test_img, test_label = load_data('../data/mnist_test.csv')
    # print(test_img.shape, test_label.shape)
    #
    # # sample 3000 examples to train and 1000 examples to test
    # train_img, train_label = train_img[:3000], train_label[:3000]
    # test_img, test_label = test_img[:1000], test_label[:1000]

    svm = SVM()
    svm.train()
    test_acc = svm.test()
    print(test_acc)


if __name__ == '__main__':
    main()
