import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_data(data_dir):
    data = pd.read_csv(data_dir)
    np_data = data.values.astype('float32')

    # data_img = np_data[:, 1:]
    # data_img /= 255
    data_img = np_data[:, 1:].copy().astype('long')
    # process img value of 0(<128) or 1(>=128)
    one_mask = (data_img >= 128)
    zero_mask = (data_img < 128)
    data_img[one_mask] = 1
    data_img[zero_mask] = 0

    data_label = np_data[:, 0]
    pos_mask = (data_label >= 5)
    neg_mask = (data_label < 5)
    data_label[pos_mask] = 1
    data_label[neg_mask] = -1

    print("Done!")

    return torch.from_numpy(data_img), torch.from_numpy(data_label)


class AdaBoost:
    def __init__(self):
        super(AdaBoost, self).__init__()
        self.x, self.y = load_data('../data/mnist_train.csv')
        self.x, self.y = self.x[:100], self.y[:100]
        # print(train_img.shape, train_label.shape)
        self.test_img, self.test_label = load_data('../data/mnist_test.csv')
        self.test_img, self.test_label = self.test_img[:100], self.test_label[:100]
        # print(test_img.shape, test_label.shape)

        self.n = self.x.shape[0]
        self.D = torch.zeros((self.n,))
        self.D.fill_(1. / self.n)

        self.m = 20
        self.threshold = torch.zeros((self.m,))  # {x > threshold, x < threshold}
        self.rule = torch.zeros((self.m,))  # 0: {-1, 1}   1: {1, -1}
        self.e = torch.zeros((self.m,))
        self.dim = torch.zeros((self.m,), dtype=torch.long)
        self.alpha = torch.zeros((self.m,))

        # self.epochs = 20

    def make_cls(self, d, threshold, rule):
        e = 0
        g = torch.zeros((self.n,))
        if rule == 0:
            l = -1
            h = 1
        else:
            l = 1
            h = -1
        for i in range(self.n):
            if self.x[i][d] > threshold:
                g[i] = l
            else:
                g[i] = h
            if g[i] != self.y[i]:
                # e_m  = \sum{w_mi}
                e += self.D[i]
        return e, g

    def train(self):
        for m in tqdm(range(self.m)):  # m G(x)
            e_min = self.n
            rule_min = None
            threshold_min = None
            dim_min = None
            g_x = None

            for d in range(self.x.shape[1]):
                for threshold in [-0.5, 0.5, 1.5]:
                    for rule in [0, 1]:
                        e, g = self.make_cls(d, threshold, rule)
                        if e < e_min:
                            e_min = e
                            g_x = g
                            rule_min = rule
                            threshold_min = threshold
                            dim_min = d

            # alpha = 0.5 * log((1 - e_m) / e_m)
            alpha = 0.5 * torch.log((1 - e_min) / e_min)
            # w_m+1,i = w_mi * exp(-alpha * y_i * G(x_i)) / z_m
            w = torch.mul(self.D, torch.exp(-alpha * torch.mul(self.y, g_x)))
            w = w / torch.sum(w)

            self.D = w
            self.threshold[m] = threshold_min
            self.rule[m] = rule_min
            self.e[m] = e_min
            self.dim[m] = dim_min
            self.alpha[m] = alpha

    def pred(self, d, threshold, rule):
        e = 0
        g = torch.zeros((self.test_img.shape[0],))
        if rule == 0:
            l = -1
            h = 1
        else:
            l = 1
            h = -1
        for i in range(self.test_img.shape[0]):
            if self.test_img[i][d] > threshold:
                g[i] = l
            else:
                g[i] = h
            if g[i] != self.test_label[i]:
                e += self.D[i]
        return e, g

    def test(self):
        y_pred = torch.zeros((self.test_img.shape[0],))

        for m in tqdm(range(self.m)):
            e, g = self.pred(self.dim[m], self.threshold[m], self.rule[m])
            y_pred += (self.alpha[m] * g)

        y_pred = torch.sign(y_pred)
        correct = (y_pred == self.test_label).sum().item()
        return correct / self.test_img.shape[0]


def main():
    # train_img, train_label = load_data('../data/mnist_train.csv')
    # print(train_img.shape, train_label.shape)
    # test_img, test_label = load_data('../data/mnist_test.csv')
    # print(test_img.shape, test_label.shape)
    #
    # # sample 3000 examples to train and 1000 examples to test
    # train_img, train_label = train_img[:3000], train_label[:3000]
    # test_img, test_label = test_img[:1000], test_label[:1000]

    adaboost = AdaBoost()
    adaboost.train()
    test_acc = adaboost.test()
    print(test_acc)

    # Done!
    # Done!
    # 100%|██████████| 20/20 [04:14<00:00, 12.74s/it]
    # 100%|██████████| 20/20 [00:00<00:00, 383.28it/s]
    # 0.53


if __name__ == '__main__':
    main()
