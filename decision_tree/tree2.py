import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier


def load_data(data_dir):
    data = pd.read_csv(data_dir)
    np_data = data.values

    data_img = np_data[:, 1:].copy().astype('long')
    # process img value of 0(<128) or 1(>=128)
    one_mask = (data_img >= 128)
    zero_mask = (data_img < 128)
    data_img[one_mask] = 1
    data_img[zero_mask] = 0

    data_label = np_data[:, 0].copy().astype('long')

    return data_img, data_label


def main():
    train_img, train_label = load_data('../data/mnist_train.csv')
    print(train_img.shape, train_label.shape)
    test_img, test_label = load_data('../data/mnist_test.csv')
    print(test_img.shape, test_label.shape)

    # sample 3000 examples to train and 1000 examples to test
    train_img, train_label = train_img[:3000], train_label[:3000]
    test_img, test_label = test_img[:1000], test_label[:1000]

    model = DecisionTreeClassifier()
    model.fit(train_img, train_label)

    test_acc = model.score(test_img, test_label)
    print(test_acc)

    # (59999, 784) (59999,)
    # (9999, 784) (9999,)
    # 0.672


if __name__ == '__main__':
    main()
