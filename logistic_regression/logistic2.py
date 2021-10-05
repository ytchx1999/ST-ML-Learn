import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression


def load_data(data_dir):
    data = pd.read_csv(data_dir)
    np_data = data.values.astype('float32')

    data_img = np_data[:, 1:]
    data_img /= 255

    data_label = np_data[:, 0]
    pos_mask = (data_label >= 5)
    neg_mask = (data_label < 5)
    data_label[pos_mask] = 1
    data_label[neg_mask] = 0

    return data_img, data_label


def main():
    train_img, train_label = load_data('../data/mnist_train.csv')
    print(train_img.shape, train_label.shape)
    test_img, test_label = load_data('../data/mnist_test.csv')
    print(test_img.shape, test_label.shape)

    # sample 3000 examples to train and 1000 examples to test
    train_img, train_label = train_img[:3000], train_label[:3000]
    test_img, test_label = test_img[:1000], test_label[:1000]

    model = LogisticRegression(max_iter=1000)
    model.fit(train_img, train_label)

    test_acc = model.score(test_img, test_label)
    print(test_acc)

    # (59999, 784) (59999,)
    # (9999, 784) (9999,)
    # 0.82


if __name__ == '__main__':
    main()
