import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.svm import SVC


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

    return data_img, data_label


def main():
    train_img, train_label = load_data('../data/mnist_train.csv')
    print(train_img.shape, train_label.shape)
    test_img, test_label = load_data('../data/mnist_test.csv')
    print(test_img.shape, test_label.shape)

    train_img, train_label = train_img[:100], train_label[:100]
    test_img, test_label = test_img[:100], test_label[:100]

    # 也可以使用一下参数 C=200, kernel='rbf', gamma=10, max_iter=20
    # 这里直接使用默认参数
    model = SVC()
    model.fit(train_img, train_label)
    test_acc = model.score(test_img, test_label)
    print(test_acc)

    # Done!
    # (59999, 784) (59999,)
    # Done!
    # (9999, 784) (9999,)
    # 0.77


if __name__ == '__main__':
    main()
