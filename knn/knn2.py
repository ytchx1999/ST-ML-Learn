import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier


def load_data(data_dir):
    data = pd.read_csv(data_dir)
    np_data = data.values

    data_img = np_data[:, 1:].copy().astype('float32')
    data_img /= 255

    data_label = np_data[:, 0].copy().astype('long')

    return data_img, data_label


def main():
    train_img, train_label = load_data('../data/mnist_train.csv')
    print(train_img.shape, train_label.shape)
    test_img, test_label = load_data('../data/mnist_test.csv')
    print(test_img.shape, test_label.shape)

    test_num = 50
    k = 25

    model = KNeighborsClassifier(n_neighbors=k, p=2)
    model.fit(train_img, train_label)

    pred_label = model.predict(test_img[:test_num])

    correct = (pred_label == test_label[:test_num]).sum()
    print(float(correct) / test_num)

    # (59999, 784) (59999,)
    # (9999, 784) (9999,)
    # 0.98


if __name__ == '__main__':
    main()
