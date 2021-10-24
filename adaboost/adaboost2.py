import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.ensemble import AdaBoostClassifier


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

    return data_img, data_label


def main():
    train_img, train_label = load_data('../data/mnist_train.csv')
    print(train_img.shape, train_label.shape)
    test_img, test_label = load_data('../data/mnist_test.csv')
    print(test_img.shape, test_label.shape)

    # sample 3000 examples to train and 1000 examples to test
    train_img, train_label = train_img[:100], train_label[:100]
    test_img, test_label = test_img[:100], test_label[:100]

    model = AdaBoostClassifier(n_estimators=20)
    model.fit(train_img, train_label)
    test_acc = model.score(test_img, test_label)
    print(test_acc)

    # Done!
    # (59999, 784) (59999,)
    # Done!
    # (9999, 784) (9999,)
    # 0.65


if __name__ == '__main__':
    main()
