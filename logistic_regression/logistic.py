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
    data_label[neg_mask] = 0

    return torch.from_numpy(data_img), torch.from_numpy(data_label)


def main():
    train_img, train_label = load_data('../data/mnist_train.csv')
    print(train_img.shape, train_label.shape)
    test_img, test_label = load_data('../data/mnist_test.csv')
    print(test_img.shape, test_label.shape)

    # sample 3000 examples to train and 1000 examples to test
    train_img, train_label = train_img[:3000], train_label[:3000]
    test_img, test_label = test_img[:1000], test_label[:1000]

    # 增广
    train_img = torch.cat([train_img, torch.ones(train_img.shape[0], 1)], dim=1)
    test_img = torch.cat([test_img, torch.ones(test_img.shape[0], 1)], dim=1)

    w = torch.zeros(1, train_img.shape[1])
    c = 0.001

    # train
    for epoch in tqdm(range(300)):
        for i in range(train_img.shape[0]):
            e_wx = torch.exp(torch.matmul(w, train_img[i].reshape(-1, 1)))
            # delta = y_i * x_i - x_i * exp(w * x_i) / (1 + exp(w * x_i))
            delta = (train_label[i] * train_img[i]) - (train_img[i] * e_wx) / (1 + e_wx)
            w += c * delta

    # test
    cnt = 0
    for i in tqdm(range(test_img.shape[0])):
        e_wx = torch.exp(torch.matmul(w, test_img[i].reshape(-1, 1)))
        # P(y=0 | x) = 1 / (1 + exp(w * x))
        # P(y=1 | x) = 1 - P(y=0 | x)
        p_y0_x = 1 / (1 + e_wx)

        if p_y0_x > 0.5:
            y_pred = 0
        else:
            y_pred = 1

        if y_pred == test_label[1]:
            cnt += 1

    print(float(cnt) / test_img.shape[0])

    # torch.Size([59999, 784]) torch.Size([59999])
    # torch.Size([9999, 784]) torch.Size([9999])
    # 100%|██████████| 300/300 [00:44<00:00,  6.76it/s]
    # 100%|██████████| 1000/1000 [00:00<00:00, 18496.26it/s]
    # 0.572


if __name__ == '__main__':
    main()
