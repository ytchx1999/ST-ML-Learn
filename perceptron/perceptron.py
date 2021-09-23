import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_data(data_dir):
    data = pd.read_csv(data_dir)
    np_data = data.values.astype('float')

    data_img = np_data[:, 1:]
    data_img /= 255

    data_label = np_data[:, 0]
    pos_mask = (data_label >= 5)
    neg_mask = (data_label < 5)
    data_label[pos_mask] = 1
    data_label[neg_mask] = -1

    return torch.from_numpy(data_img), torch.from_numpy(data_label)


def main():
    train_img, train_label = load_data('../data/mnist_train.csv')
    print(train_img.shape, train_label.shape)
    test_img, test_label = load_data('../data/mnist_test.csv')
    print(test_img.shape, test_label.shape)

    w = torch.zeros((1, train_img.shape[1]), dtype=torch.float64)
    b = 0

    lr = 0.0001

    # train
    for iteration in tqdm(range(30)):
        for i in range(len(train_img)):
            delta = train_label[i] * (torch.matmul(w, train_img[i].reshape(-1, 1)) + b)
            if delta <= 0:
                w += (lr * train_label[i] * train_img[i])
                b += (lr * train_label[i])

    # test
    err_num = 0
    total = test_img.shape[0]
    for i in range(len(test_img)):
        delta = test_label[i] * (torch.matmul(w, test_img[i].reshape(-1, 1)) + b)
        if delta <= 0:
            err_num += 1

    test_acc = float(total - err_num) / total
    print(test_acc)

    # torch.Size([59999, 784]) torch.Size([59999])
    # torch.Size([9999, 784]) torch.Size([9999])
    # 100%|██████████| 30/30 [01:08<00:00,  2.28s/it]
    # 0.7986798679867987


if __name__ == '__main__':
    main()
