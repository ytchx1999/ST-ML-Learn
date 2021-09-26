import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


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

    return torch.from_numpy(data_img), torch.from_numpy(data_label)


def main():
    train_img, train_label = load_data('../data/mnist_train.csv')
    print(train_img.shape, train_label.shape)
    test_img, test_label = load_data('../data/mnist_test.csv')
    print(test_img.shape, test_label.shape)

    # sample 1000 examples to train
    train_img, train_label = train_img[:1000], train_label[:1000]

    num_classes = 10
    tot_train = train_img.shape[0]

    lamda = 1
    s_j = 2

    # train
    # P(y=c_k)
    p_y = torch.zeros((num_classes,), dtype=torch.float)
    # P(x^(j)=a_j | y=c_k)
    p_x_y = torch.zeros((num_classes, train_img.shape[1], s_j), dtype=torch.float)

    for i in tqdm(range(train_img.shape[0])):
        c_k = train_label[i]  # label
        x = train_img[i]  # feature
        p_y[c_k] += 1
        for j in range(train_img.shape[1]):
            # Num(x^(j)=train_img[i][j], y=train_label[i])
            p_x_y[c_k][j][x[j]] += 1

    # laplacian smoothing
    for i in range(num_classes):
        p_x_y[i] += lamda
        p_x_y[i] /= (p_y[i] + s_j * lamda)

        p_y[i] += lamda
        p_y[i] /= (tot_train + num_classes * lamda)

    # log(), to prevent overflow
    p_y, p_x_y = torch.log(p_y), torch.log(p_x_y)
    print(p_y.shape)
    print(p_x_y.shape)

    # sample 1000 examples to test
    test_img, test_label = test_img[:1000], test_label[:1000]

    num_correct = 0
    tot_test = test_img.shape[0]

    # test
    for i in tqdm(range(test_img.shape[0])):
        # init pred array
        y_pred = torch.zeros((num_classes,), dtype=torch.float)
        x = test_img[i]  # feature

        for j in range(num_classes):
            # y_pred[j] = log(P(y=c_k) * mul_(P(X^(j)=x^(j) | y=c_k)))
            # = log(P(y=c_k)) + sum_(log(P(X^(j)=x^(j) | y=c_k)))
            y_pred[j] += p_y[j]
            for k in range(test_img.shape[1]):
                y_pred[j] += (p_x_y[j][k][x[k]])

        y = y_pred.argmax()  # argmax
        if y == test_label[i]:
            num_correct += 1

    print(float(num_correct) / tot_test)

    # torch.Size([59999, 784]) torch.Size([59999])
    # torch.Size([9999, 784]) torch.Size([9999])
    # 100%|██████████| 1000/1000 [00:13<00:00, 73.95it/s]
    # torch.Size([10])
    # torch.Size([10, 784, 2])
    # 100%|██████████| 1000/1000 [01:50<00:00,  9.04it/s]
    # 0.755


if __name__ == '__main__':
    main()
