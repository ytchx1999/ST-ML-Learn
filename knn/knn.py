import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def load_data(data_dir):
    data = pd.read_csv(data_dir)
    np_data = data.values

    data_img = np_data[:, 1:].copy().astype('float32')
    data_img /= 255

    data_label = np_data[:, 0].copy().astype('long')

    return torch.from_numpy(data_img), torch.from_numpy(data_label)


def dist(x_i, x_j):
    return torch.dist(x_i, x_j, p=2)


def main():
    train_img, train_label = load_data('../data/mnist_train.csv')
    print(train_img.shape, train_label.shape)
    test_img, test_label = load_data('../data/mnist_test.csv')
    print(test_img.shape, test_label.shape)

    test_num = 50
    k = 25
    # test test_num examples (not all)
    cnt = 0
    for i in tqdm(range(test_num)):
        d_list = []

        for j in range(len(train_img)):
            d = dist(test_img[i], train_img[j])
            d_list.append(d)

        d_list = torch.tensor(d_list)

        # topk shortest
        topk_val, topk_index = torch.topk(d_list, k, largest=False)
        topk_label = train_label[topk_index]
        # majority voting
        pred_label, _ = torch.mode(topk_label)

        if pred_label == test_label[i]:
            cnt += 1

    print(float(cnt) / test_num)

    # torch.Size([59999, 784]) torch.Size([59999])
    # torch.Size([9999, 784]) torch.Size([9999])
    # 100%|██████████| 50/50 [00:37<00:00,  1.32it/s]
    # 0.98


if __name__ == '__main__':
    main()
