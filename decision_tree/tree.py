import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from decision_tree.node import node
import math
import copy


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


def max_class(label):
    max_label, _ = torch.mode(label)
    return max_label


def g_D_A(img, label, label_set):
    h_d_mat = torch.zeros((img.shape[1],))
    h_d_a_mat = torch.zeros((img.shape[1],))

    for i in range(img.shape[1]):
        # H_D
        h_d = 0
        for j in label_set:
            c_k = (label == j).sum().item()
            d = label.shape[0]
            h_d += -(float(c_k) / d) * math.log2(float(c_k) / d)
        h_d_mat[i] = h_d

        # H(D | A)
        h_d_a = 0
        feat = img[:, i]
        feat_set = {i.item() for i in feat}
        for j in feat_set:
            # |D_j| and |D|
            d_j = (feat == j).sum().item()
            d = label.shape[0]

            mask = (feat == j)
            label_part = label[mask]
            label_part_set = {i.item() for i in label_part}

            # H(D_j)
            h_d_j = 0
            for k in label_part_set:
                d_jk = (label_part == k).sum().item()
                h_d_j += (float(d_jk) / d_j) * math.log2(float(d_jk) / d_j)

            h_d_a += -(float(d_j) / d) * h_d_j
        h_d_a_mat[i] = h_d_a

    # g(D, A)
    g_d_a = h_d_mat - h_d_a_mat
    # max g(D, A)
    ag, index = torch.max(g_d_a.reshape(-1, 1), dim=0)
    return ag, index


def get_D_i(img, label, index, a):
    mask = (img[:, index] == a)
    if mask.view(-1).sum().item() == 0:
        return None
    # slice img
    new_img = torch.cat([img[:, :index], img[:, index + 1:]], dim=1)
    mask_index = np.argwhere(mask.view(-1).numpy() == True)
    new_img = new_img[torch.from_numpy(mask_index).view(-1).long()]
    new_label = label[mask.view(-1)]

    return new_img, new_label


def create_d_tree(img, label, eps):
    label_set = {i.item() for i in label}

    # only have one class
    if len(label_set) == 1:
        for key in label_set:
            pass
        return node(key, left=None, right=None)  # leaf

    # feature set A = 0
    if img.shape[0] == 0:
        return node(max_class(label), left=None, right=None)  # leaf

    # get max g_d_a val, index
    ag, index = g_D_A(img, label, label_set)

    if ag < eps:
        return node(max_class(label), left=None, right=None)

    # slice to delete column index
    # left -- 0
    # right -- 1
    left_img, left_label = get_D_i(img, label, index, 0)
    right_img, right_label = get_D_i(img, label, index, 1)

    # create left, right sub_tree
    tree = node(index, left=None, right=None)
    tree.left = create_d_tree(left_img, left_label, eps)
    tree.right = create_d_tree(right_img, right_label, eps)

    return tree


def test(img, label, tree):
    cnt = 0
    for i in range(img.shape[0]):
        root = copy.deepcopy(tree)
        new_img = img[i].clone()
        index = root.val

        while not (root.left == None and root.right == None):
            if new_img[index] == 0:
                root = root.left
                new_img = torch.cat([new_img[:index], new_img[index + 1:]])
                index = root.val
            else:
                root = root.right
                new_img = torch.cat([new_img[:index], new_img[index + 1:]])
                index = root.val
        if root.val == label[i]:
            cnt += 1

    return float(cnt) / img.shape[0]


def main():
    train_img, train_label = load_data('../data/mnist_train.csv')
    print(train_img.shape, train_label.shape)
    test_img, test_label = load_data('../data/mnist_test.csv')
    print(test_img.shape, test_label.shape)

    # sample 3000 examples to train and 1000 examples to test
    train_img, train_label = train_img[:3000], train_label[:3000]
    test_img, test_label = test_img[:1000], test_label[:1000]

    label_set = {i.item() for i in train_label}
    print(label_set)

    eps = 0.1

    tree = create_d_tree(train_img, train_label, eps)

    test_acc = test(test_img, test_label, tree)
    print(test_acc)

    # torch.Size([59999, 784]) torch.Size([59999])
    # torch.Size([9999, 784]) torch.Size([9999])
    # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    # 0.711


if __name__ == '__main__':
    main()
