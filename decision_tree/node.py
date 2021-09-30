import torch


class node():
    def __init__(self, val, left=None, right=None):
        super(node, self).__init__()
        self.val = val
        self.left = left
        self.right = right


if __name__ == '__main__':
    leaf_1 = node(1)
    print(leaf_1.val, leaf_1.left)
