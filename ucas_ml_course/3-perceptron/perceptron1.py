# 用感知器算法求下列模式分类的解向量w:
# 	ω1: {(0 0 0)T, (1 0 0)T, (1 0 1)T, (1 1 0)T}
# 	ω2: {(0 0 1)T, (0 1 1)T, (0 1 0)T, (1 1 1)T}
import torch


def main():
    w = torch.zeros((4, 1))
    c = 1

    class_1 = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0]
    ], dtype=torch.float)
    class_2 = torch.tensor([
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
        [1, 1, 1],
    ], dtype=torch.float)

    aug = torch.ones((class_1.shape[0], 1))

    class_1 = torch.cat([class_1, aug], dim=1)
    class_2 = -torch.cat([class_2, aug], dim=1)
    print(class_1)
    print(class_2)

    x = torch.cat([class_1, class_2], dim=0)
    print(x)

    iteration = 0
    while True:
        cnt = 0
        print(f'iter {iteration + 1}:')

        for i in range(x.shape[0]):
            flag = True
            delta = torch.matmul(w.transpose(0, 1), x[i].reshape(-1, 1))
            if delta <= 0:
                cnt += 1
                flag = False
                w += c * x[i].reshape(-1, 1)
            print(f'w^T * x_{i + 1}=', delta.item(), ', w^T=', w.transpose(0, 1), f', {flag}')

        iteration += 1
        if cnt == 0:
            print('Done!')
            break
        print()

    print('w^T=', w.transpose(0, 1))

    # iter 1:
    # w^T * x_1= 0.0 , w^T= tensor([[0., 0., 0., 1.]]) , False
    # w^T * x_2= 1.0 , w^T= tensor([[0., 0., 0., 1.]]) , True
    # w^T * x_3= 1.0 , w^T= tensor([[0., 0., 0., 1.]]) , True
    # w^T * x_4= 1.0 , w^T= tensor([[0., 0., 0., 1.]]) , True
    # w^T * x_5= -1.0 , w^T= tensor([[ 0.,  0., -1.,  0.]]) , False
    # w^T * x_6= 1.0 , w^T= tensor([[ 0.,  0., -1.,  0.]]) , True
    # w^T * x_7= 0.0 , w^T= tensor([[ 0., -1., -1., -1.]]) , False
    # w^T * x_8= 3.0 , w^T= tensor([[ 0., -1., -1., -1.]]) , True
    #
    # iter 2:
    # w^T * x_1= -1.0 , w^T= tensor([[ 0., -1., -1.,  0.]]) , False
    # w^T * x_2= 0.0 , w^T= tensor([[ 1., -1., -1.,  1.]]) , False
    # w^T * x_3= 1.0 , w^T= tensor([[ 1., -1., -1.,  1.]]) , True
    # w^T * x_4= 1.0 , w^T= tensor([[ 1., -1., -1.,  1.]]) , True
    # w^T * x_5= 0.0 , w^T= tensor([[ 1., -1., -2.,  0.]]) , False
    # w^T * x_6= 3.0 , w^T= tensor([[ 1., -1., -2.,  0.]]) , True
    # w^T * x_7= 1.0 , w^T= tensor([[ 1., -1., -2.,  0.]]) , True
    # w^T * x_8= 2.0 , w^T= tensor([[ 1., -1., -2.,  0.]]) , True
    #
    # iter 3:
    # w^T * x_1= 0.0 , w^T= tensor([[ 1., -1., -2.,  1.]]) , False
    # w^T * x_2= 2.0 , w^T= tensor([[ 1., -1., -2.,  1.]]) , True
    # w^T * x_3= 0.0 , w^T= tensor([[ 2., -1., -1.,  2.]]) , False
    # w^T * x_4= 3.0 , w^T= tensor([[ 2., -1., -1.,  2.]]) , True
    # w^T * x_5= -1.0 , w^T= tensor([[ 2., -1., -2.,  1.]]) , False
    # w^T * x_6= 2.0 , w^T= tensor([[ 2., -1., -2.,  1.]]) , True
    # w^T * x_7= 0.0 , w^T= tensor([[ 2., -2., -2.,  0.]]) , False
    # w^T * x_8= 2.0 , w^T= tensor([[ 2., -2., -2.,  0.]]) , True
    #
    # iter 4:
    # w^T * x_1= 0.0 , w^T= tensor([[ 2., -2., -2.,  1.]]) , False
    # w^T * x_2= 3.0 , w^T= tensor([[ 2., -2., -2.,  1.]]) , True
    # w^T * x_3= 1.0 , w^T= tensor([[ 2., -2., -2.,  1.]]) , True
    # w^T * x_4= 1.0 , w^T= tensor([[ 2., -2., -2.,  1.]]) , True
    # w^T * x_5= 1.0 , w^T= tensor([[ 2., -2., -2.,  1.]]) , True
    # w^T * x_6= 3.0 , w^T= tensor([[ 2., -2., -2.,  1.]]) , True
    # w^T * x_7= 1.0 , w^T= tensor([[ 2., -2., -2.,  1.]]) , True
    # w^T * x_8= 1.0 , w^T= tensor([[ 2., -2., -2.,  1.]]) , True
    #
    # iter 5:
    # w^T * x_1= 1.0 , w^T= tensor([[ 2., -2., -2.,  1.]]) , True
    # w^T * x_2= 3.0 , w^T= tensor([[ 2., -2., -2.,  1.]]) , True
    # w^T * x_3= 1.0 , w^T= tensor([[ 2., -2., -2.,  1.]]) , True
    # w^T * x_4= 1.0 , w^T= tensor([[ 2., -2., -2.,  1.]]) , True
    # w^T * x_5= 1.0 , w^T= tensor([[ 2., -2., -2.,  1.]]) , True
    # w^T * x_6= 3.0 , w^T= tensor([[ 2., -2., -2.,  1.]]) , True
    # w^T * x_7= 1.0 , w^T= tensor([[ 2., -2., -2.,  1.]]) , True
    # w^T * x_8= 1.0 , w^T= tensor([[ 2., -2., -2.,  1.]]) , True
    # Done!
    # w^T= tensor([[ 2., -2., -2.,  1.]])


if __name__ == '__main__':
    main()
