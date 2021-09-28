import torch


def main():
    class_1 = torch.tensor([
        [-1, -1]
    ], dtype=torch.float)
    class_2 = torch.tensor([
        [0, 0]
    ], dtype=torch.float)
    class_3 = torch.tensor([
        [1, 1]
    ], dtype=torch.float)

    aug = torch.ones((class_1.shape[0], 1))

    class_1 = torch.cat([class_1, aug], dim=1)
    class_2 = torch.cat([class_2, aug], dim=1)
    class_3 = torch.cat([class_3, aug], dim=1)
    print(class_1)
    print(class_2)
    print(class_3)

    x = torch.cat([class_1, class_2, class_3], dim=0)
    print(x)

    w = torch.zeros((x.shape[0], x.shape[1]))
    c = 1

    iteration = 0
    while True:
        print(f'iter {iteration + 1}:')
        flag = torch.zeros((x.shape[0],))

        for k in range(x.shape[0]):
            res = torch.matmul(w, x[k].reshape(-1, 1))
            is_true = True
            print('k=', k + 1)

            for i in range(res.shape[0]):
                if i == k:
                    continue
                if res[k] <= res[i]:
                    w[i] -= c * x[k]
                    is_true = False

            if is_true:
                flag[k] = 1
            else:
                w[k] += c * x[k]

            print(is_true)
            for i in range(res.shape[0]):
                print(f'w_{i + 1}^T * x_{k + 1}=', res[i].item(), f', w_{i + 1}^T=', w[i])

        iteration += 1
        if flag.sum().item() >= flag.shape[0]:
            print('Done!')
            break
        print()

    print(w)

    # iter 1:
    # k= 1
    # False
    # w_1^T * x_1= 0.0 , w_1^T= tensor([-1., -1.,  1.])
    # w_2^T * x_1= 0.0 , w_2^T= tensor([ 1.,  1., -1.])
    # w_3^T * x_1= 0.0 , w_3^T= tensor([ 1.,  1., -1.])
    # k= 2
    # False
    # w_1^T * x_2= 1.0 , w_1^T= tensor([-1., -1.,  0.])
    # w_2^T * x_2= -1.0 , w_2^T= tensor([1., 1., 0.])
    # w_3^T * x_2= -1.0 , w_3^T= tensor([ 1.,  1., -2.])
    # k= 3
    # False
    # w_1^T * x_3= -2.0 , w_1^T= tensor([-1., -1.,  0.])
    # w_2^T * x_3= 2.0 , w_2^T= tensor([ 0.,  0., -1.])
    # w_3^T * x_3= 0.0 , w_3^T= tensor([ 2.,  2., -1.])
    #
    # iter 2:
    # k= 1
    # True
    # w_1^T * x_1= 2.0 , w_1^T= tensor([-1., -1.,  0.])
    # w_2^T * x_1= -1.0 , w_2^T= tensor([ 0.,  0., -1.])
    # w_3^T * x_1= -5.0 , w_3^T= tensor([ 2.,  2., -1.])
    # k= 2
    # False
    # w_1^T * x_2= 0.0 , w_1^T= tensor([-1., -1., -1.])
    # w_2^T * x_2= -1.0 , w_2^T= tensor([0., 0., 0.])
    # w_3^T * x_2= -1.0 , w_3^T= tensor([ 2.,  2., -2.])
    # k= 3
    # True
    # w_1^T * x_3= -3.0 , w_1^T= tensor([-1., -1., -1.])
    # w_2^T * x_3= 0.0 , w_2^T= tensor([0., 0., 0.])
    # w_3^T * x_3= 2.0 , w_3^T= tensor([ 2.,  2., -2.])
    #
    # iter 3:
    # k= 1
    # True
    # w_1^T * x_1= 1.0 , w_1^T= tensor([-1., -1., -1.])
    # w_2^T * x_1= 0.0 , w_2^T= tensor([0., 0., 0.])
    # w_3^T * x_1= -6.0 , w_3^T= tensor([ 2.,  2., -2.])
    # k= 2
    # True
    # w_1^T * x_2= -1.0 , w_1^T= tensor([-1., -1., -1.])
    # w_2^T * x_2= 0.0 , w_2^T= tensor([0., 0., 0.])
    # w_3^T * x_2= -2.0 , w_3^T= tensor([ 2.,  2., -2.])
    # k= 3
    # True
    # w_1^T * x_3= -3.0 , w_1^T= tensor([-1., -1., -1.])
    # w_2^T * x_3= 0.0 , w_2^T= tensor([0., 0., 0.])
    # w_3^T * x_3= 2.0 , w_3^T= tensor([ 2.,  2., -2.])
    # Done!
    # tensor([[-1., -1., -1.],
    #         [ 0.,  0.,  0.],
    #         [ 2.,  2., -2.]])


if __name__ == '__main__':
    main()
