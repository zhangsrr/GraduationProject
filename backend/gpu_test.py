# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

import torch
import time
import numpy as np

def test_one():
    print(torch.__version__)  # 返回pytorch的版本
    print(torch.cuda.is_available())  # 当CUDA可用时返回True

    a = torch.randn(10000, 1000)  # 返回10000行1000列的张量矩阵
    b = torch.randn(1000, 2000)  # 返回1000行2000列的张量矩阵

    t0 = time.time()  # 记录时间
    c = torch.matmul(a, b)  # 矩阵乘法运算
    t1 = time.time()  # 记录时间
    print(a.device, t1 - t0, c.norm(2))  # c.norm(2)表示矩阵c的二范数

    device = torch.device('cuda')  # 用GPU来运行
    a = a.to(device)
    b = b.to(device)

    # 初次调用GPU，需要数据传送，因此比较慢
    t0 = time.time()
    c = torch.matmul(a, b)
    t2 = time.time()
    print(a.device, t2 - t0, c.norm(2))

    # 这才是GPU处理数据的真实运行时间，当数据量越大，GPU的优势越明显
    t0 = time.time()
    c = torch.matmul(a, b)
    t2 = time.time()
    print(a.device, t2 - t0, c.norm(2))


def test_two():
    a = np.random.rand(10000, 5000)
    b = np.random.rand(10000, 5000)
    ta = torch.from_numpy(a)
    tat = torch.transpose(ta, 0, 1)

    tb = torch.from_numpy(b)

    device = torch.device('cuda')  # 用GPU来运行

    t0 = time.time()  # 记录时间
    tat = tat.to(device)
    print(type(tat))
    tb = tb.to(device)
    S = torch.mm(tat, tb)
    t1 = time.time()  # 记录时间
    print(tat.device, t1 - t0, S.norm(2))  # c.norm(2)表示矩阵c的二范数

    # device = torch.device('cuda')  # 用GPU来运行
    # tat = tat.to(device)
    # tb = tb.to(device)
    #
    # t1 = time.time()  # 记录时间
    # print(tat.device, t1 - t0, S.norm(2))  # c.norm(2)表示矩阵c的二范数


if __name__ == '__main__':
    torch.cuda.set_device(1)
    test_one()
    test_two()

