# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

import torch
import time
import numpy as np
import random
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
    print("S shape: " + str(S.shape))
    t1 = time.time()  # 记录时间
    print(tat.device, t1 - t0, S.norm(2))  # c.norm(2)表示矩阵c的二范数

    # device = torch.device('cuda')  # 用GPU来运行
    # tat = tat.to(device)
    # tb = tb.to(device)
    #
    # t1 = time.time()  # 记录时间
    # print(tat.device, t1 - t0, S.norm(2))  # c.norm(2)表示矩阵c的二范数

def test_three():
    # using pytorch
    device = torch.device('cuda')
    v = 32
    S = np.array([0 for k in range(v * v)])  # v行v列的矩阵
    S_tensor = torch.FloatTensor(S)
    S_tensor = S_tensor.to(device)
    oa = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ob = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    oa_tensor = torch.FloatTensor(oa)
    oa_tensor = torch.unsqueeze(oa_tensor, 0)
    oa_tensor_trans = oa_tensor.t()
    oa_tensor_trans = oa_tensor_trans.to(device)  # 传送给gpu

    ob_tensor = torch.FloatTensor(ob)
    ob_tensor = torch.unsqueeze(ob_tensor, 0)
    ob_tensor = ob_tensor.to(device)

    for i in range(2):
        e = torch.mm(oa_tensor_trans, ob_tensor)
        print(type(e))
        e = e.to(device)
        # e = np.mat(oa).T * np.mat(ob)
        print("value of e:")
        print(e)
        da = random.uniform(0,1)
        db = random.uniform(0,1)
        l = random.uniform(2,3)
        print("\nda="+str(da)+" , db="+str(db))
        w = abs(da-db)/l
        print("\nw="+str(w))
        S_tensor = S_tensor + torch.flatten(e*w)

        # S = S + np.array(e * w).flatten()  # flatten折叠成一维数组
        print("\nvalue of S:")
        print(S)
        print("\ntype of S: " + str(type(S)))
        print("\nsize of S: " + str(S.size))


if __name__ == '__main__':
    torch.cuda.set_device(1)
    # test_one()
    # test_two()
    test_three()

