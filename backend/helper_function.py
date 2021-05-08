from time import time, ctime
import torch
from tqdm import tqdm
from scipy.spatial.distance import cdist
import numpy as np

cuda0 = torch.device('cuda:0')

def output_time():
    t = time()
    return str(ctime(t))


def compute_euclidean(x_tensor, y_tensor):
    """
    :param x_tensor: N*2 dim
    :param y_tensor: N*2 dim
    :return: N*N dim
    """
    # scipy
    dist_need = cdist(x_tensor.cpu(), y_tensor.cpu())
    print(dist_need)
    for dist in dist_need:
        np.sort(dist)
    # 太慢了sos
    # dist_square = []
    # for i in tqdm(range(len(x_tensor))):
    #     for j in range(len(x_tensor)):
    #         tmp = (x_tensor[i][0] - y_tensor[j][0]).pow(2) + (x_tensor[i][1] - y_tensor[j][1]).pow(2)
    #         # print(tmp)
    #         dist_square.append(tmp.cpu())
    # dist_square = torch.tensor(dist_square)
    # # print(dist_square)
    # output = torch.sqrt(dist_square)
    # print(output)
    #
    # return output.cpu()


def test_tqdm():
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from tqdm.gui import tqdm as tqdm_gui
    arr = np.random.randn(10000, 1000)
    ans = 0
    for p in tqdm(arr):
        ans = ans + p
    print(ans)
    # df = pd.DataFrame(np.random.randint(0, 100, (100000, 6)))
    # tqdm.pandas(ncols=50)  # can use tqdm_gui, optional kwargs, etc
    # # Now you can use `progress_apply` instead of `apply`
    # df.groupby(0).progress_apply(lambda x: x ** 2)


def test():
    List = [1,2,3,4]
    print(List[-1])

if __name__ == '__main__':
#     test_tqdm()
    test()
