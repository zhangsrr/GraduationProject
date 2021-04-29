"""
聚类
根据流线的词向量表达做聚类
1. 计算流线之间的相异度
2.
"""
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans, AffinityPropagation, DBSCAN, OPTICS
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.cure import cure
import sys
from sklearn import metrics
import datetime
# np.set_printoptions(threshold=sys.maxsize)

def readfile():
    # filename = "pandas_x_for_clustering.csv"
    filename = "com_pandas_x_for_clustering.csv"

    X = pd.read_csv(filename, header=None)
    # print(type(X))
    # print(X.columns)
    p1 = X[0].to_numpy()
    # print(p1)
    p2 = X[1].to_numpy()
    # print(p2)

    # draw plot
    # plt.figure(1)
    # plt.scatter(p1, p2)
    # plt.draw()
    # plt.show()
    dataset = np.c_[p1, p2]
    # print(dataset)
    # print(type(dataset))

    return dataset


# ok
def kmeans():
    X = readfile()
    km = KMeans(n_clusters=32, random_state=28)
    print(km)
    km.fit(X)

    cluster_labels = km.predict(X)
    print("cluster_labels: ")
    print(cluster_labels)
    print(len(cluster_labels))

    cluster_centers = km.cluster_centers_
    print("\ncluster_centers: ")
    print(cluster_centers)
    print(len(cluster_centers))

    # unique_index = np.unique(cluster_indices)
    # print(unique_index)

    # addition function
    dict = __vectors2words(cluster_centers)
    print("\ndict: ")
    print(dict)
    print(len(dict))
    no = 0


# ok
def clustering_kmedoids():
    X = readfile()
    k = 16
    initial_medoids = kmeans_plusplus_initializer(X, k).initialize(return_index=True)
    kmedoids_instance = kmedoids(data=X, initial_index_medoids=initial_medoids, itermax=2000)
    kmedoids_instance.process()

    cluster_centers = kmedoids_instance.get_medoids()
    print("cluster_centers: ")
    print(cluster_centers)
    print(len(cluster_centers))

    cluster_centers_pts = []
    for idx in cluster_centers:
        cluster_centers_pts.append(X[idx])
    print(cluster_centers_pts)
    print(len(cluster_centers_pts))

    cluster_clusters = kmedoids_instance.get_clusters()
    print("cluster_clusters: ")
    print(cluster_clusters)
    print(len(cluster_clusters))  # 2

    cluster_labels = kmedoids_instance.predict(X)
    print("cluster_labels: ")
    print(cluster_labels)
    print(len(cluster_labels))
    print(np.unique(cluster_labels))


# not ok
def affinitypropagation():
    X = readfile()
    # selected parameters based on small dataset
    # smaller preference, higher damping, that is less clusters
    print("Start Clustering for AffinityPropagation...")
    print("Parameters: ")
    damping = 0.85
    print("damping=" + str(damping))
    preference = -1000  # smaller, clusters less
    print("preference=" + str(preference))
    max_iter = 2000
    print("max_iter=" + str(max_iter))
    af = AffinityPropagation(random_state=0,
                             verbose=True,
                             max_iter=max_iter,
                             damping=damping,
                             preference=preference).fit(X)
    cluster_indices = af.cluster_centers_indices_
    print(cluster_indices)
    print(len(cluster_indices))

    cluster_labels = af.fit_predict(X)
    print(cluster_labels)
    print(len(cluster_labels))


# ok
def dbscan():
    X = readfile()
    # higher eps, higher min_samples, that is less clusters
    eps = 0.2
    min_samples = 4
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

    # print(core_samples_mask)  # 空白核心样本点 All False
    # print(len(core_samples_mask))  # 2668  90182
    # print(type(core_samples_mask))

    core_samples_mask[db.core_sample_indices_] = True
    # 不在core_sample的都是噪声点，有的噪声点也被分配了簇
    core_sample_indices = db.core_sample_indices_
    print(core_sample_indices)
    print(len(core_sample_indices))

    # print(db.core_sample_indices_)  # [    0     2     3 ... 90178 90179 90180]，
    # print(len(db.core_sample_indices_))  # 1731  45913
    # print(type(db.core_sample_indices_))
    all_pts = db.components_
    # print(len(all_pts))

    # print(db.components_)  # detailed core_samples_coordinate
    # print(len(db.components_))  # 1731

    labels = db.fit_predict(X)
    # print(labels)  # [-1  0  0 ... 64 64 -1]
    # print(len(labels))  # 2668 need to remove some points
    # print("np.unique(labels): ")
    # print(np.unique(labels))

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters_)  # 25

    # 根据indices和labels构造newlabels
    # new_labels = []
    # for idx in core_sample_indices:
    #     t = labels[idx]
    #     # print(t)
    #     new_labels.append(t)
    # print(new_labels)
    # print(len(new_labels))
    new_labels = db.fit_predict(X)

    # 将labels中每簇第一个索引作为center
    uni_labels = np.unique(new_labels)
    print(uni_labels)
    __cluster_centers = []
    for i in uni_labels:
        # print(i)
        pos = list(new_labels).index(i)
        __cluster_centers.append(pos)
    print(__cluster_centers)
    cluster_centers = []
    for idx in __cluster_centers:
        pts = X[idx]
        cluster_centers.append(pts)
    print(cluster_centers)
    print(len(cluster_centers))


# ok
def optics():
    print("Start OPTICS:")
    X = readfile()
    # need adjust parameters still
    print("Parameters: ")

    min_samples = 8  # higher, less clusters
    # 一个点要想成为核心点，与其本身距离不大于epsilon的点的数目至少有min_samples个
    print("min_samples=" + str(min_samples))

    xi = .15  # higher, less clusters
    print("xi=" + str(xi))

    min_cluster_size = 10  # 一个簇至少包含的点数目
    print("min_cluster_size=" + str(min_cluster_size))

    # eps = 0.15
    # print("eps=" + str(eps))

    # print("opt.ordering_:")
    # print(opt.ordering_)
    # print(len(opt.ordering_))
    # print("\nnp.unique(opt.ordering_):")
    # print(np.unique(opt.ordering_))  # one by one

    opt = OPTICS(min_samples=min_samples,
                 xi=xi,
                 min_cluster_size=min_cluster_size,
                 # eps=eps,
                 # max_eps=max_eps
                 ).fit(X)
    cluster_labels = opt.fit_predict(X)

    # cluster_labels = opt.labels_[opt.ordering_]  # may be cluster_index
    print("\ncluster_labels:")
    print(cluster_labels)
    print(len(cluster_labels))
    print("\nnp.unique(cluster_labels):")
    print(np.unique(cluster_labels))

    # 将labels中每簇第一个索引作为center
    __cluster_centers_idx = []
    for idx in np.unique(cluster_labels):
        pos = list(cluster_labels).index(idx)
        __cluster_centers_idx.append(pos)
    cluster_centers = []
    for idx in __cluster_centers_idx:
        pts = X[idx]
        cluster_centers.append(pts)
    print("cluster_centers: \n" + str(cluster_centers))
    print(len(cluster_centers))

    # print("\nopt.cluster_hierarchy_:")
    # print(opt.cluster_hierarchy_)
    # print(len(opt.cluster_hierarchy_))

    # clusters = X[opt.ordering_][opt.cluster_hierarchy_]
    # print("\ncluster:")
    # print(clusters)
    # print(len(clusters))
    # print("\nopt.core_distances_[opt.ordering_]:")
    # print(opt.core_distances_[opt.ordering_])
    # print(len(opt.core_distances_[opt.ordering_]))
    # print("\nopt.predecessor_:")
    # print(opt.predecessor_)
    # print(len(opt.predecessor_))
    # print("\nopt.reachability_[opt.ordering_]")
    # print(opt.reachability_[opt.ordering_])
    # print(len(opt.reachability_[opt.ordering_]))


# ok but time long
def meanshift():
    X = readfile()
    print("Start Clustering for MeanShift...")
    # time-complexity O(n^2), n is the number of points
    print("Parameters for estimating bandwidth: ")
    quantile = 0.4  # 0.3=54, 0.4=52, 0.5=49, 0.8=40
    print("quantile="+str(quantile))
    random_state = 40
    print("random_state="+str(random_state))
    bandwidth = estimate_bandwidth(X, quantile=quantile, random_state=random_state)
    print("Parameters for Meanshift: ")
    print("bandwidth=" + str(bandwidth))
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    cluster_labels = ms.predict(X)
    print(cluster_labels)
    print(len(cluster_labels))
    print(np.unique(cluster_labels))
    print(len(np.unique(cluster_labels)))

    cluster_centers = ms.cluster_centers_
    print(cluster_centers)
    print(len(cluster_centers))
    sil_score = metrics.silhouette_score(X, cluster_labels, metric='euclidean')
    print("Silhouette Coefficient: " + str(sil_score))
    # all for rand10, quantile 0.2=0.5982, 0.28=0.6211, 0.3=0.6902, 0.35=0.6846,  0.4=0.7009, 0.45=0.7045, 0.5=0.6742,
    # all for rand20, quantile 0.2=0.5982, 0.28=0.6211, 0.3=0.6902, 0.35=0.6846,  0.4=0.7009, 0.5=0.6742,
    # all for rand30, quantile 0.2=0.5982, 0.28=0.6211, 0.3=0.6902, 0.35=0.6846,  0.4=0.7009, 0.5=0.6742,


# ok
def clustering_cure():
    X = readfile()
    v = 32
    number_represent_points = 4
    compression = 0.3
    cure_instance = cure(data=X,
                         number_cluster=v,
                         number_represent_points=number_represent_points,
                         compression=compression)
    cure_instance.process()

    clusters = cure_instance.get_clusters()  # 每个簇，各簇包含的数据点索引号in X
    print("clusters: ")
    print(clusters)
    print(len(clusters))

    # use cluster to get labels
    cluster_labels = [None] * len(X)

    for idx in range(len(clusters)):
        for pts in clusters[idx]:
            cluster_labels[pts] = idx
    print("cluster_labels: ")
    print(cluster_labels)
    print(np.unique(cluster_labels))
    # 以label中每个簇第一次出现的位置定为center的索引
    cluster_centers = []
    for idx in np.unique(cluster_labels):
        pos = list(cluster_labels).index(idx)
        # print(pos)
        cluster_centers.append(X[pos])
    print("cluster_centers: ")
    print(cluster_centers)
    print(len(cluster_centers))
    # 或以clusters中每个簇第一个索引作为center的索引


    sil_score = metrics.silhouette_score(X, cluster_labels, metric='euclidean')
    print("Silhouette Coefficient: " + str(sil_score))
    # v = 32, number_represent_points = 4, compression = 0.8
    #   sil_co = 0.7715
    # compression=0.5, sil_co=0.7738
    # compression=0.4, sil_co=0.78168
    # compression=0.3, sil_co=0.7844
    # compression=0.2, sil_co=0.77979


def __vectors2words(vecs):
    dictionary = []
    for pt_center in vecs:
        word = ""
        for num in pt_center:
            word = word + str(num) + "_"
        print(word)
        dictionary.append(word)
    return dictionary


def test_range():
    # data = np.array([[[1, 2, 3, 4],
    #                  [5, 6, 7, 8],
    #                  [8, 9, 10, 11],
    #                  [10, 11, 12, 13]],
    #                 [[1, 2, 3, 4],
    #                  [5, 11, 31, 32],
    #                  [31, 32, 44, 56],
    #                  [23, 54, 67, 9]],
    #                 [[1, 2, 3, 4],
    #                  [5, 11, 31, 32],
    #                  [31, 32, 44, 56],
    #                  [23, 54, 67, 9]],
    #                  [[1, 2, 3, 4],
    #                   [5, 11, 31, 32],
    #                   [31, 32, 44, 56],
    #                   [23, 54, 67, 9]],
    #                  [[1, 2, 3, 4],
    #                   [5, 11, 31, 32],
    #                   [31, 32, 44, 56],
    #                   [23, 54, 67, 9]]])
    data = []
    data.append([[0.48812084, 0.63859315, 1.0215809, 0.],
                 [0.47763169, 0.79148176, 1.03008981, 0.],
                 [0.47133242, 0.44648659, 1.04010288, 0.],
                 [0.45181917, 0.18640114, 1.05114307, 0.]])
    data.append([[0.48287627, 0.71503746, 1.02583535, 0.],
                 [0.47448206, 0.61898418, 1.03509634, 0.],
                 [0.46157579, 0.31644387, 1.04562297, 0.]])
    data.append([[0.47867916, 0.66701082, 1.03046585, 0.],
                 [0.46802893, 0.46771402, 1.04035966, 0.]])
    data.append([[0.47335404, 0.56736242, 1.03541275, 0.]])
    for i in range(len(data))[::-1]:
        print(i)
        for row in data[i]:
            print("row of pyramid[i]: \n" + str(row))
            for v in row:  # len(row)始终是4
                pass
                # print("for v in row: \n" + str(v))


def test(arg):
    print(arg)

def matrix_gpu():
    X = readfile()

def py_matmul4(a, b):
    ra, ca = a.shape
    rb, cb = b.shape
    assert ca == rb, f"{ca} != {rb}"

    return np.matmul(a, b)

if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print("Error arguments input, should be python test.py cluster_mode")
    #     print("e.g. python test.py KMeans")
    #     exit(-1)
    # cluster_mode = sys.argv[1]
    # assert (cluster_mode == "KMeans" or
    #         cluster_mode == "KMedoids" or
    #         cluster_mode == "DBSCAN" or
    #         cluster_mode == "OPTICS" or
    #         cluster_mode == "MeanShift" or
    #         cluster_mode == "AP" or
    #         cluster_mode == "AffinityPropagation" or
    #         cluster_mode == "CURE")
    # test(cluster_mode)
    # starttime = datetime.datetime.now()
    # a = np.array([[1,2,3],
    #               [4,5,6],
    #               [7,8,9],
    #               [10,11,12]])
    # b = np.array([[1,2,3,4],
    #              [5,6,7,8],
    #              [9,10,11,12]])
    # ar, ac = a.shape
    # print(ar, ac)
    # br, bc = b.shape
    # print(br, bc)
    # assert (ac == br)
    # result = py_matmul4(a, b)
    # endtime = datetime.datetime.now()
    # print(endtime-starttime)

    # kmeans() # ok
    # meanshift()  # ok
    # affinitypropagation()  # ok
    dbscan()  # ok
    # optics()  # ok
    # clustering_kmedoids()  # ok
    # clustering_cure()  # ok
    # test_range()
