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
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from sklearn.neighbors import NearestNeighbors
# np.set_printoptions(threshold=sys.maxsize)

def readfile(filename='pandas_to_csv_X_500lines.csv'):
    # filename = "pandas_x_for_clustering.csv"
    folder = "../sheet/cluster/"
    filename = folder+filename

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
    damping = 0.7
    print("damping=" + str(damping))
    preference = -200  # smaller, clusters less
    print("preference=" + str(preference))
    max_iter = 1000
    print("max_iter=" + str(max_iter))
    af = AffinityPropagation(random_state=28,
                             verbose=True,
                             max_iter=max_iter,
                             damping=damping,
                             preference=preference).fit(X)
    cluster_indices = af.cluster_centers_indices_
    print(cluster_indices)
    print(len(cluster_indices))

    cluster_labels = af.fit_predict(X)
    # print(cluster_labels)
    # print(len(cluster_labels))
    print("sil_score:"+str(metrics.silhouette_score(X=X, labels=cluster_labels)))
    # 0.72 0.53579  68
    # 0.7  0.52 66
    # 0.65 0.536  66
    # 0.6  0.489 62


# ok
def dbscan():
    X = readfile()
    # higher eps, higher min_samples, that is less clusters
    print("DBSCAN Parameters: ")
    # eps保持不变，increase min_samples，
    # that will decrease the sizes of individual clusters and
    #           increase the number of clusters
    # eps = input("eps= (usually between 0 and 1, float)\n")
    # eps = 0.5
    eps = compute_dist()
    print(eps)
    # min_samples = input("min_samples= (better for (size of dataset)/(50 to 70))\n")
    min_samples = 4  # double dataset dimensionality

    print("eps=" + str(eps))
    print("min_samples=" + str(min_samples))
    # 参数还要调整，现在0.4和20得到的簇还是100左右

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

    core_samples_mask[db.core_sample_indices_] = True
    # 不在core_sample的都是噪声点/离群点，有的噪声点也被分配了簇
    core_sample_indices = db.core_sample_indices_
    # print(core_sample_indices)
    # print(len(core_sample_indices))

    labels = db.fit_predict(X)
    labels_nonoise = []
    for idx in labels:
        if idx == -1:
            pass
        else:
            labels_nonoise.append(idx)
    # print(labels_nonoise)
    # print(np.unique(labels_nonoise))

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("n_clusters(removed noise):"+str(n_clusters_))
    total_point = 0
    n_noise = 0
    for i in labels:
        if i == -1:
            n_noise = n_noise+1
        total_point = total_point+1
    print("total points: " + str(total_point))
    print("noise: "+str(n_noise))


    # 将labels中每簇第一个索引作为center
    uni_labels = np.unique(labels)
    # print(uni_labels)
    __cluster_centers = []
    for i in uni_labels:
        # print(i)
        pos = list(labels).index(i)
        __cluster_centers.append(pos)
    # print(__cluster_centers)
    cluster_centers = []
    for idx in __cluster_centers:
        pts = X[idx]
        cluster_centers.append(pts)
    # print(cluster_centers)
    # print(len(cluster_centers))
    # print(len(labels))
    # print(len(cluster_centers))
    draw_plot(data=X, labels=labels, centers=cluster_centers)
    sil_score = metrics.silhouette_score(X=X, labels=labels)
    print("sil_score:" + str(sil_score))
    davies_bouldin_score = metrics.davies_bouldin_score(X=X, labels=labels)
    print("davies_bouldin score:"+str(davies_bouldin_score))


# ok
def optics():
    print("Start OPTICS:")
    X = readfile()
    # need adjust parameters still
    print("Parameters: ")

    min_samples = 40  # higher, less clusters
    # 一个点要想成为核心点，与其本身距离不大于epsilon的点的数目至少有min_samples个
    print("min_samples=" + str(min_samples))

    xi = .15  # higher, less clusters
    print("xi=" + str(xi))

    min_cluster_size = 50  # 一个簇至少包含的点数目
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

    n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print("n_clusters(removed noise):" + str(n_clusters_))

    # cluster_labels = opt.labels_[opt.ordering_]  # may be cluster_index
    # print("\ncluster_labels:")
    # print(cluster_labels)
    # print(len(cluster_labels))
    # print("\nnp.unique(cluster_labels):")
    # print(np.unique(cluster_labels))

    # 将labels中每簇第一个索引作为center
    __cluster_centers_idx = []
    for idx in np.unique(cluster_labels):
        pos = list(cluster_labels).index(idx)
        __cluster_centers_idx.append(pos)
    cluster_centers = []
    for idx in __cluster_centers_idx:
        pts = X[idx]
        cluster_centers.append(pts)
    # print("cluster_centers: \n" + str(cluster_centers))
    # print(len(cluster_centers))

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
    print("\nStart Clustering for CURE...")
    number_cluster = v
    number_represent_points = 10
    # number_cluster = input("number_cluster=")
    # number_represent_points = input("number_present_points=")

    print("Parameters: ")
    print("number_cluster=" + str(number_cluster))
    print("number_present_points=" + str(number_represent_points))

    compression = 0.3
    # Coefficient defines level of shrinking of representation points \
    #   toward the mean of the new created cluster after merging on each step.
    # Usually it destributed from 0 to 1
    print("compression=" + str(compression))

    cure_instance = cure(data=X,
                         number_cluster=number_cluster,
                         number_represent_points=number_represent_points,
                         compression=compression)

    cure_instance.process()
    clusters = cure_instance.get_clusters()
    print(clusters)

    # create cluster labels for all points
    cluster_labels = [None] * len(X)
    for idx in range(len(clusters)):
        for pts in clusters[idx]:
            cluster_labels[pts] = idx
    print(cluster_labels)
    # print(cluster_labels)
    # print(len(cluster_labels))
    # 以label中每个簇最早出现的位置定为center的索引
    num_cluster = len(np.unique(cluster_labels))
    cluster_centers = []
    # for idx in range(num_cluster):
    #     pos = list(cluster_labels).index(idx)
    #     cluster_centers.append(X[pos])

    representors = cure_instance.get_representors()  # [[[x,y]],[[x,y]]]
    for i in representors:
        cluster_centers.append(i[0])
    print(cluster_centers)

    print("number of clusters:" + str(len(np.unique(cluster_labels))))

    # X = readfile()
    # v = 32
    # number_represent_points = 4
    # compression = 0.3
    # cure_instance = cure(data=X,
    #                      number_cluster=v,
    #                      number_represent_points=number_represent_points,
    #                      compression=compression)
    # cure_instance.process()
    #
    # clusters = cure_instance.get_clusters()  # 每个簇，各簇包含的数据点索引号in X
    # print("clusters: ")
    # print(clusters)
    # print(len(clusters))
    #
    # # use cluster to get labels
    # cluster_labels = [None] * len(X)
    #
    # for idx in range(len(clusters)):
    #     for pts in clusters[idx]:
    #         cluster_labels[pts] = idx
    # print("cluster_labels: ")
    # print(cluster_labels)
    # print(np.unique(cluster_labels))
    # # 以label中每个簇第一次出现的位置定为center的索引
    # cluster_centers = []
    # for idx in np.unique(cluster_labels):
    #     pos = list(cluster_labels).index(idx)
    #     # print(pos)
    #     cluster_centers.append(X[pos])
    # print("cluster_centers: ")
    # print(cluster_centers)
    # print(len(cluster_centers))
    # # 或以clusters中每个簇第一个索引作为center的索引
    print(cluster_labels)

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


def test_input(argv):
    if len(argv) != 2:
        print("Error arguments input, should be python test.py cluster_mode")
        print("e.g. python test.py KMeans")
        exit(-1)
    cluster_mode = sys.argv[1]
    assert (cluster_mode == "KMeans" or
            cluster_mode == "KMedoids" or
            cluster_mode == "DBSCAN" or
            cluster_mode == "OPTICS" or
            cluster_mode == "MeanShift" or
            cluster_mode == "AP" or
            cluster_mode == "AffinityPropagation" or
            cluster_mode == "CURE")
    test(cluster_mode)


# def draw_raw():
#     X = readfile(filename='pandas_to_csv_X_40lines.csv')
#     x = []
#     y = []
#     for pts in X:
#         x.append(pts[0])
#         y.append(pts[1])
#     plt.scatter(x, y, alpha=0.5)
#     plt.show()


def draw_plot(data, labels, centers):
    colors = tuple([(np.random.random(), np.random.random(), np.random.random()) for i in range(len(np.unique(labels))-1)])
    colors = [rgb2hex(x) for x in colors]  # from  matplotlib.colors import  rgb2hex
    # print(colors)
    fig = plt.figure()
    ax = plt.subplot()

    for i, color in enumerate(colors):
        need_idx = np.where(labels == i)[0]
        # print(need_idx)
        ax.scatter(data[need_idx, 0], data[need_idx, 1], c=color, label=i, alpha=0.5)

    noise_data = []
    idx = 0
    for i in labels:
        if i == -1:
            noise_data.append(data[idx])
        idx = idx+1
    # print(noise_data)
    ax.scatter(noise_data[0], noise_data[1], c='black', alpha=0.5)
    # print(len(noise_data))
    plt.show()
    # plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    # plt.show()
    # j = 0
    # mark = np.random.rand(len(np.unique(labels)))
    # for i in labels:
    #     plt.plot([data[j:j+1, 0]], [data[j:j+1, 1]], mark[i])
    #     j = j+1


def test_data_format():
    data = readfile()
    x = [data[0:1, 0]]
    y = [data[0:1, 1]]
    print(x)
    print(y)


def compute_dist():
    data = readfile()
    nbrs = NearestNeighbors(n_neighbors=len(data)).fit(data)
    distances, indices = nbrs.kneighbors(data)
    need_dist = []
    pos = len(data)-1
    for dist in distances:
        need_dist.append(dist[pos])
    # print(need_dist)
    need_dist.sort(reverse=True)
    # print(need_dist)
    # print(distances)
    # print(indices)
    plt1 = plt.subplot(2, 2, 1)
    elbow = plt.subplot(2, 1, 2)

    elbow.plot(list(range(1, len(data) + 1)), need_dist, )  # elbow plot
    elbow.set_xlabel('list index')
    elbow.set_ylabel('point distance')
    elbow.set_title('elbow plot for epsilon')
    # elbow.show()
    eps = distances.mean()
    # plt.savefig('../imgs/test.png')
    xtick = [(len(data) - 1)//10 * i for i in range(11)]
    elbow.set_xticks(xtick)
    elbow.tick_params(labelsize=10)
    plt.tight_layout()
    plt.show()
    # print(eps)
    # print(need_dist[100])
    need_dist = np.array(need_dist)

    eps_index = np.where(need_dist < 400)
    eps = (eps_index[0]) / len(data)
    # print(np.where(need_dist < 401))
    # print(eps)
    return eps[0]


if __name__ == '__main__':
    # test_input(sys.argv)
    # kmeans() # ok
    # meanshift()  # ok
    # affinitypropagation()  # ok
    draw_raw()
    # print("")
    # compute_dist()

    # dbscan()  # ok
    # optics()  # ok
    # test_data_format()
    # clustering_kmedoids()  # ok
    # clustering_cure()  # ok
    # test_range()


