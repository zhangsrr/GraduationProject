import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import pandas as pd
import numpy as np

import sys

np.set_printoptions(threshold=sys.maxsize)

def read_pts_file(filename='../sheet/cluster/pandas_to_csv_X_500lines.csv'):
    # filename = "pandas_x_for_clustering.csv"
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


def read_label_file(filename):
    labels_sheet = pd.read_csv(filename, header=None)
    labels = labels_sheet[0].to_numpy()
    return labels


def draw_partial_based_KMeans():
    # 500lines data
    pts_filename_500 = '../sheet/cluster/pandas_to_csv_X_500lines.csv'
    pts_data_500 = read_pts_file(filename=pts_filename_500)
    labels_filename_500_tsne = '../pandas_to_csv_cluster_labels_KMeans500_tsne.csv'
    labels_data_500_tsne = read_label_file(filename=labels_filename_500_tsne)

    labels_filename_500 = '../pandas_to_csv_cluster_labels_KMeans500.csv'
    labels_data_500 = read_label_file(filename=labels_filename_500)

    # 2080lines data
    pts_filename_2080 = '../sheet/cluster/pandas_to_csv_X_2080lines.csv'
    pts_data_2080 = read_pts_file(filename=pts_filename_2080)
    labels_filename_2080 = '../sheet/cluster/pandas_to_csv_cluster_labels_KMeans2080.csv'
    labels_data_2080 = read_label_file(filename=labels_filename_2080)

    # 40lines data
    pts_filename_40 = '../pandas_to_csv_X_40lines.csv'
    pts_data_40 = read_pts_file(filename=pts_filename_40)
    labels_filename_40 = '../pandas_to_csv_cluster_labels_KMeans40.csv'
    labels_data_40 = read_label_file(filename=labels_filename_40)

    colors_500_tsne = tuple(
        [(np.random.random(), np.random.random(), np.random.random()) for i in range(len(np.unique(labels_data_500)) - 1)])
    colors_500_tsne = [rgb2hex(x) for x in colors_500_tsne]  # from  matplotlib.colors import  rgb2hex

    colors_500 = tuple(
        [(np.random.random(), np.random.random(), np.random.random()) for i in range(len(np.unique(labels_data_500)) - 1)])
    colors_500 = [rgb2hex(x) for x in colors_500]  # from  matplotlib.colors import  rgb2hex

    colors_2080 = tuple(
        [(np.random.random(), np.random.random(), np.random.random()) for i in
         range(len(np.unique(labels_data_2080)) - 1)])
    colors_2080 = [rgb2hex(x) for x in colors_2080]  # from  matplotlib.colors import  rgb2hex

    colors_40 = tuple(
        [(np.random.random(), np.random.random(), np.random.random()) for i in range(len(np.unique(labels_data_40)) - 1)])
    colors_40 = [rgb2hex(x) for x in colors_40]  # from  matplotlib.colors import  rgb2hex

    # fig, ((ax1, ax2), ax3) = plt.subplots(2, 2)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    ax1.set_title('KMeans TSNE For 500 Streamlines', fontsize=9)
    ax2.set_title('KMeans For 500 Streamlines', fontsize=9)
    ax3.set_title('KMeans For 2080 Streamlines', fontsize=9)
    ax4.set_title('KMeans For 40 Streamlines', fontsize=9)

    for i, color in enumerate(colors_500_tsne):
        need_idx = np.where(labels_data_500_tsne == i)[0]
        ax1.scatter(pts_data_500[need_idx, 0], pts_data_500[need_idx, 1], c=color, label=i, alpha=0.4)

    for i, color in enumerate(colors_500):
        need_idx = np.where(labels_data_500 == i)[0]
        ax2.scatter(pts_data_500[need_idx, 0], pts_data_500[need_idx, 1], c=color, label=i, alpha=0.4)

    for i, color in enumerate(colors_2080):
        need_idx = np.where(labels_data_2080 == i)[0]
        # print(need_idx)
        ax3.scatter(pts_data_2080[need_idx, 0], pts_data_2080[need_idx, 1], c=color, label=i, alpha=0.4)


    for i, color in enumerate(colors_40):
        need_idx = np.where(labels_data_40 == i)[0]
        # print(need_idx)
        ax4.scatter(pts_data_40[need_idx, 0], pts_data_40[need_idx, 1], c=color, label=i, alpha=0.4)
    # fig.tight_layout()
    plt.show()


def draw_partial_based_DBSCAN():
    # 500lines data
    pts_filename_500 = '../sheet/cluster/pandas_to_csv_X_500lines.csv'
    pts_data_500 = read_pts_file(filename=pts_filename_500)
    labels_filename_500 = '../sheet/cluster/pandas_to_csv_cluster_labels_DBSCAN500.csv'
    labels_data_500 = read_label_file(filename=labels_filename_500)

    # 2080lines data
    pts_filename_2080 = '../sheet/cluster/pandas_to_csv_X_2080lines.csv'
    pts_data_2080 = read_pts_file(filename=pts_filename_2080)
    labels_filename_2080 = '../sheet/cluster/pandas_to_csv_cluster_labels_DBSCAN2080.csv'
    labels_data_2080 = read_label_file(filename=labels_filename_2080)

    # 2109lines data
    # pts_filename_2109 = '../sheet/cluster/pandas_to_csv_X_2109lines.csv'
    # pts_data_2109 = read_pts_file(filename=pts_filename_2109)
    # labels_filename_2109 = '../sheet/cluster/pandas_to_csv_cluster_labels_KMeans2109.csv'
    # labels_data_2109 = read_label_file(filename=labels_filename_2109)

    colors_500 = tuple(
        [(np.random.random(), np.random.random(), np.random.random()) for i in range(len(np.unique(labels_data_500)) - 1)])
    colors_500 = [rgb2hex(x) for x in colors_500]  # from  matplotlib.colors import  rgb2hex

    colors_2080 = tuple(
        [(np.random.random(), np.random.random(), np.random.random()) for i in
         range(len(np.unique(labels_data_2080)) - 1)])
    colors_2080 = [rgb2hex(x) for x in colors_2080]  # from  matplotlib.colors import  rgb2hex

    # colors_2109 = tuple(
    #     [(np.random.random(), np.random.random(), np.random.random()) for i in range(len(np.unique(labels_data_2109)) - 1)])
    # colors_2109 = [rgb2hex(x) for x in colors_2109]  # from  matplotlib.colors import  rgb2hex

    # fig, ((ax1, ax2), ax3) = plt.subplots(2, 2)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    # ax3 = plt.subplot(212)

    ax1.set_title('DBSCAN For 500 Streamlines', fontsize=9)
    ax2.set_title('DBSCAN For 2080 Streamlines', fontsize=9)

    for i, color in enumerate(colors_500):
        need_idx = np.where(labels_data_500 == i)[0]
        if max(need_idx) > 1594:
            print(need_idx)
        # print(need_idx)
        ax1.scatter(pts_data_500[need_idx, 0], pts_data_500[need_idx, 1], c=color, label=i, alpha=0.4)

    for i, color in enumerate(colors_2080):
        need_idx = np.where(labels_data_2080 == i)[0]
        # print(need_idx)
        ax2.scatter(pts_data_2080[need_idx, 0], pts_data_2080[need_idx, 1], c=color, label=i, alpha=0.4)


    # for i, color in enumerate(colors_2109):
    #     need_idx = np.where(labels_data_2109 == i)[0]
    #     # print(need_idx)
    #     ax3.scatter(pts_data_2109[need_idx, 0], pts_data_2109[need_idx, 1], c=color, label=i, alpha=0.4)

    # fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    draw_partial_based_KMeans()
    draw_partial_based_DBSCAN()

