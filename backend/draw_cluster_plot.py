import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

import sys

np.set_printoptions(threshold=sys.maxsize)

def read_ond_dimension_file(filename):
    X = pd.read_csv(filename).to_numpy()
    print(X)
    return X


def read_dict_pandas(filename):
    X = pd.read_csv


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


def draw_3d_plot():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Data for a three-dimensional line
    zline = np.linspace(0, 15, 1000)
    xline = np.sin(zline)
    yline = np.cos(zline)
    ax.plot3D(xline, yline, zline, 'gray')
    print(zline)
    print(yline)
    print(xline)
    print(len(zline))
    print(type(zline))
    plt.show()


    # Data for three-dimensional scattered points
    # zdata = 15 * np.random.random(100)
    # xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
    # ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
    # ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');


def draw_3d(self, labels, prototype_index, streamline_index, streamline_vertex):
    remainder_labels = []
    for idx in prototype_index:
        l = labels[idx]
        remainder_labels.append(l)
    remainder_labels = list(set(remainder_labels))
    print("remainder line labels")
    print(remainder_labels)  # [19, 30, 22, 14]
    colors_tsne = tuple(
        [(np.random.random(), np.random.random(), np.random.random()) for i in
         range(len(remainder_labels))])
    colors_tsne = [rgb2hex(x) for x in colors_tsne]  # from  matplotlib.colors import  rgb2hex

    # pts_data = []
    # label_for_pts = []
    print("selected streamlines indexes")
    print(prototype_index)  # [ 49 186 187 188 189 190 191 192 193 194 195 196 197 198 199]
    # for line_idx in prototype_index:
    #     for pts_idx in self.streamlines_lines_index_data[line_idx]:  # pts_idx也是索引
    #         pts_data.append(self.streamlines_vertexs_data[pts_idx])
    #         label_for_pts.append(labels[line_idx])
    # 得到所有流线的所有点所在标签
    fig = plt.figure()
    # pts_data = np.array(pts_data)
    # ax = plt.axes(projection='3d')

    ax1 = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')


    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    # ax1.set_title('Line Plot')

    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.set_zlabel('z')
    # ax2.set_title('Scatter Plot')
    #
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')

    # OUR ONE LINER ADDED HERE:
    ax1.get_proj = lambda: np.dot(Axes3D.get_proj(ax1), np.diag([1, 0.3, 1, 1]))

    # 给每条流线分配颜色
    color_for_line = dict()  # line_idx : line_label
    # 已分配好颜色的label，存放的是label
    detached_label_color = dict()  # label : color
    color_pos = 0
    for line_idx in prototype_index:
        line_label = labels[line_idx]  # 该条流线的标签
        # 每个label一个颜色
        # print(line_label)
        if not (line_label in detached_label_color):
            # print(color_pos)
            # print(detached_label_color.items())
            detached_label_color[line_label] = colors_tsne[color_pos]
            color_pos = color_pos+1
        color_for_line[line_idx] = detached_label_color[line_label]

    for line_idx in prototype_index:
        points_xdata = []
        points_ydata = []
        points_zdata = []
        for pts_idx in streamline_index[line_idx]:
            one_point = streamline_vertex[pts_idx]
            points_xdata.append(one_point[0])
            points_ydata.append(one_point[1])
            points_zdata.append(one_point[2])
        # print(points_xdata)
        # print(len(points_xdata))
        ax1.plot(points_xdata, points_ydata, points_zdata, c=color_for_line[line_idx])


    # for i, color in zip(remainder_labels, colors_tsne):
    #     # print(i)  # 19
    #     need_idx = np.where(np.array(label_for_pts) == i)  # need_idx为什么是[None]
    #     # print(need_idx)
    #
    #     for idx in need_idx:
    #         # print(idx)
    #         # ax.plot3D(pts_data[idx, 0], pts_data[idx, 1], pts_data[idx, 2], c=color, alpha=0.5)
    #         ax2.scatter3D(pts_data[idx, 0], pts_data[idx, 1], pts_data[idx, 2], c=color, alpha=1)
    #         # print(pts_data[idx, 0], pts_data[idx, 1], pts_data[idx, 2])

    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # draw_partial_based_KMeans()
    # draw_partial_based_DBSCAN()
    streamline_index = read_ond_dimension_file(filename='../sheet/streamlines_lines_index_to_draw200lines.csv')
    streamline_vertex = read_ond_dimension_file(filename='../sheet/streamlines_vertexs_data_to_draw200lines.csv')
    labels = read_label_file(filename='../sheet/labels_to_draw_CURE200lines.csv')
    proto_line_index = read_ond_dimension_file(filename='../sheet/prototype_index_to_draw_CURE200lines.csv')

    draw_3d(labels=labels, prototype_index=proto_line_index, streamline_index=streamline_index, streamline_vertex=streamline_vertex)


