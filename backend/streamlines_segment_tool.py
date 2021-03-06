"""
A toolkit for filtering streamlines.
Writen by changxiao at 2019/9/9
"""
import vtk
import numpy as np
from backend.dissimilarity import compute_dissimilarity
from backend.streamlines_segment_token import SegmentTokenizer
import pandas as pd
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(threshold=sys.maxsize)


class StreamlinesSegment(object):
    """
    There are n key steps in this class:
    1. Load streamlines file.
    2. Filter streamlines data.
    3. Save streamlines data.
    """

    def __init__(self):
        """
        Initialize member variables.

        """
        self.streamlines_vertexs_data = None        # 顶点数据 np.array()
        self.streamlines_lines_index_data = None    # 流线数据
        self.line_map_real2formal = dict()          # 实际流线数据到原始流线数据的映射
        self.streamlines_attributes_data = dict()   # 属性数据
        self.mul_dissimilarity_matrix = []          # 多组·相异性矩阵
        self.mul_prototype_index = []               # 多组·筛选出的流线索引
        self.polydata_input = None                  # 原始流线数据POLYDATA
        self.mul_polydata_output = []               # 多组·筛选出的流线数据(vtkPolyData)
        self.mul_lines_count_output = []            # 多组·筛选出的流线数量

        """
        以下是补充的用于聚类分析的
        三维的点用t-sne降维成二维点集，label不变
        """
        self.mul_streamlines_labels = []  # 多组·每条流线的分组标签
        self.cluster_mode = None

    def __clear_all_data(self):
        """
        Clear data for all class member variables.
        :return:
        """
        self.__init__()

    def filter_streamlines(self, src_streamlines_filename,
                                 dest_streamlines_count_list,
                                 dest_streamlines_filename_prefix,
                                 cluster_mode):
        """
        Filtering streamlines.
        调用此函数后可以得到经压缩的流线
        之后再对此流线组进行聚类分析
        Parameters
        ----------
        src_streamlines_filename : str of objects
            the file path of streamlines vtk file.
        dest_streamlines_count_list : int of objects
            a list of the number of streamlines to generate.
        dest_streamlines_filename_prefix : str of objects
            the basic file path of target streamlines files.
        
        Return
        ------
        status : bool
            the status of filtering streamlines.
        """
        # Load streamline file.
        self.cluster_mode = cluster_mode
        print("Load streamline file...")
        self.__load_streamlines_data_from_vtk_file(src_streamlines_filename)  # 模板代码，读取vtk
        self.__extract_streamlines_info_from_polydata()  # 提取vtk的所有数据，顶点、流线等等
        # Filter streamline data.
        print("Filtering streamlines based on segment...")
        self._filtering_streamlines_based_on_segment(dest_streamlines_count_list, cluster_mode)
        # Save streamline data.
        print("Save streamline data...")
        self.__generate_result_polydata_to_memory()
        self.__save_result_polydata_to_vtk_file(dest_streamlines_filename_prefix)
        # Clear all data.
        print("Clear all data...")
        self.__clear_all_data()
        
        return True

    def _filtering_streamlines_based_on_segment(self, dest_streamlines_count_list, cluster_mode):
        """
        Filtering streamlines based on segment.

        :param dest_streamlines_count_list:
        :return:
        """
        t = SegmentTokenizer(self.streamlines_vertexs_data,
                                self.streamlines_lines_index_data,
                                self.streamlines_attributes_data, 
                                cluster_mode=cluster_mode)
        t.prepare_for_comparison() 
        for cnt in dest_streamlines_count_list:
            # 基于distant_typeid=0（欧几里得距离）计算不相似度. d_E 0
            # dissimilarity_matrix, prototype_index = t.calculate_main_streamline_index(cnt, distant_typeid=0) 
            # 基于distant_typeid = 1（余弦距离）计算不相似度.

            # 基于distant_typeid=2（曼哈顿距离）计算不相似度.
            # 还需要测不同的distant_typeid的计算结果
            dissimilarity_matrix, prototype_index = t.calculate_main_streamline_index(cnt, distant_typeid=0)
            # 得到不相似度矩阵，压缩流场
            if cluster_mode == 'All':
                t_cnt = 1
                for labels in t.all_cluster_modes_streamlines_labels:
                    self.draw_3d(labels=labels, prototype_index=prototype_index, cluster_order=t_cnt)
                    t_cnt = t_cnt + 1
            else:
                labels = t.all_streamlines_cluster_labels  # Series
                self.draw_3d(labels=labels, prototype_index=prototype_index)

            # series_pd1 = pd.Series(labels)
            # labels_to_draw_filename = 'labels_to_draw_'+cluster_mode+str(len(self.streamlines_lines_index_data))+'lines.csv'
            # series_pd1.to_csv(labels_to_draw_filename, header=False, index=False)
            #
            # series_pd2 = pd.Series(prototype_index)
            # prototype_index_to_draw_filename = 'prototype_index_to_draw_'+cluster_mode+str(len(self.streamlines_lines_index_data))+'lines.csv'
            # series_pd2.to_csv(prototype_index_to_draw_filename, header=False, index=False)
            # # 只可视化剩下的流线
            #
            # # df_pd3 = pd.DataFrame(self.streamlines_lines_index_data)
            # lines_index_filename = 'streamlines_lines_index_to_draw'+cluster_mode+str(len(self.streamlines_lines_index_data))+'lines.txt'
            # # df_pd3.to_csv(lines_index_filename, header=False, index=False)
            # np.savetxt(fname=lines_index_filename, X=self.streamlines_lines_index_data, fmt='%s', delimiter=',')
            #
            # df_pd4 = pd.DataFrame(self.streamlines_vertexs_data)
            # streamlines_vertexs_data_filename = 'streamlines_vertexs_data_to_draw'+cluster_mode+str(len(self.streamlines_lines_index_data))+'lines.csv'
            # df_pd4.to_csv(streamlines_vertexs_data_filename, header=False, index=False)

            # 基于distant_typeid = 3（切比雪夫距离）计算不相似度.
            # 基于distant_typeid = 4（夹角余弦距离）计算不相似度.
            # 基于distant_typeid = 5（汉明距离）计算不相似度.
            # 基于distant_typeid = 6（杰卡德相似系数距离）计算不相似度.
            #
            # 以下待补充度量方法

            self.mul_dissimilarity_matrix.append(dissimilarity_matrix)
            self.mul_prototype_index.append(prototype_index)
            self.mul_lines_count_output.append(cnt)

        # print(type(self.mul_dissimilarity_matrix))  # list
        # print(self.mul_dissimilarity_matrix[0])  # a matrix
        # print("len: " + str(len(self.mul_dissimilarity_matrix)))  # len: 5
        #
        # print(type(self.mul_prototype_index))  # list
        # print(self.mul_prototype_index[0])  # selected streamlines indices  [0], len: 5
        #
        # print(type(self.mul_lines_count_output))  # list
        # print(self.mul_lines_count_output[0])  # 1

        # np.savetxt() can only save 1 or 2-dim data, higher dimensions need to be saved in np.save()
        # """
        # Traceback (most recent call last):
        #   File "test.py", line 20, in <module>
        #     cluster_mode)
        #   File "/home/gp2021-zsr/gp/backend/streamlines_segment_tool.py", line 80, in filter_streamlines
        #     self._filtering_streamlines_based_on_segment(dest_streamlines_count_list, cluster_mode)
        #   File "/home/gp2021-zsr/gp/backend/streamlines_segment_tool.py", line 149, in _filtering_streamlines_based_on_segment
        #     np.save(file='mul_dissimilarity_matrix0.npy')
        #   File "<__array_function__ internals>", line 4, in save
        # TypeError: _save_dispatcher() missing 1 required positional argument: 'arr'
        # """
        # np.savetxt("mul_dissimilarity_matrix0.txt", self.mul_dissimilarity_matrix[0],fmt='%f', delimiter=',')
        # df = pd.DataFrame(data=self.mul_dissimilarity_matrix)
        # df.to_csv('dissimilarity_matrix.csv', header=False, index=False)
        #
        # df2 = pd.DataFrame(data=self.mul_prototype_index)  # selected lines index
        # df2.to_csv('prototype_index.csv', header=False, index=False)

        # error: could not broadcast input array from shape (2080,1) into shape (2080)
        # np.savetxt("mul_dissimilarity_matrix.txt", self.mul_dissimilarity_matrix, fmt='%f', delimiter=',')

    def draw_3d(self, labels, prototype_index, cluster_order=0):
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
        fig = plt.figure(dpi=80)
        # pts_data = np.array(pts_data)
        ax1 = plt.axes(projection='3d')

        # ax1 = fig.add_subplot(111, projection='3d')
        # ax2 = fig.add_subplot(122, projection='3d')

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
            for pts_idx in self.streamlines_lines_index_data[line_idx]:
                one_point = self.streamlines_vertexs_data[pts_idx]
                points_xdata.append(one_point[0])
                points_ydata.append(one_point[1])
                points_zdata.append(one_point[2])
            # print(points_xdata)
            # print(len(points_xdata))
            ax1.plot(points_xdata, points_ydata, points_zdata, c=color_for_line[line_idx])

            # ax1.plot(np.array(points_xdata), np.array(points_ydata), np.array(points_ydata), c=color_for_line[line_idx], alpha=1)


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
        filename = str(len(self.streamlines_lines_index_data)) + 'lines_' + 'mode' + str(cluster_order)+'_'
        plt.savefig(filename+'dpi150.png', dpi=150)
        # plt.savefig(filename+'dpi100.png', dpi=100)

        # plt.show()

    def __load_streamlines_data_from_vtk_file(self, vtk_format_filename):
        """
        Load the streamlines data.

        :param vtk_format_filename:
        :return:
        """
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(vtk_format_filename)
        reader.Update()
        self.polydata_input = reader.GetOutput()

    def __extract_streamlines_info_from_polydata(self):
        """
        Extract streamline data from the polydata to array.

        :return:
        """
        # extract points data.
        points_count = self.polydata_input.GetNumberOfPoints()
        points_list = []
        for i in range(points_count):
            point_tuple = self.polydata_input.GetPoint(i)
            points_list.append(list(point_tuple))
        self.streamlines_vertexs_data = np.array(points_list)

        # extract cells data.
        lines_count = self.polydata_input.GetNumberOfLines()
        lines = self.polydata_input.GetLines()
        lines_index_list = []
        lines.InitTraversal()
        for i in range(lines_count):  # 遍历每一条流线
            points_index_list = []
            pts = vtk.vtkIdList()  # cell id
            lines.GetNextCell(pts)  # 更新pts数组，即一条流线的点id
            points_count = pts.GetNumberOfIds()  # 一条流线包含的点的数目
            for j in range(points_count):
                index = pts.GetId(j)
                points_index_list.append(index)
            # 筛选出的每条流线至少有3个不同的点.
            points_index_list = [ points_index_list[i] for i in range(len(points_index_list))\
                                if i==0 or not (self.streamlines_vertexs_data[points_index_list[i]]\
                                ==self.streamlines_vertexs_data[points_index_list[i-1]]).all() ]  # 去除重复点
            if len(points_index_list) >= 3:
                lines_index_list.append(points_index_list)
                self.line_map_real2formal[len(lines_index_list)-1] = i  # 只保留pts数大于3的流线
        self.streamlines_lines_index_data = np.array(lines_index_list, dtype='object')

        # extract attributes data.
        attributes_data = self.polydata_input.GetPointData()
        scalars_data = attributes_data.GetScalars()
        scalars_list = []
        if scalars_data is not None:
            for i in range(scalars_data.GetNumberOfValues()):
                scalars_list.append(scalars_data.GetValue(i))
            self.streamlines_attributes_data["scalars"] = np.array(scalars_list)
        else:
            cnt = len(self.streamlines_vertexs_data)
            self.streamlines_attributes_data["scalars"] = np.array([0.0 for v in range(cnt)])
        
        vectors_data = attributes_data.GetVectors()
        vectors_list = []
        if vectors_data is not None:
            for i in range(vectors_data.GetNumberOfValues()):
                vectors_list.append(vectors_data.GetValue(i))
            self.streamlines_attributes_data["vectors"] = np.array(vectors_list)\
                                .reshape(vectors_data.GetNumberOfValues()//3,3)
        else:
            cnt = len(self.streamlines_vertexs_data)
            self.streamlines_attributes_data["vectors"] = np.array([[0.0,0.0,0.0] for v in range(cnt)])

    def __generate_result_polydata_to_memory(self):
        """
        Generate filter streamline polydata data to memory.

        :return:
        """
        for lines_index in self.mul_prototype_index:
            lines_index = np.array([self.line_map_real2formal[i] for i in lines_index])
            polydata = vtk.vtkPolyData()
            polydata.DeepCopy(self.polydata_input)
            polydata.BuildCells()
            lines_count = polydata.GetNumberOfLines()
            for index in range(lines_count):
                if index not in lines_index:
                    polydata.DeleteCell(index)
            polydata.RemoveDeletedCells()
            polydata.Modified()
            self.mul_polydata_output.append(polydata)

    def __save_result_polydata_to_vtk_file(self, vtk_format_filename_bs):
        """
        Save streamlines to vtk file, which dataset type is POLYDATA.

        :param vtk_format_filename_bs:
        :return:
        """
        for i in range(len(self.mul_polydata_output)):
            vtk_format_filename = vtk_format_filename_bs + str(self.cluster_mode) + "_" + str(self.mul_lines_count_output[i]) + ".vtk"
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(vtk_format_filename)
            writer.SetInputData(self.mul_polydata_output[i])
            writer.Write()