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
        self.streamlines_vertexs_data = None        # 顶点数据
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
            dissimilarity_matrix, prototype_index = t.calculate_main_streamline_index(cnt, distant_typeid=2)
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
        self.streamlines_lines_index_data = np.array(lines_index_list)

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