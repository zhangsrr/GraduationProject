"""
step 1. 流线分段
此py可得流线每一段的特征向量&流线本身的特征向量
1.1 先求流线每一点的特征值
1.2 根据特征值对一条流线进行分段
1.3 计算分段的特征向量
1.4 根据分段的特征向量生成词汇 需要用到聚类
1.5 根据分段的词汇生成流线的词向量表达
"""
import numpy as np
from stream_space_curve import StreamSpaceCurve
from pca import pca, plotBestFit
from sklearn.cluster import KMeans

sel_feature = [
    "curvature",  # 曲率
    "torsion",  # 扭率
    "tortuosity",  # 弯曲度
    "velocity_direction_entropy"  # 速度方向熵
]

class SegmentTokenizer(object):
    def __init__(self,
                 streamlines_vertexs_data,
                 streamlines_lines_index_data,
                 streamlines_attributes_data,
                 cluster_mode = "KMeans"):
        # original data.
        self.streamlines_vertexs_data = streamlines_vertexs_data
        self.streamlines_lines_index_data = streamlines_lines_index_data
        self.streamlines_attributes_data = streamlines_attributes_data
        self.cluster_mode = cluster_mode
        # 流线各点的特征值.
        self.all_point_features = dict()   # 流线各点的特征值.
        self.all_line_lengths = dict()     # 各流线的长度.
        self.all_line_arc_lengths = dict() # 流线各步长分段的长度.
        # 流线分段的分段点索引.
        self.segment_points_index = dict()
        self.all_line_segment_arc_lengths = dict() # 各流线分段的长度.
        # 流线分段的特征向量.
        self.segment_feature_vectors = dict()
        # 流线分段的词汇.
        self.segment_feature_vectors_kdim = dict() # 降维后的流线分段特征向量.
        self.dictionary = [] # 词典.
        self.segment_vocabularys_index = dict() # 分段对应的单词索引.
        # 流线的词向量.
        self.unique_heat_vectors_dictionary = [] # 独热向量词典. one-hot 符合该特征，则值为1
        self.all_line_vocabulary_vectors = dict() # 流线的词向量.

    def prepare_for_comparison(self):
        """数据准备阶段：\n
        1.计算流线各点的特征值；\n
        2.根据特征值对流线进行分段；\n
        3.计算流线分段的特征向量；\n
        4.基于流线分段生成词汇；\n
        5.生成流线的词向量表达。\n
        """
        if self.cluster_mode is None:
            raise("The cluster mode is None.")
        # 计算流线各点的特征值.
        self.get_all_point_features_from_line()
        # 根据特征值对流线进行分段. sel_feature[0]曲率，分段只能依靠一个特征值进行分段？
        self.segment_all_lines(sel_feature[0], n_segment=50) #【注】此处的n_segment应该由外部主函数传入.
        # 计算流线分段的特征向量.
        self.calculate_all_segment_vectors(dim=12) #【注】此处的dim应该由外部主函数传入，暂时写为特定值.
        # 基于流线分段的特征向量生成词汇. 用到聚类算法
        self.generate_vocabulary_based_on_streamline_segmentation(k=2, v=32)
        # 生成流线的词向量表达.
        self.generate_streamlined_word_vector_expressions(v=32)

    def get_all_point_features_from_line(self):
        print("get_all_point_features_from_line...")
        for line_index in range(len(self.streamlines_lines_index_data)):
            self.get_point_features_from_line(line_index)

    def get_point_features_from_line(self, line_index: int):
        line_vertexs = self.streamlines_lines_index_data[line_index]
        line_scalars = np.array([self.streamlines_attributes_data["scalars"][i] for i in line_vertexs])
        line_vectors = np.array([self.streamlines_attributes_data["vectors"][i] for i in line_vertexs])
        points       = np.array([self.streamlines_vertexs_data[i] for i in line_vertexs])
        curve = StreamSpaceCurve(points, line_vectors)
        point_features = curve.get_point_features()  # 求特征的方法pyknotid在spacechurve中都提供了
        self.all_point_features[line_index] = point_features
        # 生成流线词向量时要用到每条流线的长度和流线各步长分段的长度.
        self.all_line_lengths[line_index] = curve.get_total_length()
        self.all_line_arc_lengths[line_index] = curve.get_arc_length_matrix_array()

    def segment_all_lines(self, sel_feature=sel_feature[0], n_segment=50):
        print("segment_all_lines...")
        print("streamlines_lines_index_data =", len(self.streamlines_lines_index_data))
        for line_index in range(len(self.streamlines_lines_index_data)):
            self.segment_one_line(line_index, sel_feature, n_segment)

    def segment_one_line(self, line_index: int, sel_feat, n_segment):
        '''
        line_index:流线的索引
        sel_feat:选择用来分段的特征属性
        n_segment:最大分段数
        # dim:分段向量的维数
        '''
        length = len(self.streamlines_lines_index_data)
        perc = [(length-1)//4*v for v in range(5)]
        if line_index in perc:
            s_perc = "|"
            s = "##########"
            s_perc += s * perc.index(line_index)
            print(s_perc, int(perc.index(line_index)/(len(perc)-1)*100), "%")
        # 0.根据流线索引提取该流线的特征值数据.
        point_features = np.array(self.all_point_features[line_index])
        feature_data = np.array([])
        if sel_feat == sel_feature[0]: #curvature
            feature_data = point_features[:,0]  # 每个点第0列的数据
        elif sel_feat == sel_feature[1]: #torsion
            feature_data = point_features[:,1]
        elif sel_feat == sel_feature[2]: #tortuosity
            feature_data = point_features[:,2]
        elif sel_feat == sel_feature[3]: #velocity_direction_entropy
            feature_data = point_features[:,3]
        # 1.先按极值点分段.
        segment_point_index = []
        if len(feature_data) <= n_segment:
            segment_point_index = [i for i in range(len(feature_data))]  # 如果点的数目少于分段数，则按点数取
        else:
            diff = feature_data[1]-feature_data[0]
            for i in range(len(feature_data))[1:]:  # 取从第1列（从0开始）到最后一列的数据长度的range，即feature_data长度-1
                diff0 = feature_data[i]-feature_data[i-1]
                if diff*diff0<=0:  # 极值点
                    segment_point_index.append(i-1)
                    diff = diff0
            segment_point_index.insert(0,0) #加入首端点
            segment_point_index.append(len(feature_data)) #加入末端点+1
        # 2.若分段数大于n_segment则合并相邻最小差别的极值点.
        while len(segment_point_index)-1 > n_segment:
            mean_list = [np.mean(feature_data[segment_point_index[i-1]:segment_point_index[i]])\
                 for i in range(len(segment_point_index))[1:]]
            diff_mean_list = [(mean_list[i]-mean_list[i-1]) for i in range(len(mean_list))[1:]]
            segment_point_index.pop(diff_mean_list.index(min(diff_mean_list))+1) #合并分段
        # 3.保存极值点索引.
        self.segment_points_index[line_index] = segment_point_index
        # 4.保存流线分段的长度.
        arc_lengths = self.all_line_arc_lengths[line_index]  # 因为分段存的只是流线的一部分点的索引
        self.all_line_segment_arc_lengths[line_index] = [np.sum(arc_lengths[segment_point_index[i-1]:segment_point_index[i]]) for i in range(len(segment_point_index))[1:]]

    def calculate_all_segment_vectors(self, dim=12):
        print("calculate_all_segment_vectors...")
        print("streamlines_lines_index_data =", len(self.streamlines_lines_index_data))
        for line_index in range(len(self.streamlines_lines_index_data)):
            self.calculate_one_line_segments_vectors(line_index, dim)

    def calculate_one_line_segments_vectors(self, line_index, dim):
        length = len(self.streamlines_lines_index_data)
        perc = [(length-1)//4*v for v in range(5)]
        if line_index in perc:
            s_perc = "|"
            s = "##########"
            s_perc += s * perc.index(line_index)
            print(s_perc, int(perc.index(line_index)/(len(perc)-1)*100), "%")

        segment_feature_vectors = []
        segment_point_index = self.segment_points_index[line_index]
        one_line_segment_indexs = [[segment_point_index[i-1], segment_point_index[i]]\
                                  for i in range(len(segment_point_index))[1:]]

        print("=========one_line_segment_indexs=========")
        print(one_line_segment_indexs)

        for segment_indexs in one_line_segment_indexs:
            segment_feature_vector = self.calculate_one_segment_vectors(line_index, segment_indexs, dim)
            segment_feature_vectors.append(segment_feature_vector)
        self.segment_feature_vectors[line_index] = segment_feature_vectors

    def calculate_one_segment_vectors(self, line_index, segment_indexs, dim=12):
        # 计算每个分段的特征向量(引入金字塔模型).
        # segment_indexs 是 从索引为line_index的流线选出的分段点的array
        # 1.构建金字塔模型.
        pyramid = []
        segment_point_features = self.all_point_features[line_index][segment_indexs[0]:segment_indexs[1]]  # 分段的每一个点的所有特征
        pyramid.append(segment_point_features) #放入第1层
        while len(pyramid[len(pyramid)-1]) > 1: #计算金字塔
            data = pyramid[len(pyramid)-1] #取出金字塔顶层
            top = np.array([(data[i]+data[i-1])/2.0 for i in range(len(data))[1:]])
            pyramid.append(top) #放入第k层
        # 2.从树的根节点取dim维数据构成向量.
        segment_vector = []
        # n_features = len(pyramid[0][0]) #每个点的特征属性数量.
        # dim_max = int(math.pow(2,len(pyramid))-1)*n_features
        for i in range(len(pyramid))[::-1]:  # -1就是除去最后一个tuple
            for row in pyramid[i]:
                for v in row:
                    segment_vector.append(v)
                    if len(segment_vector) == dim: break
                if len(segment_vector) == dim: break
            if len(segment_vector) == dim: break
        if dim > len(segment_vector):
            for i in range(dim-len(segment_vector)):
                segment_vector.append(0.0)
        return segment_vector

    # 这里用不同的聚类算法
    def generate_vocabulary_based_on_streamline_segmentation(self, k=2, v=32):
        print("generate_vocabulary_based_on_streamline_segmentation...")
        # 1.pca主成分分析（降到k维）
        for line,value in self.segment_feature_vectors.items():
            XMat = np.matrix(value)
            finalData, reconMat = pca(XMat, k)
            # plotBestFit(finalData, reconMat)
            self.segment_feature_vectors_kdim[line] = np.array(finalData)
        # 2.KMeans聚类(k=v)
        X = np.array([pt for line,value in self.segment_feature_vectors_kdim.items()\
                                                    for pt in value], dtype="float64")
        km = KMeans(n_clusters=v, random_state=28)
        km.fit(X)
        cluster_indexs = km.predict(X)
        self.dictionary = self.__vectors2words(km.cluster_centers_) # 词典
        no = 0
        for line,value in self.segment_feature_vectors_kdim.items():
            words_index = []
            for seg in value:
                words_index.append(cluster_indexs[no])
                no += 1
            self.segment_vocabularys_index[line] = words_index

    def __vectors2words(self, vecs):
        dictionary = []
        for pt_center in vecs:
            word = ""
            for num in pt_center:
                word = word + str(num) + "_"
            dictionary.append(word)
        return dictionary

    def generate_streamlined_word_vector_expressions(self, v=32):
        print("generate_streamlined_word_vector_expressions...")
        # 1.生成每个分段词汇的独特向量(独热向量的长度即为词汇集合的大小V).
        for i in range(len(self.dictionary)):
            uh = [0 for v in range(v)]
            uh[i] = 1
            self.unique_heat_vectors_dictionary.append(uh)
        # 2.复现如下的算法计算流线的词向量表达.
        for index,streamline in self.segment_vocabularys_index.items():
            S = np.array([0 for k in range(v*v)])
            for i in range(len(streamline)):
                oa = self.unique_heat_vectors_dictionary[self.segment_vocabularys_index[index][i]]
                da = self.all_line_segment_arc_lengths[index][i]
                l  = self.all_line_lengths[index]
                # if l == 0:l=1 # 处理除数为0的特殊情况.
                assert(l != 0)
                for j in range(len(streamline)):
                    if i!=j:
                        ob = self.unique_heat_vectors_dictionary[self.segment_vocabularys_index[index][j]]
                        db = self.all_line_segment_arc_lengths[index][j]
                        e = np.mat(oa).T * np.mat(ob)
                        w = abs(da-db)/l
                        S = S + np.array(e*w).flatten()
            self.all_line_vocabulary_vectors[index] = S
