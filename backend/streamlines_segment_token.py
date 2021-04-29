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
from backend.stream_space_curve import StreamSpaceCurve
from backend.pca import pca
from sklearn.cluster import KMeans, DBSCAN, OPTICS, MeanShift, AffinityPropagation, estimate_bandwidth
from pyclustering.cluster.cure import cure
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn import metrics

import scipy.spatial.distance as dist
import pandas as pd
from datetime import datetime

sel_feature = [
    "curvature",  # 曲率
    "torsion",  # 扭率
    "tortuosity",  # 弯曲度
    "velocity_direction_entropy"  # 速度方向熵，计算有问题，得到的值全是0
]

starttime = datetime.now()
endtime = datetime.now()

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
        # 量化的聚类度量指标
        self.silhouette_coefficient = .0
        self.db_index = .0
        self.hubert_gamma = .0
        self.normalized_validity = .0

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
        self.segment_all_lines(sel_feature[0], n_segment=50) #【注】此处的n_segment应该由外部主函数传入. 一般取50-100足够
        # 计算流线分段的特征向量.
        self.calculate_all_segment_vectors(dim=12) #【注】此处的dim应该由外部主函数传入，暂时写为特定值. dim只能是len(sel_feature)的倍数
        # 基于流线分段的特征向量生成词汇. 用到聚类算法
        global endtime
        endtime = datetime.now()
        print("Finish calculate all segment vectors: " + str(endtime-starttime))
        self.generate_vocabulary_based_on_streamline_segmentation(k=2, v=32)

        # 生成流线的词向量表达.
        endtime = datetime.now()
        print("Start generate streamline word vector: " + str(endtime - starttime))
        self.generate_streamlined_word_vector_expressions(v=32)

    def calculate_main_streamline_index(self, cnt, distant_typeid=0):
        """应用1：流场压缩(pca+kmeans).
        """
        print("calculate_main_streamline_index ", cnt, " ...")
        assert(0 <= cnt <= len(self.all_line_vocabulary_vectors))
        # 1.计算不相似度矩阵.
        dissimilarity_matrix = self.__calculate_dissimilarity_matrix(distant_typeid)
        # 2.筛选cnt条最不相似的流线.
        S = []
        max_dissim = 0 # 初始化最大不相似度为0.
        if cnt > 0: # 筛选出具有最大不相似度的一条流线.
            S.append(np.where(dissimilarity_matrix==np.max(dissimilarity_matrix))[0][0])
        while len(S) < cnt:
            tmp_line = None # 初始化当前最不相似流线.
            max_dissim = 0 # 初始化最大不相似度为0.
            for x in self.all_line_vocabulary_vectors.keys():
                if x in S:
                    continue
                tmp_dissim = 0 # 初始化当前不相似度为0.
                for y in S:
                    tmp_dissim += dissimilarity_matrix[x][y]
                if tmp_dissim > max_dissim:
                    tmp_line = x
            S.append(tmp_line)
        # 3.整理筛选出的最不相似流线索引，及其相应的不相似度矩阵.
        S.sort()
        prototype_index = np.array(S)
        dissimilarity_matrix = dissimilarity_matrix[:,S]
        return dissimilarity_matrix, prototype_index

    def __calculate_dissimilarity_matrix(self, distant_typeid=0):
        dissimilarity_matrix = []
        for index_x, vocabulary_x in self.all_line_vocabulary_vectors.items():
            row_data = []
            for index_y, vocabulary_y in self.all_line_vocabulary_vectors.items():
                row_data.append(self.__dissim(np.array(vocabulary_x), np.array(vocabulary_y), distant_typeid))
            dissimilarity_matrix.append(np.array(row_data))
        return np.array(dissimilarity_matrix)

    def __dissim(self, x, y, distant_typeid=0):
        distance = 0
        if distant_typeid == 0:  # 欧几里得距离.
            distance = np.linalg.norm(x-y)
        elif distant_typeid == 1:  # 余弦距离.
            distance = 1 - np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
        elif distant_typeid == 2:  # 曼哈顿距离.
            distance = np.linalg.norm(x-y,ord=1)
        elif distant_typeid == 3:  # 切比雪夫距离.
            distance = np.linalg.norm(x-y,ord=np.inf)
        elif distant_typeid == 4:  # 夹角余弦(Cosine).
            distance = np.dot(x,y)/(np.linalg.norm(x)*(np.linalg.norm(y)))
        elif distant_typeid == 5:  # 汉明距离(Hamming distance).
            distance = np.shape(np.nonzero(x-y)[0])[0]
        elif distant_typeid == 6:  # 杰卡德相似系数(Jaccard similarity coefficient).
            distance = dist.pdist(np.array([x,y]),'jaccard')[0]
            # 以下待补充度量方法

        else:
            raise("[ERROR] Unkonw distant type id!")
        return distance

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
        """
        :param line_index: 流线索引
        :param sel_feat: 用于分段的特征属性
        :param n_segment: 最大分段数
        :return:
        """
        length = len(self.streamlines_lines_index_data)
        perc = [(length-1)//10*v for v in range(11)]
        # e.g. length=2001, then perc=[500*v for v in range(5)], that is perc=[0, 500, 1000, 1500, 2000]
        if line_index in perc:
            s_perc = "|"
            s = "#####"
            s_perc += s * perc.index(line_index)  # s_perc = 几倍的s ##########
            print(s_perc, int(perc.index(line_index)/(len(perc)-1)*100), "%")
            # e.g. line_index = 1500
            # print(s_perc, int(3)/(5-1)*100, "%"), that is 75%
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
        # 3.保存极值点索引. 相当于保存了流线分段
        self.segment_points_index[line_index] = segment_point_index
        # 4.保存流线分段的长度.
        arc_lengths = self.all_line_arc_lengths[line_index]  # 因为分段存的只是流线的一部分点的索引
        self.all_line_segment_arc_lengths[line_index] = [np.sum(arc_lengths[segment_point_index[i-1]:segment_point_index[i]]) for i in range(len(segment_point_index))[1:]]

    def calculate_all_segment_vectors(self, dim=12):
        print("calculate_all_segment_vectors...")
        print("streamlines_lines_index_data =", len(self.streamlines_lines_index_data))  #len()是vtk文件流线总数
        global endtime
        endtime = datetime.now()
        print("Start calculate all segment vectors: "+str(endtime-starttime))
        for line_index in range(len(self.streamlines_lines_index_data)):
            self.calculate_one_line_segments_vectors(line_index, dim)

    def calculate_one_line_segments_vectors(self, line_index, dim):
        # 以下的7行代码在相同代码段已做解释
        length = len(self.streamlines_lines_index_data)
        perc = [(length-1)//10*v for v in range(11)]
        if line_index in perc:
            s_perc = "|"
            s = "#####"
            s_perc += s * perc.index(line_index)
            print(s_perc, int(perc.index(line_index)/(len(perc)-1)*100), "%")

        segment_feature_vectors = []
        segment_point_index = self.segment_points_index[line_index]  # 得到一条流线的分段index，例如[0, 20, 46, 70]
        one_line_segment_indexs = [[segment_point_index[i-1], segment_point_index[i]]\
                                  for i in range(len(segment_point_index))[1:]]  # 一条分段首尾点的index [0, 20],[20,46],[46, 70]

        # print("=========one_line_segment_indexs=========")
        # print(one_line_segment_indexs)

        for segment_indexs in one_line_segment_indexs:
            segment_feature_vector = self.calculate_one_segment_vectors(line_index, segment_indexs, dim)
            segment_feature_vectors.append(segment_feature_vector)
        self.segment_feature_vectors[line_index] = segment_feature_vectors

    def calculate_one_segment_vectors(self, line_index, segment_indexs, dim=12):
        # 计算每个分段的特征向量(引入金字塔模型).
        # dim只能是len(sel_feature)的倍数
        # segment_indexs 是 流线分段的起始点index
        # 1.构建金字塔模型.
        pyramid = []
        segment_point_features = self.all_point_features[line_index][segment_indexs[0]:segment_indexs[1]]  # 该条流线该分段的每一个点的所有特征
        pyramid.append(segment_point_features) #放入第1层
        while len(pyramid[len(pyramid)-1]) > 1: #计算金字塔，该循环进行len(pyramid)-1次
            data = pyramid[len(pyramid)-1] #取出金字塔顶层
            # print("pyramid data: " + str(data))
            # print("len data: " + str(len(data)))
            top = np.array([(data[i]+data[i-1])/2.0 for i in range(len(data))[1:]])  # i从1开始，range()[1:]，去除首项
            pyramid.append(top) #放入第k层
        # 最后得到的pyramid相当于原始流线分段n个点的多维point_feature的全部数据整合，有多层，最后一项是顶层数据
        """
        pyramid data: 
        [[0.02024883 0.38287121 1.25724939 0.        ]
         [0.02444911 0.05811907 1.255526   0.        ]
         [0.02739758 0.18493743 1.25378871 0.        ]
         [0.03576561 0.16550325 1.25202655 0.        ]
         [0.05129829 0.06765268 1.25024731 0.        ]
         [0.06468541 0.04450216 1.24843147 0.        ]
         [0.073136   0.04595697 1.24656229 0.        ]
         [0.08813662 0.05235368 1.24463737 0.        ]
         [0.11345011 0.03221319 1.24267797 0.        ]
         [0.13589151 0.01512622 1.24064795 0.        ]
         [0.15023672 0.01289563 1.23855948 0.        ]
         [0.18579385 0.01604874 1.236414   0.        ]
         [0.1935932  0.01951918 1.23580314 0.        ]
         [0.24255992 5.85920017 1.23270757 0.        ]]
        len data: 14
        """
        # 2.从树的根节点取dim维数据构成向量.
        segment_vector = []
        # n_features = len(pyramid[0][0]) #每个点的特征属性数量.
        # dim_max = int(math.pow(2,len(pyramid))-1)*n_features
        # print("len of pyramid: \n" + str(len(pyramid)))
        # print("pyramid : \n" + str(pyramid))  # pyramid是变长数组，每一项逐index长度减一
        # e.g. len(pyramid)=14，则第一项长度14，该项内部由14个长度为4的array组成
        # 第二项长度就为13
        # print("Start for i: ")
        for i in range(len(pyramid))[::-1]:
            # -1就是逆序递减，i从len(pyramid)-1到0，但此循环最多只进行2次，why?因为一项是4维，而dim=12，第一次循环append4个数，第二次循环append8个数
            # print("i in range(len(pyramid))[::-1] : " + str(i))
            # print("Start for row: ")
            for row in pyramid[i]:
                # print("row of pyramid[i]: \n" + str(row))
                for v in row:  # len(row)始终是4
                    # print("for v in row: \n" + str(v))
                    segment_vector.append(v)
                    if len(segment_vector) == dim: break
                if len(segment_vector) == dim: break
            if len(segment_vector) == dim: break
        """
        len of pyramid: 6
        row of pyramid[i]: [0.3903928  1.49570273 1.00859226 0.        ]
        for v in row: 0.39039279908200425
        for v in row: 1.4957027281276754
        for v in row: 1.0085922622328294
        for v in row: 0.0
        row of pyramid[i]: [0.37064722 2.88413528 1.0092199  0.        ]
        for v in row: 0.370647218337418
        for v in row: 2.884135277414701
        for v in row: 1.0092199022797677
        for v in row: 0.0
        row of pyramid[i]: [0.41013838 0.10727018 1.00796462 0.        ]
        for v in row: 0.4101383798265905
        for v in row: 0.1072701788406496
        for v in row: 1.0079646221858911
        for v in row: 0.0
        """
        if dim > len(segment_vector):
            for i in range(dim-len(segment_vector)):
                segment_vector.append(0.0)
        # print("segment_vector: \n" + str(segment_vector))

        return segment_vector

    # 这里用不同的聚类算法
    def generate_vocabulary_based_on_streamline_segmentation(self, k=2, v=32):
        # cost long time

        print("generate_vocabulary_based_on_streamline_segmentation...")
        # 1.pca主成分分析（降到k维）
        # print("self.segment_feature_vectors.items():")
        # print(self.segment_feature_vectors.items())
        global endtime
        endtime = datetime.now()
        print("Start PCA reduce dimension: " + str(endtime-starttime))
        for line,value in self.segment_feature_vectors.items():
            # print("line, value: " + str(line))
            # print(value)
            # segment_feature_vectors是dict, line就是key
            XMat = np.matrix(value)
            finalData, reconMat = pca(XMat, k)
            # plotBestFit(finalData, reconMat)
            self.segment_feature_vectors_kdim[line] = np.array(finalData)
        # print("self.segment_feature_vectors_kdim: ")
        # print(self.segment_feature_vectors_kdim)
        # 2. 聚类
        # 每一种聚类方法最后只需保留cluster_labels和cluster_centers
        # cluster_labels = cluster_method.labels_ or cluster_method.predict(new_dataset)
        # cluster_centers = cluster_method.cluster_centers_

        # streamlines_segment_token.py:210: ComplexWarning: Casting complex values to real discards the imaginary part
        #   for pt in value], dtype="float64")
        # 此时self.segment_feature_vectors_kdim都降成了k维，即pt均为k维
        endtime = datetime.now()
        print("Finish PCA reduce dimension and Start Clustering: " + str(endtime - starttime))
        X = np.array([pt for line, value in self.segment_feature_vectors_kdim.items() for pt in value], dtype="float64")  # 2668

        df = pd.DataFrame(X)
        df.to_csv('pandas_to_csv_X.csv', header=False, index=False)

        # 2.1 KMeans聚类(k=v)  ok
        if (self.cluster_mode == 'KMeans'):
            random_state = 28
            print("Start Clustering for KMeans to " + str(v) + " clusters, beginning with " + str(random_state) + " centroids...")
            km = KMeans(n_clusters=v, random_state=random_state)
            km.fit(X)
            cluster_labels = km.predict(X)  # X中每个数据点的簇号
            # print(cluster_labels)
            # print(len(cluster_labels))  # 2668

            cluster_centers = km.cluster_centers_  # 每个簇的中心点坐标
            print("length of cluster_centers:")
            # print(cluster_centers)
            print(len(cluster_centers))

            # 1. Silhouette Coefficient
            # The score is higher when clusters are dense and well separated.
            self.silhouette_coefficient = self.validity_measurement_silhouette(data=X, cluster_labels=cluster_labels)

            # 2. Davies-Bouldin Index
            # Values closer to zero indicate a better partition.
            self.db_index = self.validity_measurement_db_index(data=X, cluster_labels=cluster_labels)

        # 以下为补充的聚类方法（6种）

        # 2.2 KMedoids  ok
        elif(self.cluster_mode == 'KMedoids'):
            # error when encountering big dataset, it will stuck
            initial_medoids = kmeans_plusplus_initializer(X, v).initialize(return_index=True)
            print("Initial Medoids: ")
            print(initial_medoids)
            print("Start Clustering for KMedoids to " + str(v) + " clusters, beginning with " + str(len(initial_medoids)) + " centroids...")
            iter_max = 2000
            # kmedoids_instance = kmedoids(data=X, initial_index_medoids=initial_medoids, itermax=iter_max)
            kmedoids_instance = kmedoids(data=X, initial_index_medoids=initial_medoids)
            kmedoids_instance.process()

            cluster_medoids = kmedoids_instance.get_medoids()  # 只是索引，不是具体的二维数据点

            cluster_centers = []
            for idx in cluster_medoids:
                cluster_centers.append(X[idx])
            print(cluster_centers)
            print(len(cluster_centers))

            cluster_labels = kmedoids_instance.predict(X)
            print(cluster_labels)
            print(len(cluster_labels))
            # 1. Silhouette Coefficient
            # The score is higher when clusters are dense and well separated.
            self.silhouette_coefficient = self.validity_measurement_silhouette(data=X, cluster_labels=cluster_labels)

            # 2. Davies-Bouldin Index
            # Values closer to zero indicate a better partition.
            self.db_index = self.validity_measurement_db_index(data=X, cluster_labels=cluster_labels)

        # 2.3 DBSCAN ok
        elif(self.cluster_mode == 'DBSCAN'):
            # 这两个问题已解决4.29
            # Q1: ValueError: Found input variables with inconsistent numbers of samples: [1594, 1017]
            # Q2: 问题是去除掉噪声点后，数据匹配不上self.segment_feature_vectors_kdim.items()的长度，labels短了
            print("Start Clustering for DBSCAN...")
            print("Parameters: ")
            # eps保持不变，increase min_samples，
            # that will decrease the sizes of individual clusters and increase the number of clusters
            eps = 0.2
            min_samples = 50  # double dataset dimensionality
            print("eps="+str(eps))
            print("min_samples="+str(min_samples))
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            # 做聚类结果评估的时候不需要去除噪声/离群点，所以不需要处理labels
            cluster_labels = db.fit_predict(X)
            # 以下注释掉的是去除噪声/离群点的方法
            # __cluster_labels = db.fit_predict(X)
            # cluster_labels = []
            # for idx in db.core_sample_indices_:
            #     # t = __cluster_labels[idx]
            #     # print(t)
            #     cluster_labels.append(__cluster_labels[idx])

            # 将labels中每簇第一个索引作为center
            __cluster_centers_idx = []
            for idx in np.unique(cluster_labels):
                pos = list(cluster_labels).index(idx)
                __cluster_centers_idx.append(pos)
            cluster_centers = []
            for idx in __cluster_centers_idx:
                pts = X[idx]
                cluster_centers.append(pts)
            print("length of cluster_centers:")
            # print(cluster_centers)
            print(len(cluster_centers))

            # 1. Silhouette Coefficient
            # The score is higher when clusters are dense and well separated.
            self.silhouette_coefficient = self.validity_measurement_silhouette(data=X, cluster_labels=cluster_labels)

            # 2. Davies-Bouldin Index
            # Values closer to zero indicate a better partition.
            self.db_index = self.validity_measurement_db_index(data=X, cluster_labels=cluster_labels)

            # DBSCAN得到的labels为-1的点应该要从X中去除，那么数据长度就会发生变化，对不上dictionary

        # 2.4 OPTICS ok 但聚类效果不好
        elif(self.cluster_mode == 'OPTICS'):
            print("Start Clustering for OPTICS...")
            # min_samples=8, xi=0.15, min_cluster_size=10 这样的参数设置对于原始流场来说，产生的簇太多了，有835个，sil_co分数为负

            # 一个点要想成为核心点，与其本身距离不大于epsilon的点的数目至少有min_samples个
            min_samples = 25  # 对于原始流场，这个参数应该要更大，试试25-50
            print("min_samples=" + str(min_samples))

            xi = .15 # higher, less clusters
            print("xi="+str(xi))

            # 一个簇至少包含的点数目
            min_cluster_size = 50  # 对于原始流场，这个参数应该要更大，50-100为佳
            print("min_cluster_size="+str(min_cluster_size))

            eps = 0.2
            # print("eps="+str(eps))

            max_eps = 1.
            # print("max_eps="+str(max_eps))

            print("Parameters: ")
            opt_instance = OPTICS(min_samples=min_samples,
                                  xi=xi,
                                  min_cluster_size=min_cluster_size,
                                  # eps=eps,
                                  # max_eps=max_eps
                                  ).fit(X)
            cluster_labels = opt_instance.fit_predict(X)
            # 将labels中每簇第一个索引作为center
            __cluster_centers_idx = []
            for idx in np.unique(cluster_labels):
                pos = list(cluster_labels).index(idx)
                __cluster_centers_idx.append(pos)
            cluster_centers = []
            for idx in __cluster_centers_idx:
                pts = X[idx]
                cluster_centers.append(pts)

            print("length of cluster_centers:")
            # print(cluster_centers)
            print(len(cluster_centers))

            # 1. Silhouette Coefficient
            # The score is higher when clusters are dense and well separated.
            self.silhouette_coefficient = self.validity_measurement_silhouette(data=X, cluster_labels=cluster_labels)

            # 2. Davies-Bouldin Index
            # Values closer to zero indicate a better partition.
            self.db_index = self.validity_measurement_db_index(data=X, cluster_labels=cluster_labels)

        # 2.5 MeanShift ok
        elif (self.cluster_mode == 'MeanShift'):
            print("Start Clustering for MeanShift...")
            # time-complexity O(n^2), n is the number of points
            print("Parameters for estimating bandwidth: ")
            quantile = 0.4
            print("quantile="+str(quantile))
            bandwidth = estimate_bandwidth(X, quantile=quantile)
            print("Parameters for Meanshift: ")
            print("bandwidth="+str(bandwidth))
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(X)
            cluster_labels = ms.predict(X)
            # print(cluster_labels)
            # print(len(cluster_labels))

            cluster_centers = ms.cluster_centers_
            # print(cluster_centers)
            # print(len(cluster_centers))

            print("length of cluster_centers:")
            # print(cluster_centers)
            # print(len(cluster_centers))

            # 1. Silhouette Coefficient
            # The score is higher when clusters are dense and well separated.
            self.silhouette_coefficient = self.validity_measurement_silhouette(data=X, cluster_labels=cluster_labels)

            # 2. Davies-Bouldin Index
            # Values closer to zero indicate a better partition.
            self.db_index = self.validity_measurement_db_index(data=X, cluster_labels=cluster_labels)

        # 2.6 AP ok
        elif(self.cluster_mode == 'AffinityPropagation' or self.cluster_mode == 'AP'):
            print("Start Clustering for AffinityPropagation...")
            print("Parameters: ")
            damping = 0.85
            print("damping="+str(damping))
            preference = -1000  # smaller, clusters less
            print("preference="+str(preference))
            max_iter = 2000
            print("max_iter=" + str(max_iter))

            ap_instance = AffinityPropagation(random_state=0,
                                     verbose=True,
                                     max_iter=max_iter,
                                     damping=damping,
                                     preference=preference).fit(X)
            cluster_labels = ap_instance.fit_predict(X)
            # print(cluster_labels)
            # print(len(cluster_labels))

            cluster_centers = ap_instance.cluster_centers_
            # print(cluster_centers)
            # print(len(cluster_centers))

            print("length of cluster_centers:")
            # print(cluster_centers)
            # print(len(cluster_centers))

            # 1. Silhouette Coefficient
            # The score is higher when clusters are dense and well separated.
            self.silhouette_coefficient = self.validity_measurement_silhouette(data=X, cluster_labels=cluster_labels)

            # 2. Davies-Bouldin Index
            # Values closer to zero indicate a better partition.
            self.db_index = self.validity_measurement_db_index(data=X, cluster_labels=cluster_labels)

        # 2.7 CURE
        elif(self.cluster_mode == 'CURE'):
            print("Start Clustering for CURE...")
            print("Parameters: ")
            number_cluster = v
            print("number_cluster="+str(number_cluster))
            number_represent_points = 10
            print("number_present_points="+str(number_represent_points))

            compression = 0.3
            # Coefficient defines level of shrinking of representation points \
            #   toward the mean of the new created cluster after merging on each step.
            # Usually it destributed from 0 to 1
            print("compression="+str(compression))

            cure_instance = cure(data=X,
                                 number_cluster=number_cluster,
                                 number_represent_points=number_represent_points,
                                 compression=compression)
            cure_instance.process()
            clusters = cure_instance.get_clusters()

            # create cluster labels for all points
            cluster_labels = [None]*len(X)
            for idx in range(len(clusters)):
                for pts in clusters[idx]:
                    cluster_labels[pts] = idx
            # print(cluster_labels)
            # print(len(cluster_labels))
            # 以label中每个簇最早出现的位置定为center的索引
            cluster_centers = []
            for idx in np.unique(cluster_labels):
                pos = list(cluster_labels).index(idx)
                cluster_centers.append(X[pos])
            # print("cluster_centers: ")
            # print(cluster_centers)
            # print(len(cluster_centers))

            # 1. Silhouette Coefficient
            # The score is higher when clusters are dense and well separated.
            self.silhouette_coefficient = self.validity_measurement_silhouette(data=X, cluster_labels=cluster_labels)

            # 2. Davies-Bouldin Index
            # Values closer to zero indicate a better partition.
            self.db_index = self.validity_measurement_db_index(data=X, cluster_labels=cluster_labels)

        else:
            print("Error Clustering Method...The Program Will Exit Soon...")
            exit(-1)

        # doing clustering, get cluster_labels and cluster_centers(some clustering algorithms)
        # doing metrics
        # 1. Silhouette Coefficient
        # The score is higher when clusters are dense and well separated.
        # self.silhouette_coefficient = self.validity_measurement_silhouette(data=X, cluster_labels=cluster_labels)

        # 2. Davies-Bouldin Index
        # Values closer to zero indicate a better partition.
        # self.db_index = self.validity_measurement_db_index(data=X, cluster_labels=cluster_labels)

        # 3. Hubert's gamma statistics
        # empty

        # 4. Normalized validity measurement
        # empty
        endtime = datetime.now()
        print("Finish Clustering: " + str(endtime - starttime))

        self.dictionary = self.__vectors2words(cluster_centers)  # 词典
        print("\nlength of self.dictionary:")
        print(len(self.dictionary))
        # print(self.dictionary)
        # exit()

        no = 0
        # print("Dictionary: ")
        # print(self.dictionary)
        # print(len(self.dictionary))

        pd3 = pd.DataFrame(self.dictionary)
        pd3.to_csv('X_dict.csv', header=False, index=False)

        # print("Segment_feature_vectors_kdim.items(): ")
        # print(self.segment_feature_vectors_kdim.items())
        # print(type(self.segment_feature_vectors_kdim))
        # print(len(self.segment_feature_vectors_kdim))  # 2080

        # pd2 = pd.DataFrame(self.segment_feature_vectors_kdim.items())
        # pd2.to_csv('segment_kdim_items.csv', header=False, index=False)

        # generate words and vocabulary
        for line, value in self.segment_feature_vectors_kdim.items():
            # line is the index of one streamline in all
            # value is the points' coordinate in one streamline
            words_index = []
            # print("line, value: ")
            # print(line, value)
            for seg in value:
                words_index.append(cluster_labels[no])  # no is added from 0 to numbers of total points
                no += 1
                # print("words_index: " + str(no) + " : " + str(seg))  # seg is the exact point coordinate
                # print(words_index) # len(words_index) is up to the len(streamline's points)
            self.segment_vocabularys_index[line] = words_index  # words_index's length is different

    def __vectors2words(self, vecs):
        dictionary = []
        for pt_center in vecs:
            word = ""
            for num in pt_center:
                word = word + str(num) + "_"
                # print("num and word: ")
                # print(num, word)
            dictionary.append(word)
        return dictionary

    def generate_streamlined_word_vector_expressions(self, v=32):
        """
        :param v:
        :return:
        """
        print("generate_streamlined_word_vector_expressions...")
        global endtime
        # consume time long, need a progress bar

        # 1.生成每个分段词汇的独特向量(独热向量的长度即为词汇集合的大小V).
        v = len(self.dictionary)
        for i in range(len(self.dictionary)):
            uh = [0 for v in range(v)]  # unique-hot
            # IndexError: list assignment index out of range, uh[i]=1. 因为一开始v都默认成32了
            uh[i] = 1
            self.unique_heat_vectors_dictionary.append(uh)
        # 2.复现如下的算法计算流线的词向量表达.
        # 词向量表达就是矩阵乘法运算
        # ont-hot矩阵N*1，S矩阵N*N，词向量表达ret矩阵=S*one-hot=N*1矩阵，其实就是S中的某一行向量
        len_segment_vocabulary_index = len(self.segment_vocabularys_index.items())
        process_bar = [(len_segment_vocabulary_index-1)//25 * i for i in range(26)]
        for index, streamline in self.segment_vocabularys_index.items():
            # segment_vocabularys_index.items()是[pts.x_pts.y_]这样的表达形式
            if index in process_bar:
                endtime = datetime.now()
                print("Processing " + str(index) + " streamline: " + str(endtime-starttime))
                s_perc = "|"
                s = "##"
                s_perc += s * process_bar.index(index)  # s_perc = 几倍的s ##########
                print(s_perc, int(process_bar.index(index) / (len(process_bar) - 1) * 100), "%")

            S = np.array([0 for k in range(v*v)])
            for i in range(len(streamline)):
                # print("i in range(len(streamline)): " + str(i))
                oa = self.unique_heat_vectors_dictionary[self.segment_vocabularys_index[index][i]]
                da = self.all_line_segment_arc_lengths[index][i]
                l  = self.all_line_lengths[index]
                # if l == 0:l=1 # 处理除数为0的特殊情况.
                assert(l != 0)
                for j in range(len(streamline)):
                    if i != j:
                        ob = self.unique_heat_vectors_dictionary[self.segment_vocabularys_index[index][j]]
                        db = self.all_line_segment_arc_lengths[index][j]
                        e = np.mat(oa).T * np.mat(ob)
                        w = abs(da-db)/l
                        S = S + np.array(e*w).flatten()
            self.all_line_vocabulary_vectors[index] = S

    def validity_measurement_silhouette(self, data, cluster_labels):
        sil_score = metrics.silhouette_score(data, cluster_labels, metric='euclidean')
        print("Silhouette Coefficient: " + str(sil_score))
        return sil_score

    def validity_measurement_db_index(self, data, cluster_labels):
        db_index = metrics.davies_bouldin_score(data, cluster_labels)
        print("Davies-Bouldin Index: " + str(db_index))
        return db_index

    def validity_measurement_hubert_gamma(self, data, cluster_labels):
        hubert_gamma = .0

    def validity_measurement_normalized(self, data, cluster_labels):
        normalized_validity = .0


