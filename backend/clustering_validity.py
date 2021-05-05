"""
2021
for cluster analysis
using t-sne

"""
import torch
from sklearn import metrics
from scipy.spatial import distance
import numpy as np
from tqdm import tqdm
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class clustering_validity_analysis():
    """
    需要做定量分析
    1. silhouette
    2. hubert gamma
    3. DB indexs
    4. normalized validity
    """
    def __init__(self, data, labels, centers):
        self.dataMatrix = data  # numpy array
        self.classLabel = labels  # numpy array
        self.cluster_centers = centers  # numpy array

        self.data_tensor = torch.from_numpy(data)
        self.labels_tensor = torch.IntTensor(labels)

        self.data_tensor = self.data_tensor.to(device)  # 传送给gpu
        self.labels_tensor = self.labels_tensor.to(device)

    def Silhouette_Coefficient(self):
        validation = metrics.silhouette_score(X=self.dataMatrix, labels=self.classLabel, metric='euclidean')
        return validation

    def Davies_Bouldin_Index(self):
        validation = metrics.davies_bouldin_score(X=self.dataMatrix, labels=self.classLabel)
        return validation

    def Hubert_Gamma_Score(self):
        """
        		The Modified Hubert T Statistic, a measure of compactness.
        		"""
        print("The Modified Hubert T Statistic, a measure of compactness")
        sumDiff = 0

        # compute the centers of all the clusters
        # list_center = []
        # numCluster = max(self.classLabel) + 1
        # for i in range(numCluster):
        #     indices = [t for t, x in enumerate(self.classLabel) if x == i]
        #     clusterMember = self.dataMatrix[indices, :]
        #     list_center.append(np.mean(clusterMember, 0))

        size = len(self.classLabel)

        # iterate through each of the two pairs exhaustively
        print("iterate through each of the two pairs exhaustively...")
        for i in tqdm(range(size - 1)):
            for j in range(i + 1, size):
                # get the cluster labels of the two objects
                label1 = self.classLabel[i]
                label2 = self.classLabel[j]
                # compute the distance of the two objects
                pairDistance = distance.euclidean(self.dataMatrix[i], self.dataMatrix[j])
                # compute the distance of the cluster center of the two objects
                centerDistance = distance.euclidean(self.cluster_centers[label1], self.cluster_centers[label2])
                # add the product to the sum
                sumDiff = sumDiff + pairDistance * centerDistance
        # compute the fitness
        validation = 2 * sumDiff / (size * (size - 1))
        return validation

    def PBM_index(self):
        """
        The PBM index, a measure of compactness
        """
        ew=0
        et=0
        list_centerDis=[]
        numCluster=max(self.classLabel)+1
        #compute the center of the dataset
        dataCenter=np.mean(self.dataMatrix,0)
        #iterate through the  clusters
        for i in tqdm(range(numCluster)):
            indices=[t for t, x in enumerate(self.classLabel) if x == i]
            clusterMember=self.dataMatrix[indices,:]
            #compute the center of the cluster
            clusterCenter=np.mean(clusterMember,0)
            #compute the center distance
            list_centerDis.append(distance.euclidean(dataCenter, clusterCenter))
            #iterate through the member of the  cluster
            for member in clusterMember:
                ew=ew+distance.euclidean(member, clusterCenter)
                et=et+distance.euclidean(member, dataCenter)
        db=max(list_centerDis)
        #compute the fitness
        validation = math.pow(et*db/(numCluster*ew),2)
        return validation
