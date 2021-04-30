"""
4项聚类结果度量指标
1. Silhouette Coefficient
    The score is higher when clusters are dense and well separated.
2. Davies-Bouldin Index
    Values closer to zero indicate a better partition.
    The Davies-Bouldin index, the average of all cluster similarities.
3. Hubert's gamma statistics

4. Normalized validity measurement

"""
import torch
from sklearn import metrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MyValidity():
    def __init__(self, data, labels):
        self.data_tensor = torch.from_numpy(data)
        self.labels_tensor = torch.IntTensor(labels)

        self.data_tensor = self.data_tensor.to(device)  # 传送给gpu
        self.labels_tensor = self.labels_tensor.to(device)

    def Silhouette_Coefficient(self):
        validation = metrics.silhouette_score(X=self.data_tensor.cpu(), labels=self.labels_tensor.cpu(), metric='euclidean')
        return validation

    def Davies_Bouldin_Index(self):
        validation = metrics.davies_bouldin_score(X=self.data_tensor.cpu(), labels=self.labels_tensor.cpu())
        return validation
