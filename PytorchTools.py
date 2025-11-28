import torch
from torch.utils.data import Dataset
import scipy.sparse as sp
import numpy as np
from sklearn.preprocessing import StandardScaler

class scRNADataset(Dataset):
    def __init__(self, train_set, num_gene, flag=False):
        super(scRNADataset, self).__init__()
        self.train_set = train_set
        self.num_gene = num_gene
        self.flag = flag

    def __getitem__(self, idx):
        train_data = self.train_set[:,:2]
        train_label = self.train_set[:,-1]

        if self.flag:
            train_len = len(train_label)
            train_tan = np.zeros([train_len,2])
            train_tan[:,0] = 1 - train_label
            train_tan[:,1] = train_label
            train_label = train_tan

        data = train_data[idx].astype(np.int64)
        label = train_label[idx].astype(np.float32)

        return data, label

    def __len__(self):
        return len(self.train_set)

    def Adj_Generate(self, TF_set, direction=False, loop=False):
        adj = sp.dok_matrix((self.num_gene, self.num_gene), dtype=np.float32)

        for pos in self.train_set:
            tf = pos[0]
            target = pos[1]

            if direction == False:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    adj[target, tf] = 1.0
            else:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    if target in TF_set:
                        adj[target, tf] = 1.0

        if loop:
            adj = adj + sp.identity(self.num_gene)
        adj = adj.todok()
        return adj

class load_data():
    def __init__(self, data, normalize=True):
        self.data = data
        self.normalize = normalize

    def data_normalize(self, data):
        standard = StandardScaler()
        epr = standard.fit_transform(data.T)
        return epr.T

    def exp_data(self):
        data_feature = self.data.values

        if self.normalize:
            data_feature = self.data_normalize(data_feature)

        data_feature = data_feature.astype(np.float32)
        return data_feature