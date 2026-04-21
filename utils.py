import pandas as pd
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

class scRNADataset(Dataset):
    def __init__(self, train_set, num_gene, flag=False):
        super(scRNADataset, self).__init__()
        self.train_set = train_set
        self.num_gene = num_gene
        self.flag = flag

    def __getitem__(self, idx):
        train_data = self.train_set[:, :2]
        train_label = self.train_set[:, -1]

        if self.flag:
            train_len = len(train_label)
            train_tan = np.zeros([train_len, 2])
            train_tan[:, 0] = 1 - train_label
            train_tan[:, 1] = train_label
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

def adj2saprse_tensor(adj):
    coo = adj.tocoo()
    i = torch.LongTensor(np.array([coo.row, coo.col]))
    v = torch.from_numpy(coo.data).float()
    adj_sp_tensor = torch.sparse_coo_tensor(i, v, coo.shape)
    return adj_sp_tensor

def Evaluation(y_true, y_pred, flag=False):
    if flag:
        y_p = y_pred[:, -1]
        y_p = y_p.cpu().detach().numpy().flatten()
    else:
        y_p = y_pred.cpu().detach().numpy().flatten()

    y_t = y_true.cpu().numpy().flatten().astype(int)

    AUC = roc_auc_score(y_true=y_t, y_score=y_p)
    AUPR = average_precision_score(y_true=y_t, y_score=y_p)
    AUPR_norm = AUPR / np.mean(y_t)

    return AUC, AUPR, AUPR_norm

def normalize(expression):
    std = StandardScaler()
    epr = std.fit_transform(expression)
    return epr

def embed2file(tf_embed, tg_embed, gene_file, tf_path, target_path):
    tf_embed = tf_embed.cpu().detach().numpy()
    tg_embed = tg_embed.cpu().detach().numpy()

    gene_set = pd.read_csv(gene_file, index_col=0)

    tf_embed = pd.DataFrame(tf_embed, index=gene_set['Gene'].values)
    tg_embed = pd.DataFrame(tg_embed, index=gene_set['Gene'].values)

    tf_embed.to_csv(tf_path)
    tg_embed.to_csv(target_path)