import math
import torch
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score

def getHitRatio(recommend_list, gt_item):
    if gt_item in recommend_list:
        return 1
    else:
        return 0

def getNDCG(recommend_list, gt_item):
    idx = np.where(recommend_list == gt_item)[0]
    if len(idx) > 0:
        return math.log(2)/math.log(idx+2)
    else:
        return 0
    
def build_adj_mx(dims, interactions):
    train_mat = sp.dok_matrix((dims, dims), dtype=np.float32)
    for x in tqdm(interactions, desc="BUILDING ADJACENCY MATRIX..."):
        train_mat[x[0], x[1]] = 1.0
        train_mat[x[1], x[0]] = 1.0

    return train_mat

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """ Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)