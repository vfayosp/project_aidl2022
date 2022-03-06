import numpy as np
import torch
from utils.FM_model import FeaturesLinear, FM_operation
import torch.utils.data
from torch_geometric.nn import (
    GCNConv,
    GATConv,
)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GraphModel(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, features, train_mat, heads_to_use=8, attention=False):

        super().__init__()

        self.A = train_mat
        self.features = features
        if attention:
            self.GCN_module = GATConv(int(field_dims), embed_dim, heads=heads_to_use, dropout=0.6)
        else:
            self.GCN_module = GCNConv(field_dims, embed_dim)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return self.GCN_module(self.features, self.A)[x]


class FactorizationMachineModel_withGCN(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim, X, A, heads=8, attention=False):
        super().__init__()

        self.linear = FeaturesLinear(field_dims)
        self.embedding = GraphModel(field_dims, embed_dim, X, A, heads_to_use=heads, attention=attention)
        self.fm = FM_operation(reduce_sum=True)

    def forward(self, interaction_pairs):
        """
        :param interaction_pairs: Long tensor of size ``(batch_size, num_fields)``
        """
        out = self.linear(interaction_pairs) + self.fm(
            self.embedding(interaction_pairs)
        )
        return out.squeeze(1)

    def predict(self, interactions, device):
        # return the score, inputs are numpy arrays, outputs are tensors

        test_interactions = torch.from_numpy(interactions).to(
            dtype=torch.long, device=device
        )
        output_scores = self.forward(test_interactions)
        return output_scores
