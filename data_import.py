import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import torch
import os


def build_adj_mx(dims, interactions):
    train_mat = sp.dok_matrix((dims, dims), dtype=np.float32)
    for x in tqdm(interactions, desc="BUILDING ADJACENCY MATRIX..."):
        train_mat[x[0], x[1]] = 1.0
        train_mat[x[1], x[0]] = 1.0

    return train_mat


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        type_data,
        data_path="data",
        num_negatives_train=4,
        num_negatives_test=None,
    ):
        # type_data can be drug_protein,drug_disease or protein_disease

        self.data = pd.read_csv(
            os.path.join(data_path, "df_train_" + type_data + ".csv"),
            sep="\t",
            index_col="Unnamed: 0",
        )
        self.test_data = pd.read_csv(
            os.path.join(data_path, "df_test_" + type_data + ".csv"),
            sep="\t",
            index_col="Unnamed: 0",
        )

        if num_negatives_test is None:
            self.test_neg = self.test_data.append(self.data).iloc[:, 1].nunique()
        else:
            self.test_neg = num_negatives_test

        self.data = self.data.to_numpy()
        self.test_data = self.test_data.to_numpy()

        self.nitems = np.concatenate((self.data, self.test_data))
        self.nitems = np.max(self.nitems, axis = 0)
        self.nitems = self.nitems[0]

        self.items = self.preprocess_items(self.data)
        self.targets = self.data[:, 2]

        self.field_dims = np.max(self.items, axis=0) + 1
        self.train_mat = build_adj_mx(np.max(self.field_dims), self.items.copy())

        self.negative_sampling(num_negatives=num_negatives_train)

        self.test_set = self.build_test_set(
            self.preprocess_items(self.test_data), num_neg_samples_test=self.test_neg
        )

    def __len__(self):
        return self.targets.shape[0]

    def preprocess_items(self, data):
        reindexed_items = data[:, :2].astype(np.int)
        reindexed_items[:, 1] = reindexed_items[:, 1] + self.nitems
        return reindexed_items

    def __getitem__(self, index):
        return self.interactions[index]

    def negative_sampling(self, num_negatives):
        self.interactions = []
        data = np.c_[(self.items, self.targets)].astype(int)
        max_users, max_items = self.field_dims[:2]

        for x in tqdm(
            data, desc="Performing negative sampling on test data..."
        ):  # x are triplets (u, i , 1)
            # Append positive interaction
            self.interactions.append(x)
            # Copy user and maintain last position to 0. Now we will need to update neg_triplet[1] with j
            neg_triplet = np.vstack(
                [
                    x,
                ]
                * (num_negatives)
            )
            neg_triplet[:, 2] = np.zeros(num_negatives)

            # Generate num_negatives negative interactions
            for idx in range(num_negatives):
                j = np.random.randint(max_users, max_items)
                while (x[0], j) in self.train_mat:
                    j = np.random.randint(max_users, max_items)
                neg_triplet[:, 1][idx] = j
            self.interactions.append(neg_triplet.copy())

        self.interactions = np.vstack(self.interactions)

    def build_test_set(self, gt_test_interactions, num_neg_samples_test):
        max_users, max_items = self.field_dims[:2]
        test_set = []
        for pair in tqdm(gt_test_interactions, desc="BUILDING TEST SET..."):
            negatives = []
            for t in range(num_neg_samples_test):
                j = np.random.randint(max_users, max_items)
                while (pair[0], j) in self.train_mat or j == pair[1]:
                    j = np.random.randint(max_users, max_items)
                negatives.append(j)

            single_user_test_set = np.vstack(
                [
                    pair,
                ]
                * (len(negatives) + 1)
            )
            single_user_test_set[:, 1][1:] = negatives
            test_set.append(single_user_test_set.copy())
        return test_set
