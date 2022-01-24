import torch
if not torch.cuda.is_available():
    raise Exception("You should enable GPU runtime")
device = torch.device("cuda")
import pandas as pd
import numpy as np
from utils import build_adj_mx
from Dataset_movielens import MovieLens100kDataset
from torch.utils.data import DataLoader, Dataset
from statistics import mean
from model import  FeaturesLinear, FM_operation, FactorizationMachineModel
from utils import getHitRatio, getNDCG


# LOAD TRAINING DATA
colnames = ["user_id", 'item_id', 'label', 'timestamp']
data = pd.read_csv('data_movielens/ml-100k/movielens.train.rating', sep="\t", header=None, names=colnames)

# LOAD TESTING DATA
colnames = ["user_id", 'item_id', 'label', 'timestamp']
test_data = pd.read_csv('data_movielens/ml-100k/movielens.test.rating', sep="\t", header=None, names=colnames)

# Pre-process
## userId,movieId,rating,timestamp
data = data.to_numpy()
items = data[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
reindex_items = items.copy()
reindex_items[:, 1] = reindex_items[:, 1] + 943
field_dims = np.max(reindex_items, axis=0) + 1
train_mat = build_adj_mx(field_dims[-1], reindex_items.copy())
targets = data[:, 2]

# Build test Dataset
dataset_path = 'data_movielens/ml-100k/movielens'
test_data = pd.read_csv(f'{dataset_path}.test.rating', sep='\t',
                        header=None, names=colnames).to_numpy()

# Take number of users and items from reindex items from train set
users, items = np.max(reindex_items, axis=0)[:2] + 1 # [ 943, 1682])

# Reindex test items and substract 1
pairs_test = test_data[:, :2].astype(np.int) - 1    
pairs_test[:, 1] = pairs_test[:, 1] + users

pair = pairs_test[0]
# GENERATE TEST SET WITH NEGATIVE EXAMPLES TO EVALUATE
max_users, max_items = field_dims[:2] # number users (943), number items (2625)
negatives = []
for t in range(10):
    j = np.random.randint(max_users, max_items)
    while (pair[0], j) in train_mat or j == pair[1]:
        j = np.random.randint(max_users, max_items)
    negatives.append(j)
    
single_user_test_set = np.vstack([pair, ] * (len(negatives)+1))
single_user_test_set[:, 1][1:] = negatives

# Building dataset

full_dataset= MovieLens100kDataset(dataset_path, num_negatives_train=4, num_negatives_test=99)
data_loader = DataLoader(full_dataset, batch_size=256, shuffle=True, num_workers=0)

# Train

def train_one_epoch(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = []

    for i, (interactions) in enumerate(data_loader):
        interactions = interactions.to(device)
        targets = interactions[:,2]
        predictions = model(interactions[:,:2])
        
        loss = criterion(predictions, targets.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    return mean(total_loss)

user_test = full_dataset.test_set[0]
# Defining dummy model with 8 embedding dimensions
dummy_model = FactorizationMachineModel(full_dataset.field_dims[-1], 8).to(device)
out = dummy_model.predict(user_test, device)

def test(model, full_dataset, device, topk=10):
    # Test the HR and NDCG for the model @topK
    model.eval()

    HR, NDCG = [], []

    for user_test in full_dataset.test_set:
        gt_item = user_test[0][1]

        predictions = model.predict(user_test, device)
        _, indices = torch.topk(predictions, topk)
        recommend_list = user_test[indices.cpu().detach().numpy()][:, 1]

        HR.append(getHitRatio(recommend_list, gt_item))
        NDCG.append(getNDCG(recommend_list, gt_item))
    return mean(HR), mean(NDCG)

model = FactorizationMachineModel(full_dataset.field_dims[-1], 32).to(device)
criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
topk = 10

# DO EPOCHS NOW
tb = True
topk = 10
for epoch_i in range(20):
    #data_loader.dataset.negative_sampling()
    train_loss = train_one_epoch(model, optimizer, data_loader, criterion, device)
    hr, ndcg = test(model, full_dataset, device, topk=topk)

    print('\n')

    print(f'epoch {epoch_i}:')
    print(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f} ')
    print('\n')
    # if tb:
    #     tb_fm.add_scalar('train/loss', train_loss, epoch_i)
    #     tb_fm.add_scalar('eval/HR@{topk}', hr, epoch_i)
    #     tb_fm.add_scalar('eval/NDCG@{topk}', ndcg, epoch_i)

