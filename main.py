import argparse
import os
import torch.nn
from FM_GCN_model import FactorizationMachineModel_withGCN

from data_import import Dataset
from torch.utils.data import DataLoader
from train import train_epochs
from FM_model import FactorizationMachineModel
from utils import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument(
    "--log_dir", help="tensorboard log directory", type=str, default="runs"
)
parser.add_argument("--topk", help="topk values to retrieve", type=int, default=10)
parser.add_argument(
    "--dataset",
    help="dataset to use(drug_disease or protein_drug)",
    type=str,
    default="drug_disease",
)
parser.add_argument("--n_outputs", help="amount of outputs", type=int, default=10)
parser.add_argument("--epochs", help="number of epochs to train", type=int, default=5)
parser.add_argument("--batch_size", help="batch size", type=int, default=100)
parser.add_argument("--lr", help="learning rate", type=float, default=0.1)
args = parser.parse_args()

if not torch.cuda.is_available():
    raise Exception("You should enable GPU runtime")
device = torch.device("cuda")

logs_base_dir = args["log_dir"]
os.makedirs(logs_base_dir, exist_ok=True)

dataset = Dataset(type_data=args["dataset"])
criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

data_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

model = FactorizationMachineModel(dataset.field_dims[-1], 32).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

tb_fm = SummaryWriter(log_dir=f"{logs_base_dir}/{logs_base_dir}_FM/")

train_epochs(
    model,
    optimizer,
    data_loader,
    dataset,
    criterion,
    device,
    args["topk"],
    tb_fm,
    epochs=150,
)
