import argparse
import os
import torch.nn
import torch.optim
import wandb

from torch_geometric.utils import from_scipy_sparse_matrix
from utils.FM_GCN_model import (
    FactorizationMachineModel_withGCN,
    sparse_mx_to_torch_sparse_tensor,
)
from scipy.sparse import identity
from utils.data_import import Dataset
from torch.utils.data import DataLoader
from utils.train import train_epochs
from utils.FM_model import FactorizationMachineModel
from utils.side import SummaryWriter, save_model

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", help="choose model(FM;FM_GCN;FM_GCNwAT)", type=str, default="FM"
)
parser.add_argument(
    "--data_folder", help="data folder", type=str, default="data"
)
parser.add_argument(
    "--dataset",
    help="dataset to use(drug_disease or protein_drug)",
    type=str,
    default="drug_disease",
)
parser.add_argument("--epochs", help="number of epochs to train", type=int, default=150)
parser.add_argument(
    "--embed_dims", help="amount of embedding dimensions", type=int, default=32
)
parser.add_argument("--batch_size", help="batch size", type=int, default=256)
parser.add_argument("--lr", help="learning rate", type=float, default=0.01)
parser.add_argument("--topk", help="topk values to retrieve", type=int, default=10)
parser.add_argument("--heads", help="heads to be used when using attention layers", type=int, default=8)
parser.add_argument(
    "--log_dir", help="tensorboard log directory", type=str, default="runs"
)
parser.add_argument("--wandb_run", help="WandB run name", type=str, default="runs")
parser.add_argument(
    "--wandb_project", help="WandB project", type=str, default="project"
)
args = parser.parse_args()

if not torch.cuda.is_available():
    raise Exception("You should enable GPU runtime")
device = torch.device("cuda")

logs_base_dir = args.log_dir
os.makedirs(logs_base_dir, exist_ok=True)
wandb.init(project=args.wandb_project)
wandb.run.name = args.wandb_run
dataset = Dataset(type_data=args.dataset, data_path=args.data_folder)

criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

data_loader = DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
)

if args.model == "FM":
    model = FactorizationMachineModel(
        dataset.field_dims[-1], embed_dim=args.embed_dims
    ).to(device)
elif args.model in ["FM_GCN", "FM_GCNwAT"]:
    X = sparse_mx_to_torch_sparse_tensor(identity(dataset.train_mat.shape[0]))
    edge_idx, edge_attr = from_scipy_sparse_matrix(dataset.train_mat)
    model = FactorizationMachineModel_withGCN(
        dataset.field_dims[-1],
        args.embed_dims,
        X.to(device),
        edge_idx.to(device),
        heads= args.heads,
        attention=args.model == "FM_GCNwAT",
    ).to(device)
else:
    raise Exception(
        "Wrong model provided. Available models are: FM, FM_GCN or FM_GCNwAT"
    )

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

writer = SummaryWriter(log_dir=f"{logs_base_dir}/{logs_base_dir}_{args.model}")

if __name__ == "__main__":

    print(f"Starting {args.model} trainning.\n")

    model = train_epochs(
        model,
        optimizer,
        data_loader,
        dataset,
        criterion,
        device,
        args.topk,
        args.model,
        writer,
        epochs=args.epochs,
    )

    save_model(model, args.model + ".model")
