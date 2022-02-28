import utils
import torch
import torch.utils.data
import wandb
from statistics import mean


def train_one_epoch(model, optimizer, data_loader, criterion, device):
    model.train()
    total_loss = []

    for _, (interactions) in enumerate(data_loader):
        interactions = interactions.to(device)
        targets = interactions[:, 2]
        predictions = model(interactions[:, :2])

        loss = criterion(predictions, targets.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    return mean(total_loss)


def test(model, full_dataset, device, topk):
    # Test the HR and NDCG for the model @topK
    model.eval()

    HR, NDCG = [], []
    i = 0
    for user_test in full_dataset.test_set:
        i += 1
        gt_item = user_test[0][1]
        predictions = model.predict(user_test, device)
        _, indices = torch.topk(predictions, topk)
        indices = indices.cpu().detach().numpy()
        recommend_list = user_test[indices][:, 1]
        HR.append(utils.getHitRatio(recommend_list, gt_item))
        NDCG.append(utils.getNDCG(recommend_list, gt_item))
    return mean(HR), mean(NDCG)


def train_epochs(
    model,
    optimizer,
    data_loader,
    dataset,
    criterion,
    device,
    topk,
    tb_model,
    epochs=150,
):
    for epoch_i in range(epochs):
        train_loss = train_one_epoch(model, optimizer, data_loader, criterion, device)
        hr, ndcg = test(model, dataset, device, topk=topk)

        print(f"epoch {epoch_i}:")
        print(
            f"training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f} "
        )
        print("\n")

        dict = {
            "train/loss": train_loss,
            f"eval/HR@{topk}": hr,
            f"eval/NDCG@{topk}": ndcg,
            "epoch": epoch_i,
        }

        wandb.log(dict)
        tb_model.add_scalar_dict(dict, epoch_i)
