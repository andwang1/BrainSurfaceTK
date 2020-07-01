import math
import os

import dgl
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.gNNs.data_utils import BrainNetworkDataset
from models.gNNs.networks import BasicGCNSegmentation


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels).view(len(graphs), -1)


def denorm_target(target, dataset):
    return (target.cpu() * dataset.targets_std) + dataset.targets_mu


def loss_function(predicts, targets):
    neg = targets == 0
    weight = (neg.sum(dim=0) - targets.sum(dim=0)) / targets.sum(dim=0)
    return torch.nn.functional.cross_entropy(predicts, targets, weight=weight.float())


if __name__ == "__main__":

    # # Local
    # load_path = os.path.join(os.getcwd(), "data")
    # meta_data_file_path = os.path.join(os.getcwd(), "meta_data.tsv")
    # save_path = os.path.join(os.getcwd(), "tmp", "dataset")

    # Imperial
    load_path = os.path.join("/vol/biomedic2/aa16914/shared/MScAI_brain_surface/vtps/white/30k/left")
    meta_data_file_path = os.path.join("/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/meta_data.tsv")
    save_path = "/vol/bitbucket/cnw119/tmp/dataset"

    lr = 8e-4
    T_max = 10
    eta_min = 1e-6

    writer = SummaryWriter(comment="segmentationbasicgcn")
    batch_size = 12
    train_test_split = 0.8

    train_dataset = BrainNetworkDataset(load_path, meta_data_file_path, save_path=save_path, max_workers=8,
                                        dataset="train", train_split_per=train_test_split)

    test_dataset = BrainNetworkDataset(load_path, meta_data_file_path, save_path=save_path, max_workers=8,
                                       dataset="test")

    print("Building dataloaders")
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=8)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=8)

    # Create model
    print("Creating Model")
    # model = BasicGCN(5, 256, 1)
    model = BasicGCNSegmentation(5, 256, 40)  # 5 features, 40 outputs (segmentation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    print("Model made")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model is on: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(model)

    best_loss = math.inf

    print("Starting")
    for epoch in range(1000):

        # Train
        model.train()
        train_epoch_loss = 0
        train_epoch_error = 0.
        train_epoch_worst_diff = 0.
        train_total_size = 0
        for iter, (bg, _) in enumerate(train_dl):
            optimizer.zero_grad()

            bg = bg.to(device)
            bg_node_features = bg.ndata["features"].to(device)
            batch_labels = bg.ndata["segmentation"].to(device)  # TEMP FIX WITH FLOAT HERE

            prediction = model(bg, bg_node_features)
            loss = loss_function(prediction, batch_labels)
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()

        # train_epoch_loss = train_epoch_loss / (iter + 1)
        train_epoch_loss /= (iter + 1)  # Calculate mean batch loss over this epoch

        # Test
        with torch.no_grad():

            model.eval()
            test_epoch_loss = 0
            test_epoch_error = 0.
            test_total_size = 0
            test_epoch_worst_diff = 0.
            for test_iter, (bg, _) in enumerate(test_dl):
                # bg stands for batch graph
                bg = bg.to(device)
                # get node feature
                bg_node_features = bg.ndata["features"].to(device)
                # get segmentation labels
                batch_labels = bg.ndata["segmentation"].to(device)

                prediction = model(bg, bg_node_features)
                loss = loss_function(prediction, batch_labels)

                test_epoch_loss += loss.item()

            test_epoch_loss /= (test_iter + 1)

        print('Epoch {}, train_loss {:.4f}, test_loss {:.4f}'.format(epoch, train_epoch_loss, test_epoch_loss))

        # Record to TensorBoard
        writer.add_scalar("Loss/Train", train_epoch_loss, epoch)
        writer.add_scalar("Loss/Test", test_epoch_loss, epoch)
