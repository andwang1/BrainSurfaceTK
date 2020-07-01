import os

import dgl
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.gNNs.data_utils import BrainNetworkDataset
from models.gNNs.networks import BasicGCNRegressor


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels).view(len(graphs), -1)


def denorm_target(target, dataset):
    return (target.cpu() * dataset.targets_std) + dataset.targets_mu


if __name__ == "__main__":

    # # Local
    # load_path = os.path.join(os.getcwd(), "data")
    # pickle_split_filepath = os.path.join(os.getcwd(), "names_04152020_noCrashSubs.pk")
    # meta_data_file_path = os.path.join(os.getcwd(), "meta_data.tsv")
    # save_path = os.path.join(os.getcwd(), "tmp", "dataset")

    # Imperial
    load_path = "/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native_04152020/hemispheres/reducedto_10k/white/vtk"
    pickle_split_filepath = "/vol/bitbucket/cnw119/neodeepbrains/models/gNNs/names_04152020_noCrashSubs.pk"
    meta_data_file_path = os.path.join("/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/meta_data.tsv")
    save_path = "/vol/bitbucket/cnw119/tmp/basicdataset"

    lr = 8e-4
    T_max = 10
    eta_min = 1e-6

    writer = SummaryWriter(comment="basicgcn")
    batch_size = 64

    train_dataset = BrainNetworkDataset(load_path, meta_data_file_path, save_path=save_path, max_workers=8,
                                        dataset="train", index_split_pickle_fp=pickle_split_filepath)

    val_dataset = BrainNetworkDataset(load_path, meta_data_file_path, save_path=save_path, max_workers=8,
                                      dataset="val")

    test_dataset = BrainNetworkDataset(load_path, meta_data_file_path, save_path=save_path, max_workers=8,
                                       dataset="test")

    print("Building dataloaders")
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=8)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=8)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=8)

    # Create model
    print("Creating Model")
    # model = BasicGCN(5, 256, 1)
    model = BasicGCNRegressor(8, 256, 1)  # 5 features in a node, 256 in the hidden, 1 output (age)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    print("Model made")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model is on: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(model)

    loss_function = nn.MSELoss(reduction="sum")

    diff_func = nn.L1Loss(reduction="none")

    print("Starting")
    for epoch in range(1000):

        # Train
        model.train()
        train_epoch_loss = 0
        train_epoch_error = 0.
        train_epoch_worst_diff = 0.
        train_total_size = 0
        for iter, (bg, batch_labels) in enumerate(train_dl):
            optimizer.zero_grad()

            bg = bg.to(device)
            bg_node_features = bg.ndata["features"].to(device)
            batch_labels = batch_labels.to(device)

            prediction = model(bg, bg_node_features)
            loss = loss_function(prediction, batch_labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_diff = diff_func(denorm_target(prediction, train_dataset),
                                       denorm_target(batch_labels, train_dataset))  # shape: (batch_size, 1)
                train_epoch_error += train_diff.sum().detach().item()
                worst_diff = torch.max(train_diff).detach().item()
                if worst_diff > train_epoch_worst_diff:
                    train_epoch_worst_diff = worst_diff
            train_epoch_loss += loss.detach().item()
            train_total_size += len(batch_labels)

        # train_epoch_loss = train_epoch_loss / (iter + 1)
        train_epoch_loss /= (iter + 1)  # Calculate mean sum batch loss over this epoch MSELoss
        train_epoch_error /= train_total_size  # Calculate mean L1 error over all the training data over this epoch

        # Val
        with torch.no_grad():

            model.eval()
            val_epoch_loss = 0
            val_epoch_error = 0.
            val_total_size = 0
            val_epoch_worst_diff = 0.
            for val_iter, (bg, batch_labels) in enumerate(val_dl):
                # bg stands for batch graph
                bg = bg.to(device)
                # get node feature
                bg_node_features = bg.ndata["features"].to(device)
                batch_labels = batch_labels.to(device)

                prediction = model(bg, bg_node_features)
                loss = loss_function(prediction, batch_labels)

                val_diff = diff_func(denorm_target(prediction, val_dataset),
                                     denorm_target(batch_labels, val_dataset))
                val_epoch_error += val_diff.sum().detach().item()
                worst_diff = torch.max(val_diff).detach().item()
                if worst_diff > val_epoch_worst_diff:
                    val_epoch_worst_diff = worst_diff
                val_epoch_loss += loss.item()

                val_total_size += len(batch_labels)

            val_epoch_loss /= (val_iter + 1)
            val_epoch_error /= val_total_size

        # Test
        with torch.no_grad():

            model.eval()
            test_epoch_loss = 0
            test_epoch_error = 0.
            test_total_size = 0
            test_epoch_worst_diff = 0.
            for test_iter, (bg, batch_labels) in enumerate(test_dl):
                # bg stands for batch graph
                bg = bg.to(device)
                # get node feature
                bg_node_features = bg.ndata["features"].to(device)
                batch_labels = batch_labels.to(device)

                prediction = model(bg, bg_node_features)
                loss = loss_function(prediction, batch_labels)

                test_diff = diff_func(denorm_target(prediction, test_dataset),
                                      denorm_target(batch_labels, test_dataset))
                test_epoch_error += test_diff.sum().detach().item()
                worst_diff = torch.max(test_diff).detach().item()
                if worst_diff > test_epoch_worst_diff:
                    test_epoch_worst_diff = worst_diff
                test_epoch_loss += loss.detach().item()

                test_total_size += len(batch_labels)

            test_epoch_loss /= (test_iter + 1)
            test_epoch_error /= test_total_size

        print('Epoch {}, train_loss {:.4f}, test_loss {:.4f}'.format(epoch, train_epoch_loss, test_epoch_loss))

        # Record to TensorBoard
        writer.add_scalar("Loss/Train", train_epoch_loss, epoch)
        writer.add_scalar("Loss/Val", val_epoch_loss, epoch)
        writer.add_scalar("Loss/Test", test_epoch_loss, epoch)
        writer.add_scalar("Error/Train", train_epoch_error, epoch)
        writer.add_scalar("Error/Val", val_epoch_error, epoch)
        writer.add_scalar("Error/Test", test_epoch_error, epoch)
        writer.add_scalar("Max Error/Train", train_epoch_worst_diff, epoch)
        writer.add_scalar("Max Error/Val", val_epoch_worst_diff, epoch)
        writer.add_scalar("Max Error/Test", test_epoch_worst_diff, epoch)
