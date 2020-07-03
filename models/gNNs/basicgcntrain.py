import math
import os

import dgl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.gNNs.data_utils import BrainNetworkDataset
from models.gNNs.networks import BasicGCNRegressor


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    subjects, graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return np.array(subjects).reshape(len(graphs), -1), batched_graph, torch.tensor(labels).view(len(graphs), -1)


def denorm_target_f(target, dataset):
    return (target.cpu() * dataset.targets_std) + dataset.targets_mu


def str_to_bool(x):
    if x == "True":
        return True
    elif x == "False":
        return False
    else:
        raise ValueError("Expected True or False for featureless.")


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    # Dataset/Dataloader Args
    parser.add_argument("part", help="part of the brain", type=str)
    parser.add_argument("res", help="number of vertices", type=str)
    parser.add_argument("featureless", help="include features?", type=str_to_bool)
    parser.add_argument("--meta_data_file_path", help="tsv file", type=str,
                        default="/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/meta_data.tsv")
    parser.add_argument("--pickle_split_filepath", help="split file", type=str,
                        default="/vol/bitbucket/cnw119/neodeepbrain/models/gNNs/names_06152020_noCrashSubs.pk")
    parser.add_argument("--ds_max_workers", help="max_workers for building dataset", type=int, default=8)
    parser.add_argument("--dl_max_workers", help="max_workers for dataloader", type=int, default=4)
    parser.add_argument("--save_path", help="where to store data", type=str, default="../tmp")

    # Training Args
    parser.add_argument("--max_epochs", help="max epochs", type=int, default=300)
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("--lr", help="lr", type=float, default=8e-4)
    parser.add_argument("--T_max", help="T_max", type=int, default=10)
    parser.add_argument("--eta_min", help="eta_min", type=float, default=1e-6)

    # Results Args
    parser.add_argument("--results", help="where to store results", type=str, default="./results")

    args = parser.parse_args()

    if args.part == "left" or args.part == "right":
        args.load_path = f"/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native_04152020/hemispheres/reducedto_{args.res}/white/vtk"
    else:
        args.part = "merged"
        args.load_path = f"/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native_04152020/merged/reducedto_{args.res}/white/vtk"

    args.save_path = f"{args.save_path}/{args.part}_{args.res}-{'featureless' if args.featureless else 'features'}_dataset"

    args.experiment_name = f"GCN-part-{args.part}-res-{args.res}-featureless-{args.featureless}"

    args.experiment_folder = os.path.join(args.results, args.experiment_name)

    if not os.path.exists(args.experiment_folder):
        os.makedirs(args.experiment_folder)

    print("Using files from: ", args.load_path)
    print("Using: ", args.part)
    print("Data saved in: ", args.save_path)
    print("Results stored in: ", args.experiment_folder)

    return args


def get_dataloaders(args):
    train_dataset = BrainNetworkDataset(args.load_path, args.meta_data_file_path, save_path=args.save_path,
                                        max_workers=args.ds_max_workers,
                                        dataset="train", index_split_pickle_fp=args.pickle_split_filepath,
                                        part=args.part, featureless=args.featureless)

    val_dataset = BrainNetworkDataset(args.load_path, args.meta_data_file_path, save_path=args.save_path,
                                      max_workers=args.ds_max_workers,
                                      dataset="val", part=args.part, featureless=args.featureless)

    test_dataset = BrainNetworkDataset(args.load_path, args.meta_data_file_path, save_path=args.save_path,
                                       max_workers=args.ds_max_workers,
                                       dataset="test", part=args.part, featureless=args.featureless)

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                          num_workers=args.dl_max_workers)
    val_dl = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                        num_workers=args.dl_max_workers)
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                         num_workers=args.dl_max_workers)
    print("Dataloaders created")
    return train_dl, val_dl, test_dl, train_dataset, val_dataset, test_dataset


def train(model, train_dl, train_ds, loss_function, diff_func, denorm_target, optimizer, scheduler, device):
    model.train()
    train_epoch_loss = 0
    train_epoch_error = 0.
    train_epoch_worst_diff = 0.
    train_total_size = 0
    for iter, (_, bg, batch_labels) in enumerate(train_dl):
        optimizer.zero_grad()

        bg = bg.to(device)
        bg_node_features = bg.ndata["features"].to(device)
        batch_labels = batch_labels.to(device)

        prediction = model(bg, bg_node_features)
        loss = loss_function(prediction, batch_labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_diff = diff_func(denorm_target(prediction, train_ds),
                                   denorm_target(batch_labels, train_ds))  # shape: (batch_size, 1)
            train_epoch_error += train_diff.sum().detach().item()
            worst_diff = torch.max(train_diff).detach().item()
            if worst_diff > train_epoch_worst_diff:
                train_epoch_worst_diff = worst_diff
        train_epoch_loss += loss.detach().item()
        train_total_size += len(batch_labels)

    # train_epoch_loss = train_epoch_loss / (iter + 1)
    train_epoch_loss /= (iter + 1)  # Calculate mean sum batch loss over this epoch MSELoss
    train_epoch_error /= train_total_size  # Calculate mean L1 error over all the training data over this epoch

    scheduler.step()

    return train_epoch_loss, train_epoch_error, train_epoch_worst_diff


def evaluate(model, dl, ds, loss_function, diff_func, denorm_target_f, device):
    with torch.no_grad():
        model.eval()
        epoch_loss = 0
        epoch_error = 0.
        total_size = 0
        epoch_max_diff = 0.
        batch_subjects = list()
        batch_preds = list()
        batch_targets = list()
        batch_diffs = list()
        for iter, (subjects, bg, batch_labels) in enumerate(dl):
            # bg stands for batch graph
            bg = bg.to(device)
            # get node feature
            bg_node_features = bg.ndata["features"].to(device)
            batch_labels = batch_labels.to(device)

            predictions = model(bg, bg_node_features)
            loss = loss_function(predictions, batch_labels)

            diff = diff_func(denorm_target_f(predictions, ds),
                             denorm_target_f(batch_labels, ds))
            epoch_error += diff.sum().item()
            # Identify max difference
            max_diff = torch.max(diff).item()
            if max_diff > epoch_max_diff:
                epoch_max_diff = max_diff
            epoch_loss += loss.item()

            # Store
            batch_subjects.append(subjects)
            batch_preds.append(predictions.cpu())
            batch_targets.append(batch_labels.cpu())
            batch_diffs.append(diff.cpu())

            total_size += len(batch_labels)

        epoch_loss /= (iter + 1)
        epoch_error /= total_size

        all_subjects = np.concatenate(batch_subjects)
        all_preds = denorm_target_f(torch.cat(batch_preds), ds)
        all_targets = denorm_target_f(torch.cat(batch_targets), ds)
        all_diffs = torch.cat(batch_diffs)

        csv_material = np.concatenate((all_subjects, all_preds.numpy(), all_targets.numpy(), all_diffs.numpy()),
                                      axis=-1)

    return epoch_loss, epoch_error, torch.max(all_diffs).item(), csv_material


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def update_best_model(model, validation_loss, best_validation_loss, args):
    if validation_loss < best_validation_loss:
        torch.save(model, f=os.path.join(args.experiment_folder, "best_model"))
        return validation_loss
    else:
        return best_validation_loss


def update_writer(writer, train_epoch_loss, val_epoch_loss, test_epoch_loss, train_epoch_error, val_epoch_error,
                  test_epoch_error, train_epoch_max_diff, val_epoch_max_diff, test_epoch_max_diff, epoch):
    writer.add_scalar("Loss/Train", train_epoch_loss, epoch)
    writer.add_scalar("Loss/Val", val_epoch_loss, epoch)
    writer.add_scalar("Loss/Test", test_epoch_loss, epoch)
    writer.add_scalar("Error/Train", train_epoch_error, epoch)
    writer.add_scalar("Error/Val", val_epoch_error, epoch)
    writer.add_scalar("Error/Test", test_epoch_error, epoch)
    writer.add_scalar("Max Error/Train", train_epoch_max_diff, epoch)
    writer.add_scalar("Max Error/Val", val_epoch_max_diff, epoch)
    writer.add_scalar("Max Error/Test", test_epoch_max_diff, epoch)


# def convert_npfile_to_csv(fp, csv_fp):
#     ndarray = np.load(fp)
#     pd.DataFrame(ndarray).to_csv(csv_fp)


def record_csv_material(fp, data):
    if os.path.exists(fp):
        ndarray = np.load(fp)
        ndarray = np.concatenate((ndarray, data.reshape(1, *data.shape)))
    else:
        ndarray = data.reshape(1, *data.shape)
    np.save(file=fp, arr=ndarray)


if __name__ == "__main__":
    # TODO MAKE NOT HARD CODED FOR IMPERIAL

    args = get_args()

    val_log_fp = os.path.join(args.experiment_folder, "val_log")
    test_log_fp = os.path.join(args.experiment_folder, "test_log")

    train_dl, val_dl, test_dl, train_ds, val_ds, test_ds = get_dataloaders(args)

    writer = SummaryWriter(comment=f"-{args.experiment_name}")

    # Create model
    print("Creating Model")
    model = BasicGCNRegressor(3 if args.featureless else 8, 256, 1)  # 5 features in a node, 256 in the hidden, 1 output (age)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(model)
    print(f"Model is on: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("Total number of parameters: ", count_parameters(model))

    loss_function = nn.MSELoss(reduction="sum")

    diff_func = nn.L1Loss(reduction="none")

    best_val_loss = math.inf

    print("Starting")
    for epoch in range(args.max_epochs):
        # Train
        train_epoch_loss, train_epoch_error, train_epoch_max_diff = train(model, train_dl, train_ds, loss_function,
                                                                          diff_func, denorm_target_f, optimizer,
                                                                          scheduler, device)

        # Val
        val_epoch_loss, val_epoch_error, val_epoch_max_diff, val_csv_material = evaluate(model, val_dl, val_ds,
                                                                                         loss_function,
                                                                                         diff_func, denorm_target_f,
                                                                                         device)
        # Test
        test_epoch_loss, test_epoch_error, test_epoch_max_diff, test_csv_material = evaluate(model, test_dl, test_ds,
                                                                                             loss_function,
                                                                                             diff_func, denorm_target_f,
                                                                                             device)

        # Record to TensorBoard
        update_writer(writer, train_epoch_loss, val_epoch_loss, test_epoch_loss, train_epoch_error, val_epoch_error,
                      test_epoch_error, train_epoch_max_diff, val_epoch_max_diff, test_epoch_max_diff, epoch)

        # Record material to be converted to csv later
        record_csv_material(val_log_fp + ".npy", val_csv_material)
        record_csv_material(test_log_fp + ".npy", test_csv_material)

        # Save model
        update_best_model(model, val_epoch_loss, best_val_loss, args)
        torch.save(model, os.path.join(args.experiment_folder, "curr_model"))

        print('Epoch {}, train_loss {:.4f}, test_loss {:.4f}'.format(epoch, train_epoch_loss, test_epoch_loss))

    # convert_npfile_to_csv(val_log_fp, val_log_fp + ".csv")
    # convert_npfile_to_csv(test_log_fp, test_log_fp + ".csv")
