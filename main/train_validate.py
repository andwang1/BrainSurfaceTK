import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import Module, Conv3d, ConvTranspose3d, Linear, ReLU, Sequential, Linear, Flatten, L1Loss, BatchNorm3d, \
    Dropout, BatchNorm1d
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from utils.utils import read_meta, get_file_path, zero_mean_unit_var, clean_data, display_image, split_data, get_ids_and_ages, plot_to_tensorboard
from utils.models import ImageSegmentationDataset, Part3, resample_image, PrintTensor
import os.path as osp


def save_graphs_train(fn, num_epochs, training_loss, val_loss_epoch5):


    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', f'{fn}/')


    plt.plot([epoch for epoch in range(num_epochs)], training_loss, color='b', label='Train')
    plt.plot([5 * i for i in range(len(val_loss_epoch5))], val_loss_epoch5, color='r', label='Val')
    plt.title("Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.ylim(0, 5)
    plt.xlim(-5, num_epochs + 5)
    plt.legend()
    plt.savefig(path + f'graph.png')

    plt.close()


def save_to_log(model, params, fn, final_MAE, num_epochs, batch_size, lr, feats, gamma, smoothen, edgen, dropout_p, img_spacing, img_size, scheduler_freq):

    print(f"Average Loss on whole val set: {final_MAE}")

    result = f"""
        ########################################################################
        
        *****    Score = {final_MAE}    *****

        2. Number of epochs:
        num_epochs = {num_epochs}

        Batch size during training
        batch_size = {batch_size}

        Learning rate for optimizers
        lr = {lr}

        Size of feature amplifier
        Feature Amplifier: {feats}


        Gamma (using sched)
        Gamma: {gamma}
        Frequency of step: {scheduler_freq}

        7. Image spacing and size
        img_spacing = {img_spacing}
        img_size = {img_size}

        Smooth:
        smoothen = {smoothen}

        Edgen:
        edgen = {edgen}

        Amount of dropout:
        dropout_p = {dropout_p}

        Total number of parameters is: {params}

        Model:
        {model.__str__()}
        ########################################################################
        """

    with open(f'{fn}/log.txt', 'a+') as log:
        log.write('\n')
        log.write(result)
        log.write('\n')
        torch.save(model, f'{fn}/model.pth')

    with open('all_log.txt', 'a+') as log:
        log.write('\n')
        log.write(f'SUBJECT #{fn[-1]}:    Validation = {final_MAE},    ')


def train_validate(lr, feats, num_epochs, gamma, batch_size, dropout_p, dataset_train, dataset_val, fn, number_here, scheduler_freq, writer):

    # 1. Display GPU Settings:
    cuda_dev = '0'  # GPU device 0 (can be changed if multiple GPUs are available)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")
    print('Device: ' + str(device))
    if use_cuda:
        print('GPU: ' + str(torch.cuda.get_device_name(int(cuda_dev))))

    # 2. Define loss function
    loss_function = L1Loss()

    # 3. Print parameters
    print(f"Learning Rate: {lr} and Feature Amplifier: {feats}, Num_epochs: {num_epochs}, Gamma: {gamma}")

    # 4. Define collector lists
    folds_val_scores = []
    training_loss = []
    val_loss_epoch5 = []
    i_fold_val_scores = []

    # 5. Create data loaders
    train_loader = DataLoader(dataset_train, batch_size=batch_size)
    val_loader = DataLoader(dataset_val, batch_size=batch_size)

    # 6. Define a model
    model = Part3(feats, dropout_p).to(device=device)

    # 7. Print parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params: {params}")

    # 8. Create an optimizer + LR scheduler
    optimizer = Adam(model.parameters(), lr, weight_decay=0.005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma, last_epoch=-1)

    # 9. Proceed to train
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = []
        for batch_data, batch_labels in train_loader:
            batch_labels = batch_labels.to(device=device)
            batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
            batch_preds = model(batch_data)
            loss = loss_function(batch_preds, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())


        training_MAE = np.mean(epoch_loss)
        training_loss.append(training_MAE)

        writer.add_scalar('MAE Loss/train', training_MAE, epoch)


        if epoch % scheduler_freq == 0:
            scheduler.step()

        # 10. Validate every N epochs
        if (epoch % 5 == 0):
            val_loss = []
            model.eval()
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
                    batch_labels = batch_labels.to(device=device)
                    batch_preds = model(batch_data)
                    loss = loss_function(batch_preds, batch_labels)
                    val_loss.append(loss.item())

                mean_val_error5 = np.mean(val_loss)
                val_loss_epoch5.append(mean_val_error5)
            print(f"Epoch: {epoch}:: Learning Rate: {scheduler.get_lr()[0]}")
            print(
                f"{number_here}:: Maxiumum Age Error: {np.round(np.max(epoch_loss))} Average Age Error: {training_MAE}, MAE Validation: {mean_val_error5}")

            writer.add_scalar('Max Age Error/validate', np.round(np.max(epoch_loss)), epoch)
            writer.add_scalar('MAE Loss/validate', mean_val_error5, epoch)


    # 11. Validate the last time
    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
            batch_labels = batch_labels.to(device=device)
            batch_preds = model(batch_data)
            loss = loss_function(batch_preds, batch_labels)
            i_fold_val_scores.append(loss.item())

    # 12. Summarise the results
    mean_fold_score = np.mean(i_fold_val_scores)
    val_loss_epoch5.append(mean_fold_score)
    print(f"Mean Age Error: {mean_fold_score}")

    folds_val_scores.append(mean_fold_score)

    final_MAE = np.mean(folds_val_scores)

    save_graphs_train(fn, num_epochs, training_loss, val_loss_epoch5)

    return model, params, final_MAE