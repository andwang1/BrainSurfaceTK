"""## Part C: CNN-based regression using grey matter maps

The third approach is similar in nature to the second approach in task B, but instead of using PCA for dimensionality reduction in order to use a more classical regression model, now we will use convolutional neural networks (CNNs) on the grey matter maps for predicting the subject's age directly.

You will need to implement a CNN model that takes a grey matter map as an input and maps it to a one-dimensional, real-valued output. A good starting point may be a LeNet-type architecture and adapt the last layers to convert the classification into a regression network. You should have all the necessary ingredients now from above tasks and the notebooks from the lab tutorials for how to set up a CNN model in PyTorch, how to implement a suitable training and testing routine, and how to run a two-fold cross-validation on the 500 subjects similar to tasks A and B.

*Note:* For part C, only the spatially normalised grey matter maps should be used. Similar to task A, you may want to set up a configuration for the CNN training that may also involve some resampling of the input data.
"""

# ! wget https://www.doc.ic.ac.uk/~bglocker/teaching/notebooks/brainage-data.zip
# ! unzip brainage-data.zip
# ! wget https://www.doc.ic.ac.uk/~bglocker/teaching/notebooks/meta_data_reg_test.csv
# ! pip install SimpleITK==1.2.2

########################################
# Imports for Model
########################################
from torch.nn import Module, Conv3d, ConvTranspose3d, Linear, ReLU, Sequential, Linear, Flatten, L1Loss, BatchNorm3d, Dropout, BatchNorm1d
from torch.optim import Adam, lr_scheduler
import numpy as np
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from models import ImageSegmentationDataset, Part3, resample_image, PrintTensor
from utils import read_meta, split_data, get_file_path, zero_mean_unit_var, clean_data, display_image
########################################
# End Imports
########################################


"""# Dataset initialisation"""

categories = {'gender': 3, 'birth_age': 4, 'weight': 5, 'scan_age': 7, 'scan_num': 8}
meta_column_idx = categories['scan_age']

# 1. Read the data
meta_data = read_meta()

# Cleaning data. Dropping rows, that we don't have files for.
meta_data = clean_data(meta_data)

# 3. Iterate through all patient ids
ids = []
ages = []
for idx, patient_id in enumerate(meta_data[:, 1]):
    session_id = meta_data[idx, 2]
    file_path = get_file_path(patient_id, session_id)
    if os.path.isfile(file_path):
        ids.append((patient_id, session_id))
        ages.append(float(meta_data[idx, meta_column_idx]))


training_size = 0.50
smoothen = 5
edgen = False
test_size = 0.09
val_size = 0.1
random_state = 42

for smoothen in [0, 2, 4, 6, 8]:

# meta_data_reg_train = pd.read_csv(data_dir + 'meta/meta_data_reg_train.csv')
# ids = meta_data_reg_train['subject_id'].tolist()
# ages = meta_data_reg_train['age'].tolist()
# X_fold1, X_fold2, y_fold1, y_fold2 = train_test_split(ids, ages, train_size=0.5, random_state=42)

_, bins = np.histogram(meta_data[:, meta_column_idx].astype(float), bins='doane')
y_binned = np.digitize(meta_data[:, meta_column_idx].astype(float), bins)

X_train, X_test, y_train, y_test = train_test_split(ids, ages,
                                                    test_size=test_size,
                                                    random_state=random_state,
                                                    stratify=y_binned)


if val_size > 0:
    _, bins = np.histogram(np.array(y_train).astype(float), bins='doane')
    y_binned = np.digitize(np.array(y_train).astype(float), bins)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                       test_size=val_size,
                                                       random_state=random_state,
                                                       stratify=y_binned)


# ImageSegmentationDataset
dataset_train = ImageSegmentationDataset(X_train, y_train, smoothen, edgen)
dataset_val = ImageSegmentationDataset(X_val, y_val, smoothen, edgen)


########################################
# User Parameters:
########################################
# Percentage Training Size (%)
USE_GPU = True
dtype = torch.float32
feats = 5
num_epochs = 200
lr = 0.006882801723742766
gamma = 0.97958263796472
batch_size = 32
dropout_p = 0.5
sns.set(style='darkgrid')
########################################
# End User Parameters
########################################




# Display GPU Settings:
cuda_dev = '1' #GPU device 0 (can be changed if multiple GPUs are available)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")
print('Device: ' + str(device))
if use_cuda:
    print('GPU: ' + str(torch.cuda.get_device_name(int(cuda_dev))))

print("Creating Subject Folder")
number_here = 0
while True:
    fn = f'Subject_{number_here}'
    if not os.path.exists(fn):
        print(f"Making {number_here}")
        os.makedirs(fn)
        with open(f'{fn}/log.txt', 'w+') as log:
            log.write('\n')
        break
    else:
        print(f"Subject_{number_here} exists")
        number_here += 1
print("Created Subject Folder")

loss_function = L1Loss()

print(f"Learning Rate: {lr} and Feature Amplifier: {feats}, Num_epochs: {num_epochs}, Gamma: {gamma}")
folds_val_scores = []

for i in [0]:#, 1]:
    training_loss = []
    val_loss_epoch5 = []
    i_fold_val_scores = []
    if i == 0:
        train_loader = DataLoader(dataset_train, batch_size=batch_size)
        val_loader = DataLoader(dataset_val, batch_size=batch_size)
    else:
        train_loader = DataLoader(dataset_val, batch_size=batch_size)
        val_loader = DataLoader(dataset_train, batch_size=batch_size)

    model = Part3(feats, dropout_p).to(device=device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params: {params}")

    optimizer = Adam(model.parameters(), lr, weight_decay=0.005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma, last_epoch=-1)

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

        scheduler.step()

        if (epoch%5==0):
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
            print(f"{number_here}::{i} Maxiumum Age Error: {np.round(np.max(epoch_loss))} Average Age Error: {training_MAE}, MAE Validation: {mean_val_error5}")

    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
            batch_labels = batch_labels.to(device=device)
            batch_preds = model(batch_data)
            loss = loss_function(batch_preds, batch_labels)
            i_fold_val_scores.append(loss.item())

    mean_fold_score = np.mean(i_fold_val_scores)
    val_loss_epoch5.append(mean_fold_score)
    print(f"Mean Age Error: {mean_fold_score}")

    folds_val_scores.append(mean_fold_score)

    plt.plot([epoch for epoch in range(num_epochs)], training_loss, color='b', label='Train')
    plt.plot([5*i for i in range(len(val_loss_epoch5))], val_loss_epoch5, color='r', label='Val')
    plt.title("Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.ylim(0, 30)
    plt.xlim(-5, num_epochs+5)
    plt.legend()
    plt.savefig(f'{fn}/graph_{i}.png')
    plt.close()

    if i == 0:
        train_0 = training_loss
        val_0 = val_loss_epoch5
    else:
        train_1 = training_loss
        val_1 = val_loss_epoch5

final_MAE = np.mean(folds_val_scores)
print(f"Average Loss on whole val set: {final_MAE}")

result = f"""
########################################################################
# Score = {final_MAE}

# Number of epochs:
num_epochs = {num_epochs}

# Batch size during training
batch_size = {batch_size}

# Learning rate for optimizers
lr = {lr}

# Size of feature amplifier
Feature Amplifier: {feats}


# Gamma (using sched)
Gamma: {gamma}

# Smooth:
smoothen = {smoothen}

# Edgen:
edgen = {edgen}

# Amount of dropout:
dropout_p = {dropout_p}

Total number of parameters is: {params}

# Model:
{model.__str__()}
########################################################################
"""

with open(f'{fn}/log.txt', 'a+') as log:
    log.write('\n')
    log.write(result)
    log.write('\n')
    torch.save(model, f'{fn}/model.pth')


plt.plot([epoch for epoch in range(num_epochs)], train_0, color='b', label='Train-0')
plt.plot([5*i for i in range(len(val_0))], val_0, color='r', label='Val-0')
# plt.plot([epoch for epoch in range(num_epochs)], train_1, '--', color='b', label='Train-1')
# plt.plot([5*i for i in range(len(val_1))], val_1, '-', color='r', label='Val-1')
plt.title("Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.ylim(0, 40)
plt.xlim(-5, num_epochs+5)
plt.legend()
plt.savefig(f'{fn}/cv_graph.png')
plt.close()

with open(f'{fn}/log.txt', 'a+') as log:
    log.write('\n')
    log.write(result)
    log.write('\n')
    torch.save(model, f'{fn}/model.pth')




"""# Full Train & Final Test"""

########################################
# User Parameters:
########################################
USE_GPU = True
dtype = torch.float32

smoothen = 8
edgen = False
feats = 5
num_epochs = 200
lr = 0.006882801723742766
gamma = 0.97958263796472
batch_size = 32
dropout_p = 0.5

########################################
# End User Parameters
########################################

print('='*60)
print('='*60)
print('------- TESTING STAGE -------')
print('='*60)
print('='*60)

_, bins = np.histogram(meta_data[:, meta_column_idx].astype(float), bins='doane')
y_binned = np.digitize(meta_data[:, meta_column_idx].astype(float), bins)
X_train, X_test, y_train, y_test = train_test_split(ids, ages,
                                                    test_size=test_size,
                                                    random_state=random_state,
                                                    stratify=y_binned)

# ImageSegmentationDataset
train_ds = ImageSegmentationDataset(X_train, y_train, smoothen, edgen)
test_ds = ImageSegmentationDataset(X_test, y_test, smoothen, edgen)

# Display GPU Settings:
cuda_dev = '1' #GPU device 0 (can be changed if multiple GPUs are available)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")
print('Device: ' + str(device))
if use_cuda:
    print('GPU: ' + str(torch.cuda.get_device_name(int(cuda_dev))))

print("Creating Subject Folder")
number_here = 0
while True:
    fn = f'Test_{number_here}'
    if not os.path.exists(fn):
        print(f"Making {number_here}")
        os.makedirs(fn)
        with open(f'{fn}/log.txt', 'w+') as log:
            log.write('\n')
        break
    else:
        print(f"Test_{number_here} exists")
        number_here += 1
print("Created Subject Folder")

loss_function = L1Loss()
train_loader = DataLoader(train_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)
print(f"Learning Rate: {lr} and Feature Amplifier: {feats}, Num_epochs: {num_epochs}, Gamma: {gamma}")

training_loss = []
test_loss_epoch5 = []

model = Part3(feats, dropout_p).to(device=device)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Params: {params}")

optimizer = Adam(model.parameters(), lr, weight_decay=0.005)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma, last_epoch=-1)

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

    scheduler.step()

    if (epoch%5==0):
        test_loss = []
        model.eval()
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
                batch_labels = batch_labels.to(device=device)
                batch_preds = model(batch_data)
                loss = loss_function(batch_preds, batch_labels)
                test_loss.append(loss.item())
            mean_test_error5 = np.mean(test_loss)
            test_loss_epoch5.append(mean_test_error5)
        print(f"Epoch: {epoch}:: Learning Rate: {scheduler.get_lr()[0]}")
        print(f"{number_here}:: Maxiumum Age Error: {np.round(np.max(epoch_loss))} Average Age Error: {training_MAE}, MAE Test: {mean_test_error5}")

model.eval()
test_scores = []
with torch.no_grad():
    for batch_data, batch_labels in test_loader:
        batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
        batch_labels = batch_labels.to(device=device)
        batch_preds = model(batch_data)
        loss = loss_function(batch_preds, batch_labels)
        test_scores.append(loss.item())

score = np.mean(test_scores)
test_loss_epoch5.append(score)
print(f"Mean Age Error: {score}")

plt.plot([epoch for epoch in range(num_epochs)], training_loss, color='b', label='Train')
plt.plot([5*i for i in range(len(test_loss_epoch5))], test_loss_epoch5, color='r', label='Test')
plt.title("Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.ylim(0, 30)
plt.xlim(-5, num_epochs+5)
plt.legend()
plt.savefig(f'{fn}/test_loss_graph.png')
plt.close()

print(f"Average Loss on whole test set: {score}")

result = f"""
########################################################################
# Score = {score}

# Number of epochs:
num_epochs = {num_epochs}

# Batch size during training
batch_size = {batch_size}

# Learning rate for optimizers
lr = {lr}

# Size of feature amplifier
Feature Amplifier: {feats}

# Gamma (using sched)
Gamma: {gamma}

# Smooth:
smoothen = {smoothen}

# Edgen:
edgen = {edgen}

# Amount of dropout:
dropout_p = {dropout_p}

Total number of parameters is: {params}

# Model:
{model.__str__()}
########################################################################
"""

with open(f'{fn}/test_log.txt', 'a+') as log:
    log.write('\n')
    log.write(result)
    log.write('\n')
    torch.save(model, f'{fn}/test_model.pth')

model.eval()
pred_ages = []
actual_ages = []
with torch.no_grad():
    # for batch_data, batch_labels in train_loader:
    #     batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
    #     batch_labels = batch_labels.to(device=device)
    #     batch_preds = model(batch_data)
    #     pred_ages.append([batch_preds[i].item() for i in range(len(batch_preds))])
    #     actual_ages.append([batch_labels[i].item() for i in range(len(batch_labels))])

    for batch_data, batch_labels in test_loader:
        batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
        batch_labels = batch_labels.to(device=device)
        batch_preds = model(batch_data)
        pred_ages.append([batch_preds[i].item() for i in range(len(batch_preds))])
        actual_ages.append([batch_labels[i].item() for i in range(len(batch_labels))])

pred_ages = np.array(pred_ages).flatten()
actual_ages = np.array(actual_ages).flatten()

pred_array = []
age_array = []
for i in range(len(pred_ages)):
    for j in range(len(pred_ages[i])):
        pred_array.append(pred_ages[i][j])
        age_array.append(actual_ages[i][j])

y = age_array
predicted = pred_array

fig, ax = plt.subplots()
ax.scatter(y, predicted, marker='.')
ax.plot([min(y), max(y)], [min(y), max(y)], 'k--', lw=2)
ax.set_xlabel('Real Age')
ax.set_ylabel('Predicted Age')
plt.savefig(f'{fn}/scatter_part_c.png')
plt.close()
