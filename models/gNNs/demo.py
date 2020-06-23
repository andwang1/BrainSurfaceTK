import os

import dgl
import torch
import torch.nn as nn
from models.gNNs.data_utils import BrainNetworkDataset
from dgl.nn.pytorch import GraphConv
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Predictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Predictor, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.predict_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, graph, features):
        # Perform graph convolution and activation function.
        hidden = self.conv1(graph, features)
        hidden = self.conv2(graph, hidden)
        with graph.local_scope():
            graph.ndata['tmp'] = hidden
            # Calculate graph representation by averaging all the node representations.
            hg = dgl.mean_nodes(graph, 'tmp')
        return self.predict_layer(hg)


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels).view(len(graphs), -1)


if __name__ == "__main__":
    # load_path = os.path.join(os.getcwd(), "data")
    load_path = os.path.join("/vol/biomedic2/aa16914/shared/MScAI_brain_surface/vtps/white/30k/left")
    # load_path ="/vol/bitbucket/cnw119/tmp_data"
    # meta_data_file_path = os.path.join(os.getcwd(), "meta_data.tsv")
    meta_data_file_path = os.path.join("/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/meta_data.tsv")
    # save_path = os.path.join(os.getcwd(), "tmp", "dataset")
    save_path = "/vol/bitbucket/cnw119/tmp/dataset"

    writer = SummaryWriter()
    batch_size = 12

    train_test_split = 0.8
    # Use PyTorch's DataLoader and the collate function
    # defined before.
    dataset = BrainNetworkDataset(load_path, meta_data_file_path, save_path=save_path, max_workers=8,
                                  save_dataset=True, load_from_pk=True)

    # Calculate the train/test splits
    train_size = round(len(dataset) * train_test_split)
    test_size = len(dataset) - train_size
    # Split the dataset randomly
    print("splitting dataset")
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    # Create the dataloaders for both the training and test datasets
    print("Building dataloaders")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

    # Create model
    print("Creating Model")
    model = Predictor(5, 256, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    print("Model made")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model is on: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(model)

    loss_func = nn.L1Loss()

    print("Starting")
    for epoch in range(200):

        # Train
        model.train()
        train_epoch_loss = 0
        for iter, (bg, label) in enumerate(train_dl):
            optimizer.zero_grad()

            bg = bg.to(device)
            bg_features = bg.ndata["features"].to(device)
            label = label.to(device)

            prediction = model(bg, bg_features)
            loss = loss_func(prediction, label)
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.detach().item()

        train_epoch_loss /= (iter + 1)

        # Test
        with torch.no_grad():

            model.eval()
            test_epoch_loss = 0
            for test_iter, (bg, label) in enumerate(test_dl):
                bg = bg.to(device)
                bg_features = bg.ndata["features"].to(device)
                label = label.to(device)
                prediction = model(bg, bg_features)
                loss = loss_func(prediction, label)
                test_epoch_loss += loss.detach().item()
            test_epoch_loss /= (test_iter + 1)

        print('Epoch {}, train_loss {:.4f}, test_loss {:.4f}'.format(epoch, train_epoch_loss, test_epoch_loss))

        # Record to TensorBoard
        writer.add_scalar("Loss/Train", train_epoch_loss)
        writer.add_scalar("Loss/Test", test_epoch_loss)
