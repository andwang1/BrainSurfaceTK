import os

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch.utils.data.dataloader import DataLoader

from data_utils import BrainNetworkDataset


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.in_degrees().view(-1, 1).to(device)  # TODO Why do I have to do this here :(
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels).view(len(graphs), -1)


if __name__ == "__main__":
    # load_path = os.path.join(os.getcwd(), "data")
    load_path = os.path.join("/vol/biomedic2/aa16914/shared/MScAI_brain_surface/vtps/white/30k/left")
    # meta_data_file_path = os.path.join(os.getcwd(), "meta_data.tsv")
    meta_data_file_path = os.path.join("/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/meta_data.tsv")
    # save_path = os.path.join(os.getcwd(), "tmp", "dataset.pk")
    save_path = "/vol/bitbucket/cnw119/tmp/dataset"


    train_test_split = 0.8
    # Use PyTorch's DataLoader and the collate function
    # defined before.
    dataset = BrainNetworkDataset(load_path, meta_data_file_path, save_path, max_workers=8, 
                                    save_dataset=True, load_from_pk=True)

    # Calculate the train/test splits
    train_size = round(len(dataset) * train_test_split)
    test_size = len(dataset) - train_size
    # Split the dataset randomly
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    # Create the dataloaders for both the training and test datasets
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=True, collate_fn=collate)

    # Create model
    model = Classifier(1, 256, 1)
    loss_func = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(200):
        train_epoch_loss = 0
        model.train()
        for iter, (bg, label) in enumerate(train_dl):
            bg = bg.to(device)
            label = label.to(device)
            prediction = model(bg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.detach().item()
        train_epoch_loss /= (iter + 1)
        test_epoch_loss = 0
        model.eval()
        with torch.no_grad():
            for test_iter, (bg, label) in enumerate(test_dl):
                bg = bg.to(device)
                label = label.to(device)
                prediction = model(bg)
                loss = loss_func(prediction, label)
                test_epoch_loss += loss.detach().item()
        print('Epoch {}, train_loss {:.4f}, test_loss {:.4f}'.format(epoch, train_epoch_loss, test_epoch_loss))
