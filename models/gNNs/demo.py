import os
from tqdm import tqdm
import dgl
import numpy as np
import pyvista as pv
import torch
from dgl.nn.pytorch import GraphConv
from models.gNNs.data_utils import BrainNetworkDataset
from torch.utils.data.dataloader import DataLoader


import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.in_degrees().view(-1, 1).to(device)#.float()
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
    load_path = os.path.join(os.getcwd(), "models", "gNNs", "data")
    # Use PyTorch's DataLoader and the collate function
    # defined before.
    dataset = BrainNetworkDataset(load_path, max_workers=8)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate)

    # Create model
    model = Classifier(1, 256, 1)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    epoch_losses = []
    for epoch in range(80):
        epoch_loss = 0
        for iter, (bg, label) in enumerate(data_loader):
            bg = bg.to(device)
            label = label.to(device)
            prediction = model(bg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)

