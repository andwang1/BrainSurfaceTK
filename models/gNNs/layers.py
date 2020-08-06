import dgl.function as fn
import torch
from torch import nn


class GNNLayer(nn.Module):
    def __init__(self, node_dims, edge_dims, output_dims, activation):
        super(GNNLayer, self).__init__()
        self.W_msg = nn.Linear(node_dims + edge_dims, output_dims)
        self.W_apply = nn.Linear(output_dims + node_dims, output_dims)
        self.activation = activation

    def message_func(self, edges):
        return {'m': self.activation(self.W_msg(torch.cat([edges.src['h'], edges.data['h']], -1)))}

    def forward(self, g, node_features, edge_features):
        with g.local_scope():
            g.ndata['h'] = node_features
            g.edata['h'] = edge_features
            g.update_all(self.message_func, fn.sum('m', 'h_neigh'))
            g.ndata['h'] = self.activation(self.W_apply(torch.cat([g.ndata['h'], g.ndata['h_neigh']], dim=-1)))
            return g.ndata['h']
