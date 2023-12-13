import ipdb

import torch 
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv
import torch.nn.functional as F

class GCN(torch.nn.Module):

    layer_type_lookup = {
            'gcn-conv' : GCNConv, 
            'gcn-sage' : SAGEConv,
            'gcn-gat' : GATConv,
            # 'gcn-gin' : GINConv,
            'gcn-transformer' : TransformerConv,
        }

    def __init__(self, layer_type_name, input_dim, hidden_dim, output_dim, num_layers,
                 dropout=0.2, return_embeds=False):
        super(GCN, self).__init__()
        
        if layer_type_name not in self.layer_type_lookup.keys():
            raise ValueError(f"Layer type name [{layer_type_name}] not valid ")
        layer_type = self.layer_type_lookup[layer_type_name]

        self.num_layers = num_layers
        self.convs = [ layer_type(input_dim, hidden_dim) ]
        for i in range(num_layers-2):
          self.convs.append(layer_type(hidden_dim, hidden_dim))
        self.convs.append(layer_type(hidden_dim, output_dim))
        self.convs = torch.nn.ModuleList(self.convs)
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers-1)])
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.dropout = dropout
        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        out = x
        for i in range(self.num_layers-1):
          out = self.convs[i](out, edge_index)
          out = self.bns[i](out)
          out = F.relu(out)
          if self.training:
            out = F.dropout(out, p=self.dropout)

        out = self.convs[-1](out, edge_index)
        return out

