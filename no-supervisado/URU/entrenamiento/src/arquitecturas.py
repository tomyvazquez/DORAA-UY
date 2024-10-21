import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU, LogSoftmax, BatchNorm1d, LeakyReLU
from torch_geometric.nn import GCNConv, TAGConv
from torch_geometric.nn.norm import BatchNorm

def non_linearity(U, min, max):
  ''' U is the output of the GNN, BxNx3 (Qgen, V, angle)
      a and b Nx3 are the upper and lower limits for the three magnitudes '''
  a_batch = min.repeat(U.shape[0],1,1)
  b_batch = max.repeat(U.shape[0],1,1)

  gamma = a_batch + (b_batch-a_batch)*torch.sigmoid(U/10)
  return gamma

class GNNUnsupervised(nn.Module):
    def __init__(self, dim, edge_index, Y_bus, num_layers, K,val_min,val_max, num_nodes,dropout,batch_norm=True):
        super(GNNUnsupervised, self).__init__()
        self.val_min = val_min
        self.val_max = val_max
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.batchnorm = torch.nn.ModuleList()
        self.batch_norm = batch_norm
        self.num_nodes = num_nodes
        self.dim = dim
        self.dropout = nn.Dropout(dropout)

        for layer in range(num_layers):
          self.convs.append(TAGConv(dim[layer],dim[layer+1],K[layer],bias=True))
          self.batchnorm.append(BatchNorm(num_nodes*dim[layer+1]))
        # Initialize weights
        # for name, params in self.named_parameters():
        #     if 'lins' in name:
        #         print(params)
        #         nn.init.xavier_uniform_(params, gain=nn.init.calculate_gain('relu'))

        self.edge_index = edge_index
        self.edge_weights = torch.abs(Y_bus)
        self.relu = LeakyReLU()


    def forward(self, x):

        # Apply the GNN to the node features
        num_layers = self.num_layers
        convs = self.convs
        relu = self.relu
        edge_index = self.edge_index
        edge_weights = self.edge_weights

        out = x
        for layer in range(num_layers-1):
          out = convs[layer](out,edge_index)
          if self.batch_norm:
             out = out.view(-1, self.num_nodes * self.dim[layer + 1])
             out = self.batchnorm[layer](out)
             out = out.view(-1, self.num_nodes, self.dim[layer + 1])
          out = relu(out)
          out = self.dropout(out)

        out = convs[-1](out,edge_index)
        out = non_linearity(out,self.val_min,self.val_max)
        return out
      
      


class FCNNUnsupervised(nn.Module):
    def __init__(self, dim, edge_index, Y_bus, num_layers, K,val_min,val_max, num_nodes,dropout, batch_norm=True):
        super(FCNNUnsupervised, self).__init__()
        self.val_min = val_min
        self.val_max = val_max
        self.num_layers = num_layers
        self.linears = torch.nn.ModuleList()
        self.batchnorm = torch.nn.ModuleList()
        self.batch_norm = batch_norm
        self.num_nodes = num_nodes
        self.dim = dim
        self.dropout = nn.Dropout(dropout)

        
        for layer in range(num_layers-1):
          if layer == 0:
            self.linears.append(nn.Linear(dim[layer]*num_nodes,dim[layer+1]))
          else:
            self.linears.append(nn.Linear(dim[layer],dim[layer+1]))
          if self.batch_norm:
            self.batchnorm.append(BatchNorm(dim[layer+1]))
        self.relu = nn.LeakyReLU()

        self.linears.append(nn.Linear(dim[num_layers - 1], dim[-1] * num_nodes))
        self.relu = LeakyReLU()


    def forward(self, x):

        # Apply the GNN to the node features
        num_layers = self.num_layers
        linears = self.linears
        relu = self.relu
        if self.batch_norm:
          batchnorm = self.batchnorm

        out = x.reshape(-1,self.dim[0]*self.num_nodes)
        for layer in range(num_layers-1):
          out = linears[layer](out)
          if self.batch_norm:
            out = batchnorm[layer](out)
          out = relu(out)
          out = self.dropout(out)

        out = linears[num_layers-1](out)
        out = out.reshape(-1, self.num_nodes, self.dim[-1])
        out = non_linearity(out,self.val_min,self.val_max)
        return out