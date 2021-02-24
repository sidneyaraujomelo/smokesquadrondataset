import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

"""
This model is responsible for creating projection matrices for each type of node in a 
heterogeneous graph.
"""

class TypeProjection(torch.nn.Module):
    def __init__(self, dim_types, output_dim):
        super(TypeProjection, self).__init__()
        self.dim_types = dim_types
        self.output_dim = output_dim
        #self.norm_mats = torch.nn.ModuleList()
        self.proj_mats = torch.nn.ModuleList()
        for dim in self.dim_types:
            #self.norm_mats.append(torch.nn.BatchNorm1d(dim))
            self.proj_mats.append(torch.nn.Linear(dim, output_dim))
        for i in range(len(self.proj_mats)):
            torch.nn.init.kaiming_uniform_(self.proj_mats[i].weight)

    def reset_parameters(self):
        for proj_mat in self.proj_mats:
            proj_mat.reset_parameters()
  
    def forward(self, x_ts, maps):
        # We forward slices of X according to maps to their specific projection matrix
        #for t, node_ids in enumerate(maps):
        #    x[node_ids] = self.proj_mats[t](x_ts[t])
        #return torch.cat(tuple([self.proj_mats[t](self.norm_mats[t](x_ts[t])) for t, _ in enumerate(maps)]), dim=0)
        return torch.cat(tuple([self.proj_mats[t](x_ts[t]) for t, _ in enumerate(maps)]), dim=0)
        #return x