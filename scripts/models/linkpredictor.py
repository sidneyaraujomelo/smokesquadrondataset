import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)

EPS = 1e-15

function_dict = {
    "concatenate" : (lambda x,y: torch.cat((x,y), 1)),
    "hadamard": (lambda x,y: x*y),
    "sum": (lambda x,y:x+y)
}

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, composition_function):
        super(LinkPredictor, self).__init__()

        self.lin_1 = torch.nn.Linear(in_channels, in_channels)
        self.lin_2 = torch.nn.Linear(in_channels, 1)
        if composition_function in function_dict:
            self.compose_edge = function_dict[composition_function]
        else:
            self.compose_edge = function_dict["concatenate"]
    
    def reset_parameters(self):
        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

    def forward(self, z, edge_index, sigmoid=False):        
        e = torch.cat((z[edge_index[0]], z[edge_index[1]]), dim=1)
        #print(e)
        e = self.lin_1(e)
        value = self.lin_2(e)
        #print(value[0])
        return torch.sigmoid(value) if sigmoid else value
    
    def recon_loss(self, z, pos_edge_index):
        #Calculate loss of positive samples
        pos_edge_logit = self.forward(z, pos_edge_index, sigmoid=True)
        #print(pos_edge_logit)
        
        # Do not include self-loops in negative samples
        _edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                           num_nodes=z.size(0))
        
        #Generate negative samples
        neg_edge_index = negative_sampling(pos_edge_index_with_self_loops,
            num_nodes=z.size(0),
            num_neg_samples=pos_edge_index.size(1))

        neg_edge_logit = self.forward(z, neg_edge_index, sigmoid=True)
        criterion = torch.nn.BCELoss()
        logits = torch.cat([pos_edge_logit,neg_edge_logit],dim=0)
        labels = torch.cat([torch.ones_like(pos_edge_logit),
                            torch.zeros_like(neg_edge_logit)], dim=0)
        loss = criterion(logits,labels)
        #print(loss)
        return loss