import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

"""
This implementation of GraphSAGE is derived from Pytorch Geometric's example
script ogbn_products_sage.
"""

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(GraphSAGE, self).__init__()
        self.num_layers = len(hidden_channels)+1
        self.convs = torch.nn.ModuleList()
        if len(hidden_channels) > 0:
            self.convs.append(SAGEConv(in_channels, hidden_channels[0]))
            for i in range(len(hidden_channels)-1):
                self.convs.append(SAGEConv(hidden_channels[i],
                                           hidden_channels[i+1]))
            self.convs.append(SAGEConv(hidden_channels[-1], out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, out_channels))
        #for conv in self.convs:
            #torch.nn.init.kaiming_uniform_(conv.lin_l.weight)
            #torch.nn.init.kaiming_uniform_(conv.lin_r.weight)
        self.dropout = dropout
  
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
  
    def forward(self, x, adjs, sampler=True):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]] # Target nodes are always placed first
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
        
    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all
    
    def run_train(self, data, train_loader, optimizer, device):
        # default training routine for GraphSAGE
        self.train()
        total_loss = total_correct = 0
        for batch_size, n_id, adjs in train_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(device) for adj in adjs]

            optimizer.zero_grad()
            out = self.forward(data.x[n_id], adjs)
            loss = F.nll_loss(out, data.y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()

            total_loss += float(loss)
            total_correct += int(out.argmax(dim=-1).eq(data.y[n_id[:batch_size]]).sum())

        loss = total_loss / len(train_loader)
        approx_acc = total_correct / data.train.sum().item()
        return loss, approx_acc