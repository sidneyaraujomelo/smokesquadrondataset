import json
import os
import torch
import numpy as np
from argparse import ArgumentParser
from networkx.readwrite import json_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import degree
from torch_cluster import random_walk
from torch_geometric.utils.convert import from_networkx
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from models import load_model
from pytorchtools import EarlyStopping
from sshet_exp import SSHetBaseExperiment
import pickle
import time

experiment = SSHetBaseExperiment(experiment_tag="fsp", epochs=1000)
print(f"Saving output to {experiment.output_file}")
#Read all necessary data
data, atbsets_list, node_maps, node_feats, train_folds, test_folds = experiment.read_data()

#Get list of all attributes in any type of node
atbs_list = [atb for atbsets_sublist in atbsets_list for atb in atbsets_sublist]
atbs_list = list(dict.fromkeys(atbs_list))
#Create a dictionary mapping each attribute to its index on node_feats
atbs_maps = {atb : {} for atb in atbs_list}
f = lambda l,v: l.index(v) if v in l else -1
for atb in atbs_maps.keys():
    atbs_maps[atb] = {k:f(v,atb) for k,v in enumerate(atbsets_list)}
print(atbs_maps)
print(node_feats[0].shape)
for i,_ in enumerate(node_feats):
    new_node_feats = torch.zeros((node_feats[i].shape[0],len(atbs_list)*2))
    for atb, atb_dict in atbs_maps.items():
        atb_final_index = atbs_list.index(atb)
        if atb_dict[i] != -1:
            new_node_feats[:,atb_final_index] = node_feats[i][:, atb_dict[i]]
            new_node_feats[:,len(atbs_list)+atb_final_index] = torch.ones(node_feats[i].shape[0])
    node_feats[i] = new_node_feats

node_feats = torch.cat(node_feats)

experiment.log(f"Standardization: {experiment.standardization}\n")
if experiment.standardization:
    node_feats = experiment.standardize(node_feats)

#Configuration
num_samples = [2, 2]
batch_size = 64
walk_length = 1
num_neg_samples = 1
sage_input_dim = 64

dict_x2m = {}
#update edge_index according to node_maps
for node_map in node_maps:
    offset = len(dict_x2m)
    dict_x2m.update({k+offset:v for k,v in enumerate(node_map)})
data.edge_index[0] = torch.LongTensor([dict_x2m[idx] for idx in data.edge_index[0].tolist()])
data.edge_index[1] = torch.LongTensor([dict_x2m[idx] for idx in data.edge_index[1].tolist()])
data.x = node_feats
dim_types = [data.x.shape[-1]]
print(dim_types)
data.node_map = [0]

subgraph_loader = NeighborSampler(
    data.edge_index, node_idx=None,
    sizes=[-1], batch_size=batch_size, shuffle=False,
    num_workers=0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sage_config = json.load(open(experiment.model_config))[0]
sage_config["in_channels"] = sage_input_dim
sage_model = load_model(sage_config)
sage_model = sage_model.to(device)

typeproj_config = {
    "model": "typeprojection",
    "dim_types": dim_types,
    "dim_output": sage_config["in_channels"]
}
typeproj_model = load_model(typeproj_config)
typeproj_model = typeproj_model.to(device)

typeproj_optimizer = torch.optim.Adam(typeproj_model.parameters(), lr=1e-3)
sage_optimizer = torch.optim.Adam(sage_model.parameters(), lr=1e-3)
x = data.x.to(device)

#torch.autograd.set_detect_anomaly(True)
experiment.log(f"Device: {device}, visible devices: {os.environ['CUDA_VISIBLE_DEVICES']}\n" +
    f"Type Projection: {typeproj_model}\n" +
    f"Graph Representation Learning Model: {sage_model}")

def train(epoch):
    sage_model.train()
    typeproj_model.train()

    pbar = tqdm(total=data.num_nodes)
    pbar.set_description(f'Epoch {epoch:02d}')

    #Type Projection Step
    #print(x)
    x_p = typeproj_model([x], [0])
    #print(x_p)
    #print(x[0])

    total_loss = 0

    node_idx = torch.randperm(data.num_nodes)  # shuffle all nodes
    train_loader = NeighborSampler(
        data.edge_index, node_idx=node_idx,
        sizes=num_samples, batch_size=batch_size, shuffle=False,
        num_workers=0)

    rw = random_walk(
        data.edge_index[0], data.edge_index[1],
        node_idx, walk_length=walk_length)
    rw_idx = rw[:, 1:].flatten()
    pos_loader = NeighborSampler(
        data.edge_index, node_idx=rw_idx,
        sizes=num_samples, batch_size=batch_size * walk_length, shuffle=False,
        num_workers=0)

    # negative sampling as node2vec
    deg = degree(data.edge_index[0])
    distribution = deg ** 0.75
    neg_idx = torch.multinomial(
        distribution, data.num_nodes * num_neg_samples, replacement=True)
    neg_loader = NeighborSampler(
        data.edge_index, node_idx=neg_idx,
        sizes=num_samples, batch_size=batch_size * num_neg_samples,
        shuffle=True, num_workers=0)

    typeproj_optimizer.zero_grad()

    for (batch_size_, u_id, adjs_u), (_, v_id, adjs_v), (_, vn_id, adjs_vn) in\
            zip(train_loader, pos_loader, neg_loader):
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs_u = [adj.to(device) for adj in adjs_u]
        z_u = sage_model(x_p[u_id], adjs_u)

        adjs_v = [adj.to(device) for adj in adjs_v]
        z_v = sage_model(x_p[v_id], adjs_v)

        adjs_vn = [adj.to(device) for adj in adjs_vn]
        z_vn = sage_model(x_p[vn_id], adjs_vn)

        sage_optimizer.zero_grad()
        pos_loss = -F.logsigmoid(
            (z_u.repeat_interleave(walk_length, dim=0)*z_v)
            .sum(dim=1)).mean()
        neg_loss = -F.logsigmoid(
            -(z_u.repeat_interleave(num_neg_samples, dim=0)*z_vn)
            .sum(dim=1)).mean()
        loss = pos_loss + neg_loss
        loss.backward(retain_graph=True)
        sage_optimizer.step()

        total_loss += loss.item()
        pbar.update(batch_size_)
    typeproj_optimizer.step()
    pbar.close()

    loss = total_loss / len(train_loader)
    return loss

sage_early_stopping = EarlyStopping(patience=50, verbose=True,
    path=f"embed_{experiment.experiment_tag}_{experiment.timestamp}.pt")
proj_early_stopping = EarlyStopping(patience=50, verbose=True,
    path=f"proj_{experiment.experiment_tag}_{experiment.timestamp}.pt")
best_loss = 50
for epoch in range(1, experiment.epochs):
    loss = train(epoch)
    if loss < best_loss:
        best_loss = loss
    sage_early_stopping(loss, sage_model)
    proj_early_stopping(loss, typeproj_model)
    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')
    if sage_early_stopping.early_stop:
        print("Early Stopping!")
        break
print(f'Best Loss: {best_loss:.4f}')
experiment.log(f'Epoch: {epoch} -> Loss (Representation Learning): {best_loss:.4f}')

typeproj_model.load_state_dict(torch.load(proj_early_stopping.path))
sage_model.load_state_dict(torch.load(sage_early_stopping.path))
x_p = typeproj_model([x], [0])
z = sage_model.inference(x_p, subgraph_loader, device).detach()
z = z.to(device)

linkpred_config = json.load(open(experiment.model_config))[1]
linkpred_config["in_channels"] = z.size(1)*2
linkpred_model = load_model(linkpred_config).to(device)
link_optimizer = torch.optim.Adam(linkpred_model.parameters(), lr=1e-3)

experiment.link_prediction_step(device, linkpred_model, link_optimizer, train_folds, z, test_folds, dict_x2m)