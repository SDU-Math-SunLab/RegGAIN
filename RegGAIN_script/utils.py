import numpy as np
import pandas as pd
from anndata import AnnData
import networkx as nx
import scanpy as sc
from typing import Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr
import torch
import random
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data



def get_PYG_data(adata: AnnData, device: torch.device) -> Data:
    """
    Converts an AnnData object to a PyTorch Geometric Data object.
    """
    # edge index
    source_nodes = adata.uns['edgelist']['from'].tolist()
    target_nodes = adata.uns['edgelist']['to'].tolist()
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long).to(device)

    # features
    df_numeric = adata.to_df().T.apply(pd.to_numeric, errors='coerce').fillna(0)
    x = torch.from_numpy(df_numeric.to_numpy()).float().to(device)

    # degree
    num_nodes = x.size(0)
    degree = torch.zeros(num_nodes, dtype=torch.long, device=device)
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long, device=device))

    return Data(x=x, edge_index=edge_index, degree=degree)


def data_preparation(input_expData: Union[str, sc.AnnData, pd.DataFrame],
                     input_priorNet: Union[str, pd.DataFrame]) -> AnnData:
    """
    Prepare input expression data and prior network into an AnnData object
    with a graph structure and metadata annotations.
    """

    print('Start preprocessing! ')

    # [1] Load single-cell expression data
    if isinstance(input_expData, str):
        p = Path(input_expData)
        if p.suffix == '.csv':
            adata = sc.read_csv(input_expData, first_column_names=True)
        else:  # .h5ad
            adata = sc.read_h5ad(input_expData)
    elif isinstance(input_expData, sc.AnnData):
        adata = input_expData
    elif isinstance(input_expData, pd.DataFrame):
        adata = sc.AnnData(X=input_expData.iloc[:, 1:].values)
        adata.var_names = input_expData.columns[1:]
    else:
        raise Exception("Invalid input! Must be .csv, .h5ad, AnnData or DataFrame.", input_expData)

    # Convert gene symbols to uppercase
    adata.var_names = adata.var_names.str.upper()


    # [2] Load prior network
    if isinstance(input_priorNet, str):
        netData = pd.read_csv(input_priorNet, index_col=None, header=0)
    elif isinstance(input_priorNet, pd.DataFrame):
        netData = input_priorNet.copy()
    else:
        raise Exception("Invalid input!", input_priorNet)
    # Filter edges with genes in expression matrix
    netData['from'] = netData['from'].str.upper()
    netData['to'] = netData['to'].str.upper()
    netData = netData.loc[netData['from'].isin(adata.var_names.values)
                          & netData['to'].isin(adata.var_names.values), :]

    netData = netData.drop_duplicates(subset=['from', 'to'], keep='first', inplace=False)


    # [3] Create mapping from gene names to indices
    idx_GeneName_map = pd.DataFrame({
        'idx': range(adata.n_vars),
        'geneName': adata.var_names
    }, index=adata.var_names)
    # Build directed edge list
    directed_edges = []
    for _, row in netData[netData['edge_type'] == 'directed'].iterrows():
        from_idx = idx_GeneName_map.loc[row['from'], 'idx']
        to_idx = idx_GeneName_map.loc[row['to'], 'idx']
        directed_edges.append({'from': from_idx, 'to': to_idx})

    for _, row in netData[netData['edge_type'] == 'undirected'].iterrows():
        from_idx = idx_GeneName_map.loc[row['from'], 'idx']
        to_idx = idx_GeneName_map.loc[row['to'], 'idx']
        directed_edges.append({'from': from_idx, 'to': to_idx})
        directed_edges.append({'from': to_idx, 'to': from_idx})

    edgelist = pd.DataFrame(directed_edges)
    print(f"Total number of prior network edges: {len(directed_edges)}")
    out_degree = edgelist['from'].value_counts()
    high_out_degree_nodes = out_degree[out_degree > 50]
    print(f"Number of nodes with out-degree > 50: {len(high_out_degree_nodes)}")



    # [4] Normalize expression matrix
    adata.X = adata.X / adata.X.sum(axis=0) * 1e4
    sc.pp.log1p(adata)
    adata.varm['idx_GeneName_map'] = idx_GeneName_map
    adata.uns['edgelist'] = edgelist

    print(f"Finish! Data shape: n_genes × n_cells = {adata.n_vars} × {adata.n_obs}")
    return adata



def generate_pos_mask(data, k):
    """
    Generate a positive edge mask matrix.

    Args:
        data (object): PyG data object with `.edge_index`.
        k (float): Value to assign to positive positions.

    Returns:
        torch.Tensor: A (num_nodes, num_nodes) matrix with k at positive edge positions.
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    pos_mask = torch.ones((num_nodes, num_nodes), dtype=torch.float32)
    for i in range(edge_index.shape[1]):
        start_node = edge_index[0, i]
        end_node = edge_index[1, i]
        pos_mask[start_node, end_node] = k
    return pos_mask


def find_special_structures(data, k):
    """
    Identify edges originating from high out-degree nodes.

    Args:
        data (object): PyG data object.
        k (int): Out-degree threshold.

    Returns:
        torch.Tensor: List of special edges as (src, dst) pairs.
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    out_degree = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    out_degree.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.long))
    high_out_degree_nodes = (out_degree > k).nonzero(as_tuple=True)[0]
    mask = torch.isin(edge_index[0], high_out_degree_nodes)
    special_edges = edge_index[:, mask].t().tolist()
    special_edges = torch.tensor(special_edges, dtype=torch.long, device=data.edge_index.device)
    return special_edges


def drop_edges(data, alpha, beta, special_edges):
    """
    Drop a proportion of special and normal edges from the graph.

    Args:
        data (object): PyG graph data.
        alpha (float): Drop ratio for special edges.
        beta (float): Drop ratio for normal edges.
        special_edges (torch.Tensor): Tensor of special edges (num_edges, 2).

    Returns:
        torch.Tensor: Updated edge index (2, num_remaining_edges).
    """
    edge_index = data.edge_index
    special_edges_set = set(map(tuple, special_edges.tolist()))
    edge_list = edge_index.t().tolist()
    is_special_edge = torch.tensor([tuple(edge) in special_edges_set for edge in edge_list])
    special_edge_indices = is_special_edge.nonzero(as_tuple=True)[0]
    normal_edge_indices = (~is_special_edge).nonzero(as_tuple=True)[0]
    num_special_to_drop = int(alpha * len(special_edge_indices))
    drop_special_indices = torch.randperm(len(special_edge_indices))[:num_special_to_drop]
    num_normal_to_drop = int(beta * len(normal_edge_indices))
    drop_normal_indices = torch.randperm(len(normal_edge_indices))[:num_normal_to_drop]
    drop_indices = torch.cat([special_edge_indices[drop_special_indices], normal_edge_indices[drop_normal_indices]])
    keep_mask = torch.ones(len(edge_list), dtype=torch.bool)
    keep_mask[drop_indices] = False
    new_edge_index = edge_index[:, keep_mask]
    return new_edge_index


def drop_feature(data, alpha, beta, k):
    """
    Perform degree-based random feature dropout.

    Args:
        data (object): PyG data object with `.x` and `.degree`.
        alpha (float): Drop rate for high-degree nodes.
        beta (float): Drop rate for low-degree nodes.
        k (int): Degree threshold to distinguish node types.

    Returns:
        torch.Tensor: Feature matrix after dropout.
    """
    x = data.x
    degree = data.degree
    drop_prob = torch.where(degree > k, alpha, beta).unsqueeze(1)
    drop_mask = torch.rand_like(x, device=x.device) < drop_prob
    x = x.clone()
    x[drop_mask] = 0
    return x



def parse_hidden_layers(hidden_layers_str):
    """
    Parse hidden layer config string into list of layer dimensions.
    Example: '40 40 5,16 16 2' → [[40, 40, 5], [16, 16, 2]]
    """
    hidden_layers = []
    for layer_str in hidden_layers_str.split(','):
        dims = list(map(int, layer_str.split()))
        hidden_layers.append(dims)
    return hidden_layers
