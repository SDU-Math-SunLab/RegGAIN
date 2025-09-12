import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, Tuple
import random
from .Model import Model, ConvModel
from .utils import (
    data_preparation, 
    get_PYG_data,     
    parse_hidden_layers,
    find_special_structures,
    drop_edges,
    drop_feature,
    generate_pos_mask
)
from .evaluation_utils import calculate_epr_aupr

# --- Core training functions ---
def _train_single_epoch(model: Model, data, optimizer, params: Dict[str, Any], device: torch.device):
    """
    Helper function to run a single training epoch.
    """
    model.train()
    optimizer.zero_grad()
    
    # Get hyperparameters for data augmentation from the params dictionary
    special_edges = find_special_structures(data, params['k'])
    
    edge_index_1 = drop_edges(data, params['edge_alpha1'], params['edge_beta1'], special_edges)
    edge_index_2 = drop_edges(data, params['edge_alpha2'], params['edge_beta2'], special_edges)
    
    x_1 = drop_feature(data, params['node_alpha1'], params['node_beta1'], params['k'])
    x_2 = drop_feature(data, params['node_alpha2'], params['node_beta2'], params['k'])
    
    z1_out, z1_in = model(x_1.to(device), edge_index_1.to(device))
    z2_out, z2_in = model(x_2.to(device), edge_index_2.to(device))
    
    loss_out = model.loss(z1_out, z2_out)
    loss_in = model.loss(z1_in, z2_in)
    loss = loss_in + loss_out
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def _test(model: Model, data, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to get embeddings from a trained model.
    """
    model.eval()
    with torch.no_grad():
        z_in, z_out = model(data.x.to(device), data.edge_index.to(device))
    return z_in.cpu().numpy(), z_out.cpu().numpy()


# --- Main API function ---
def run_reggain(
    exp_data, 
    prior_net,
    config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Main API function to run the RegGAIN algorithm.

    Args:
        exp_data (Union[str, pd.DataFrame]): Path to expression data or DataFrame.
        prior_net (Union[str, pd.DataFrame]): Path to prior network or DataFrame.
        config (Dict[str, Any]): A dictionary of hyperparameters.

    Returns:
        A dictionary containing the resulting GRN DataFrame and embedding DataFrames.
    """
    # 1. Set random seeds and device
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device_str = config.get('device', 'cuda')
        device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU, no GPU available.")
        
    # 2. Data preprocessing
    adata = data_preparation(exp_data, prior_net)
    pyg_data = get_PYG_data(adata, device)
    
    # 3. Parse model parameters
    num_features = pyg_data.x.size(1)
    adjacency_powers = config.get('adjacency_powers', [0, 1, 2])
    first_layer_dims = config.get('first_layer_dims', [80, 80, 10])
    hidden_layer_dims_list = parse_hidden_layers(config.get('hidden_layer_dims_list', "40 40 5,16 16 2"))

    print("Start training!")
    all_z_in, all_z_out = [], []
    
    # 4. Training loop
    num_repeats = config.get('repeat', 1)
    for run in range(num_repeats):
        # Initialize the model
        encoder_out = ConvModel(
            edge_index=pyg_data.edge_index, 
            input_dim=num_features,
            adjacency_powers=adjacency_powers,
            first_layer_dim_per_power=first_layer_dims,  
            hidden_layer_dims_per_power_list=hidden_layer_dims_list
        ).to(device)
        encoder_in = ConvModel(
            edge_index=pyg_data.edge_index, 
            input_dim=num_features,
            adjacency_powers=adjacency_powers,
            first_layer_dim_per_power=first_layer_dims,
            hidden_layer_dims_per_power_list=hidden_layer_dims_list
        ).to(device)
        model = Model(encoder_out=encoder_out, encoder_in=encoder_in, num_proj_hidden=64, tau=3).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
        
        # Train a single model
        epochs = config.get('epochs', 500)
        for epoch in tqdm(range(epochs), desc=f'Run {run + 1}/{num_repeats}', unit='epoch'):
            loss = _train_single_epoch(model, pyg_data, optimizer, config, device)
            
        # Test and collect results
        z_in, z_out = _test(model, pyg_data, device)
        all_z_in.append(z_in)
        all_z_out.append(z_out)

    print("Training finished. Processing results...")
    

    num_genes = pyg_data.x.shape[0]
    GRN_matrix_sum = np.zeros((num_genes, num_genes))
    pos_mask = generate_pos_mask(pyg_data, config.get('pos', 10)).cpu().numpy()

    # Iterate over the embeddings from each run
    for z_in, z_out in zip(all_z_in, all_z_out):
        # Normalize the embeddings for a single run
        std_z_in = np.std(z_in, axis=1, keepdims=True) + 1e-8
        std_z_out = np.std(z_out, axis=1, keepdims=True) + 1e-8
        mean_z_in = np.mean(z_in, axis=1, keepdims=True)
        mean_z_out = np.mean(z_out, axis=1, keepdims=True)
        
        normalized_z_in = (z_in - mean_z_in) / std_z_in
        normalized_z_out = (z_out - mean_z_out) / std_z_out
        # Calculate the GRN matrix for the single run
        GRN_matrix = np.dot(normalized_z_out, normalized_z_in.T)
        GRN_matrix = GRN_matrix * pos_mask
        np.fill_diagonal(GRN_matrix, 0)
        
        GRN_matrix_sum += GRN_matrix
        
    # Calculate the average GRN matrix
    avg_GRN_matrix = GRN_matrix_sum / num_repeats
    
    # Convert the average GRN matrix to DataFrame format
    original_indices = adata.var.index.tolist()
    rows, cols = np.indices(avg_GRN_matrix.shape)
    
    GRN_df = pd.DataFrame({
        'TF': [original_indices[i] for i in rows.flatten()],
        'Target': [original_indices[j] for j in cols.flatten()],
        'value': avg_GRN_matrix.flatten()
    })
    GRN_df['value'] = abs(GRN_df['value'])
    GRN_df = GRN_df.sort_values('value', ascending=False).reset_index(drop=True)

    avg_z_in = np.mean(all_z_in, axis=0)
    avg_z_out = np.mean(all_z_out, axis=0)
    norm_avg_z_in = (avg_z_in - np.mean(avg_z_in, axis=1, keepdims=True)) / (np.std(avg_z_in, axis=1, keepdims=True) + 1e-8)
    norm_avg_z_out = (avg_z_out - np.mean(avg_z_out, axis=1, keepdims=True)) / (np.std(avg_z_out, axis=1, keepdims=True) + 1e-8)
    
    # Prepare the embedding DataFrames for return
    gene_names = adata.var_names.tolist()
    z_in_df = pd.DataFrame(norm_avg_z_in, index=gene_names)
    z_out_df = pd.DataFrame(norm_avg_z_out, index=gene_names)

    print("Result processing complete.")

    return {
        "GRN": GRN_df,
        "embedding_in": z_in_df,
        "embedding_out": z_out_df
    }