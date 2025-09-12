# run.py

import argparse
import os
import pandas as pd
from RegGAIN_script.run_RegGAIN import run_reggain

def main():
    """
    Parses command-line arguments and runs the RegGAIN pipeline.
    """
    parser = argparse.ArgumentParser(description="Train RegGAIN model from the command line.")
    
    # --- Input/Output Arguments ---
    parser.add_argument('--exp_data', type=str, required=True, help="Path to expression data CSV")
    parser.add_argument('--prior_net', type=str, required=True, help="Path to prior network CSV")
    parser.add_argument('--output_dir', type=str, default="./RegGAIN_results", help="Directory to save the output files")
    
    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=500, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--device', type=str, default="cuda", choices=["cpu", "cuda"], help="Device to run the model")
    parser.add_argument('--repeat', type=int, default=1, help="Number of training repetitions")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility") 
    
    # --- Model & Data Augmentation Parameters ---
    parser.add_argument('--k', type=int, default=50, help="Degree centrality threshold for special nodes")
    parser.add_argument('--adjacency_powers', type=int, nargs='+', default=[0, 1, 2], help="List of adjacency matrix powers.")
    parser.add_argument('--first_layer_dims', type=int, nargs='+', default=[80, 80, 10], help="Dimensions of embeddings for each layer.")
    parser.add_argument('--hidden_layer_dims_list', type=str, default="40 40 5,16 16 2",
                        help="Dimensions for hidden layers. Example: '40 40 5,16 16 2'")
    parser.add_argument('--pos', type=int, default=10, help="Gamma parameter for positive mask")
    parser.add_argument('--edge_alpha1', type=float, default=0.6, help="Drop ratio for special edges (view 1)")
    parser.add_argument('--edge_alpha2', type=float, default=0.3, help="Drop ratio for special edges (view 2)")
    parser.add_argument('--edge_beta1', type=float, default=0.3, help="Drop ratio for normal edges (view 1)")
    parser.add_argument('--edge_beta2', type=float, default=0.3, help="Drop ratio for normal edges (view 2)")
    
    parser.add_argument('--node_alpha1', type=float, default=0.5, help="Drop rate for high-degree node features (view 1)")
    parser.add_argument('--node_alpha2', type=float, default=0.2, help="Drop rate for high-degree node features (view 2)")
    parser.add_argument('--node_beta1', type=float, default=0.2, help="Drop rate for low-degree node features (view 1)")
    parser.add_argument('--node_beta2', type=float, default=0.2, help="Drop rate for low-degree node features (view 2)")
    
    args = parser.parse_args()
    config = vars(args)

    print("Starting RegGAIN pipeline from command line...")
    results = run_reggain(
        exp_data=args.exp_data,
        prior_net=args.prior_net,
        config=config
    )
    print("Pipeline finished. Saving results...")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    grn_df = results['GRN']
    grn_df.to_csv(os.path.join(output_dir, "inferred_GRN.csv"), index=False)
    print(f"Saved inferred GRN to: {os.path.join(output_dir, 'inferred_GRN.csv')}")

    embedding_in = results['embedding_in']
    embedding_in.to_csv(os.path.join(output_dir, "embedding_in.csv"))
    print(f"Saved IN-embeddings to: {os.path.join(output_dir, 'embedding_in.csv')}")

    embedding_out = results['embedding_out']
    embedding_out.to_csv(os.path.join(output_dir, "embedding_out.csv"))
    print(f"Saved OUT-embeddings to: {os.path.join(output_dir, 'embedding_out.csv')}")
    
    print("All results have been saved.")

if __name__ == "__main__":
    main()