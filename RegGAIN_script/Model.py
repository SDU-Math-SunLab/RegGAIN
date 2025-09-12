import torch
import torch.nn as nn
import torch.nn.functional as F
from .Layer import HighOrderConv


class ConvLayer(nn.Module):

    """
    A single high-order convolution layer that applies multiple powers of the adjacency matrix.

    Args:
        input_dim (int): Dimensionality of input node features.
        dim_per_power (List[int]): Output feature dimension for each power of the adjacency matrix.
        adjacency_powers (List[int]): List of powers of the normalized adjacency matrix to use (e.g., [0, 1, 2]).
        add_self_loops (bool): Whether to add self-loops to the graph.
    """

    def __init__(self, input_dim, dim_per_power, adjacency_powers=[0, 1, 2], add_self_loops=True):
        super(ConvLayer, self).__init__()
        self.dim_per_power = dim_per_power
        self.HighOrderConv = HighOrderConv(
            in_channels=input_dim,
            dim_per_power=dim_per_power,  # Output dimension for each adjacency power
            powers=adjacency_powers,
            add_self_loops=add_self_loops
        )

    def forward(self, x, edge_index):
        # Perform high-order convolution
        return self.HighOrderConv(x, edge_index)


class ConvModel(nn.Module):

    """
    Three-layer GNN model for multi-scale feature aggregation.

    Args:
        edge_index (Tensor): Edge list in COO format with shape (2, num_edges).
        input_dim (int): Number of input features per node.
        adjacency_powers (List[int]): Adjacency matrix powers used in layers.
        first_layer_dim_per_power (List[int]): Output dimensions for each adjacency power in the first layer.
        hidden_layer_dims_per_power_list (List[List[int]]): List of output dims for subsequent layers.
    """

    def __init__(self, edge_index, input_dim, adjacency_powers, first_layer_dim_per_power, hidden_layer_dims_per_power_list):
        super(ConvModel, self).__init__()
        self.edge_index = edge_index

        # First high order convolution layer with user-defined output dimension per power
        self.HOLayer1 = ConvLayer(input_dim=input_dim, dim_per_power=first_layer_dim_per_power, adjacency_powers=adjacency_powers)

        # Second high order convolution layer; input dim is the sum of output dims from the first layer
        self.HOLayer2 = ConvLayer(input_dim=sum(first_layer_dim_per_power), dim_per_power=hidden_layer_dims_per_power_list[0], adjacency_powers=adjacency_powers)

        # Third high order convolution layer
        self.HOLayer3 = ConvLayer(input_dim=sum(hidden_layer_dims_per_power_list[0]), dim_per_power=hidden_layer_dims_per_power_list[1], adjacency_powers=adjacency_powers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # Pass through three layers with tanh activation
        x = torch.tanh(self.HOLayer1(x, edge_index))
        x = torch.tanh(self.HOLayer2(x, edge_index))
        embeddings = torch.tanh(self.HOLayer3(x, edge_index))
        return embeddings


class Model(nn.Module):

    """
    Contrastive learning model that encodes two views of a graph using separate High-order convolutional layer encoders
    and computes symmetric contrastive loss between projected embeddings.

    Args:
        encoder_out (ConvModel): GNN encoder for outward edges.
        encoder_in (ConvModel): GNN encoder for inward (reversed) edges.
        num_proj_hidden (int): Number of hidden units in the projection head.
        tau (float): Temperature parameter for contrastive loss.
    """

    def __init__(self, encoder_out: ConvModel, encoder_in: ConvModel, num_proj_hidden: int, tau: float):
        super(Model, self).__init__()
        self.encoder_out: ConvModel = encoder_out
        self.encoder_in: ConvModel = encoder_in
        self.tau: float = tau

        # Projection head: input dim is the sum of output dims from the third layer
        encoder_in_dim = sum(encoder_in.HOLayer3.dim_per_power)
        encoder_out_dim = sum(encoder_out.HOLayer3.dim_per_power)
        self.fc1 = nn.Linear(encoder_out_dim, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, encoder_out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        # Encode features using both forward and reversed edge directions
        zout = self.encoder_out(x, edge_index)            # Outward direction embeddings
        zin = self.encoder_in(x, edge_index[[1, 0], :])   # Inward direction embeddings (reversed edges)
        return zout, zin

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        # Non-linear projection head: ELU + Linear
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        # Compute pairwise cosine similarity after L2 normalization
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        # Contrastive loss for self-supervised training
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))         # Positive samples (same view)
        between_sim = f(self.sim(z1, z2))      # Negative samples (different views)

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())
        )

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Compute semi-supervised loss in mini-batches to save memory
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))      # [B, N]
            between_sim = f(self.sim(z1[mask], z2))   # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())
            ))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: int = 0):
        # Full loss: symmetric contrastive loss with optional batching
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
