from typing import List, Optional
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm


class HighOrderConv(MessagePassing):


    def __init__(
        self,
        in_channels: int,
        dim_per_power: List[int],  # Output dimension for each adjacency power
        powers: Optional[List[int]] = None,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if powers is None:
            powers = [0, 1, 2]

        self.in_channels = in_channels
        self.dim_per_power = dim_per_power
        self.powers = powers
        self.add_self_loops = add_self_loops

        assert len(dim_per_power) == len(powers), \
            "dim_per_power must match the length of powers"

        # Initialize one linear transformation per power
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, dim_per_power[i], bias=False)
            if p in powers else torch.nn.Identity()
            for i, p in enumerate(range(max(powers) + 1))
        ])

        if bias:
            self.bias = Parameter(torch.empty(sum(dim_per_power)))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Reset all internal layers and bias
        for lin in self.lins:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        # Normalize adjacency matrix (add self-loops if needed)
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, x.dtype)

        # Store outputs for each power
        outs = [self.lins[0](x)]

        for lin in self.lins[1:]:
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            outs.append(lin.forward(x))

        # Concatenate outputs along feature dimension
        out = torch.cat([outs[i] for i in range(len(self.powers))], dim=-1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        # Message passing: apply edge weights if available
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        # Sparse matrix multiplication for efficiency
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{sum(self.dim_per_power)}, powers={self.powers})')
