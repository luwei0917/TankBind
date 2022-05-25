# code from torchdrug.
import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint
from torch_scatter import scatter_mean, scatter_add, scatter_max
from torchdrug import data, utils, layers
from collections.abc import Sequence


class GraphIsomorphismConv(nn.Module):
    """
    Graph isomorphism convolution operator from `How Powerful are Graph Neural Networks?`_

    .. _How Powerful are Graph Neural Networks?:
        https://arxiv.org/pdf/1810.00826.pdf
    
    torchdrug:
        https://torchdrug.ai/

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        hidden_dims (list of int, optional): hidden dimensions
        eps (float, optional): initial epsilon
        learn_eps (bool, optional): learn epsilon or not
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, hidden_dims=None, eps=0, learn_eps=False,
                 batch_norm=False, activation="relu"):
        super(GraphIsomorphismConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim

        eps = torch.tensor([eps], dtype=torch.float32)
        if learn_eps:
            self.eps = nn.Parameter(eps)
        else:
            self.register_buffer("eps", eps)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if hidden_dims is None:
            hidden_dims = []
        self.mlp = layers.MLP(input_dim, list(hidden_dims) + [output_dim], activation)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, edge_list, edge_feature, input):
        #
        node_in = edge_list[:, 0]
        message = input[node_in]
        if self.edge_linear:
            message += self.edge_linear(edge_feature.float())
        return message


    def aggregate(self, edge_list, edge_weight, num_node, message):
        node_out = edge_list[:, 1]
        edge_weight = edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=num_node)
        return update


    def message_and_aggregate(self, edge_list, edge_weight, edge_feature, num_node, input):
        adjacency = utils.sparse_coo_tensor(edge_list.t()[:2], edge_weight,
                                            (num_node, num_node))
        update = torch.sparse.mm(adjacency.t().float(), input.float())
        if self.edge_linear:
            edge_input = edge_feature.float()
            edge_weight = edge_weight.unsqueeze(-1)
            if self.edge_linear.in_features > self.edge_linear.out_features:
                edge_input = self.edge_linear(edge_input)
            edge_update = scatter_add(edge_input * edge_weight, edge_list[:, 1], dim=0,
                                      dim_size=num_node)
            if self.edge_linear.in_features <= self.edge_linear.out_features:
                edge_update = self.edge_linear(edge_update)
            update += edge_update

        return update


    def combine(self, input, update):
        output = self.mlp((1 + self.eps) * input + update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

    def forward(self, edge_list, edge_weight, edge_feature, num_node, input):
        """
        Parameters:
            input (Tensor): node representations of shape :math:`(|V|, ...)`
        """
        update = self.message_and_aggregate(edge_list, edge_weight, edge_feature, num_node, input)
        output = self.combine(input, update)
        return output

class GIN(nn.Module):
    """
    Graph Ismorphism Network proposed in `How Powerful are Graph Neural Networks?`_

    .. _How Powerful are Graph Neural Networks?:
        https://arxiv.org/pdf/1810.00826.pdf
    
    .. _torchdrug:
        https://torchdrug.ai/
    
    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        edge_input_dim (int, optional): dimension of edge features
        num_mlp_layer (int, optional): number of MLP layers
        eps (int, optional): initial epsilon
        learn_eps (bool, optional): learn epsilon or not
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim=None, hidden_dims=None, edge_input_dim=None, num_mlp_layer=2, eps=0, learn_eps=False,
                 short_cut=False, batch_norm=False, activation="relu", concat_hidden=False,
                 readout="sum"):
        super(GIN, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            layer_hidden_dims = [self.dims[i + 1]] * (num_mlp_layer - 1)
            self.layers.append(GraphIsomorphismConv(self.dims[i], self.dims[i + 1], edge_input_dim,
                                                           layer_hidden_dims, eps, learn_eps, batch_norm, activation))

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, edge_list, edge_weight, edge_feature, num_node, input, all_loss=None, metric=None):
        """
        Compute the node representations.

        Parameters:
            
            input (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node_feature`` fields:
                node representations of shape :math:`(|V|, d)`
        """
        hiddens = []
        layer_input = input

        for layer in self.layers:
            hidden = layer(edge_list, edge_weight, edge_feature, num_node, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            layer_input = hidden

        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]

        return {
            "node_feature": node_feature
        }