# code from torchdrug.
import functools

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint
from torch_scatter import scatter_mean, scatter_add, scatter_max
from torchdrug import data, utils, layers
from collections.abc import Sequence

class GraphAttentionConv(nn.Module):
    """
    Graph attentional convolution operator from `Graph Attention Networks`_.

    .. _Graph Attention Networks:
        https://arxiv.org/pdf/1710.10903.pdf

    .. _torchdrug:
        https://torchdrug.ai/

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        num_head (int, optional): number of attention heads
        negative_slope (float, optional): negative slope of leaky relu activation
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    eps = 1e-10

    def __init__(self, input_dim, output_dim, edge_input_dim=None, num_head=1, negative_slope=0.2, concat=True,
                 batch_norm=False, activation="relu"):
        super(GraphAttentionConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        self.num_head = num_head
        self.concat = concat
        self.leaky_relu = functools.partial(F.leaky_relu, negative_slope=negative_slope)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        if output_dim % num_head != 0:
            raise ValueError("Expect output_dim to be a multiplier of num_head, but found `%d` and `%d`"
                             % (output_dim, num_head))

        self.linear = nn.Linear(input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, output_dim)
        else:
            self.edge_linear = None
        self.query = nn.Parameter(torch.zeros(num_head, output_dim * 2 // num_head))
        nn.init.kaiming_uniform_(self.query, negative_slope, mode="fan_in")

    def message(self, edge_list, edge_weight, edge_feature, num_node, input, device):
        # add self loop
        node_in = torch.cat([edge_list[:, 0], torch.arange(num_node, device=device)])
        node_out = torch.cat([edge_list[:, 1], torch.arange(num_node, device=device)])
        edge_weight = torch.cat([edge_weight, torch.ones(num_node, device=device)])
        edge_weight = edge_weight.unsqueeze(-1)
        hidden = self.linear(input.float())

        key = torch.stack([hidden[node_in], hidden[node_out]], dim=-1)
        if self.edge_linear:
            edge_input = self.edge_linear(edge_feature.float())
            edge_input = torch.cat([edge_input, torch.zeros(num_node, self.output_dim, device=device)])
            key += edge_input.unsqueeze(-1)
        key = key.view(-1, *self.query.shape)
        weight = torch.einsum("hd, nhd -> nh", self.query, key)
        weight = self.leaky_relu(weight)

        weight = weight - scatter_max(weight, node_out, dim=0, dim_size=num_node)[0][node_out]
        attention = weight.exp() * edge_weight
        # why mean? because with mean we have normalized message scale across different node degrees
        normalizer = scatter_mean(attention, node_out, dim=0, dim_size=num_node)[node_out]
        attention = attention / (normalizer + self.eps)

        value = hidden[node_in].view(-1, self.num_head, self.query.shape[-1] // 2)
        attention = attention.unsqueeze(-1).expand_as(value)
        message = (attention * value).flatten(1)
        return message


    def aggregate(self, edge_list, edge_weight, edge_feature, num_node, message, device):
        # add self loop
        node_out = torch.cat([edge_list[:, 1], torch.arange(num_node, device=device)])
        update = scatter_mean(message, node_out, dim=0, dim_size=num_node)
        return update


    def combine(self, input, update):
        output = update
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
    def message_and_aggregate(self, edge_list, edge_weight, edge_feature, num_node, input, device):
        """
        Fused computation of message and aggregation over the graph.
        This may provide better time or memory complexity than separate calls of
        :meth:`message <MessagePassingBase.message>` and :meth:`aggregate <MessagePassingBase.aggregate>`.

        Parameters:
            input (Tensor): node representations of shape :math:`(|V|, ...)`

        Returns:
            Tensor: node updates of shape :math:`(|V|, ...)`
        """
        message = self.message(edge_list, edge_weight, edge_feature, num_node, input, device)
        update = self.aggregate(edge_list, edge_weight, edge_feature, num_node, message, device)
        return update

    def forward(self, edge_list, edge_weight, edge_feature, num_node, input, device):
        """
        Parameters:
            input (Tensor): node representations of shape :math:`(|V|, ...)`
        """
        update = self.message_and_aggregate(edge_list, edge_weight, edge_feature, num_node, input, device)
        output = self.combine(input, update)
        return output

class GAT(nn.Module):
    """
    Graph Attention Network proposed in `Graph Attention Networks`_.

    .. _Graph Attention Networks:
        https://arxiv.org/pdf/1710.10903.pdf

    .. _torchdrug:
        https://torchdrug.ai/

    Parameters:
        input_dim (int): input dimension
        hidden_dims (list of int): hidden dimensions
        edge_input_dim (int, optional): dimension of edge features
        num_head (int, optional): number of attention heads
        negative_slope (float, optional): negative slope of leaky relu activation
        short_cut (bool, optional): use short cut or not
        batch_norm (bool, optional): apply batch normalization or not
        activation (str or function, optional): activation function
        concat_hidden (bool, optional): concat hidden representations from all layers as output
        readout (str, optional): readout function. Available functions are ``sum`` and ``mean``.
    """

    def __init__(self, input_dim, hidden_dims, edge_input_dim=None, num_head=1, negative_slope=0.2, short_cut=False,
                 batch_norm=False, activation="relu", concat_hidden=False, readout="sum"):
        super(GAT, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1)
        self.dims = [input_dim] + list(hidden_dims)
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(GraphAttentionConv(self.dims[i], self.dims[i + 1], edge_input_dim, num_head,
                                                         negative_slope, batch_norm, activation))

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
        device = input.device
        for layer in self.layers:
            hidden = layer(edge_list, edge_weight, edge_feature, num_node, layer_input, device)
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