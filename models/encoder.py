import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch.nn import BatchNorm1d, Identity
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from .conv import gcn_conv
import torch.nn as nn
from easyGOOD.utils.register import register


def get_activation(name: str):
        activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }
        return activations[name]

@register.encoder_register
class GCN_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128, activation="relu", dropout=0.5, use_bn=True, last_activation=True):
        super(GCN_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.use_bn = use_bn

        self.convs = ModuleList()
        self.bns = ModuleList()
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(GCNConv(input_dim, hidden)) 
            for i in range(layer_num-2):
                self.convs.append(GCNConv(hidden, hidden))
                # glorot(self.convs[i].weight) # initialization
            self.convs.append(GCNConv(hidden, hidden))
            # glorot(self.convs[-1].weight)
            for i in range(layer_num):
                if use_bn:
                    self.bns.append(BatchNorm1d(hidden))
                else:
                    self.bns.append(Identity())

        else: # one layer gcn
            self.convs.append(GCNConv(input_dim, hidden)) 
            # glorot(self.convs[-1].weight)
            if use_bn:
                self.bns.append(BatchNorm1d(hidden))
            else:
                self.bns.append(Identity())
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index, edge_weight=None):
        # print('Inside Model:  num graphs: {}, device: {}'.format(
        #     data.num_graphs, data.batch.device))
        # x, edge_index = data.x, data.edge_index
        for i in range(self.layer_num):
            # x = self.convs[i](x, edge_index, edge_weight)
            # print(i, x.dtype, self.convs[i].lin.weight.dtype)
            x = self.bns[i](self.convs[i](x, edge_index, edge_weight))
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
            # x = self.activation(self.convs[i](x, edge_index, edge_weight))
            # x = self.bns[i](x)
            # x = self.activation(self.bns[i](self.convs[i](x, edge_index)))
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            if self.use_bn:
                self.bns[i].reset_parameters()


@register.encoder_register
class SAGE_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128, activation="relu", dropout=0.5, use_bn=True, last_activation=True):
        super(SAGE_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.use_bn = use_bn

        self.convs = ModuleList()
        self.bns = ModuleList()
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(SAGEConv(input_dim, hidden)) 
            for i in range(layer_num-2):
                self.convs.append(SAGEConv(hidden, hidden))
                # glorot(self.convs[i].weight) # initialization
            self.convs.append(SAGEConv(hidden, hidden))
            # glorot(self.convs[-1].weight)
            for i in range(layer_num):
                if use_bn:
                    self.bns.append(BatchNorm1d(hidden))
                else:
                    self.bns.append(Identity())

        else: # one layer gcn
            self.convs.append(SAGEConv(input_dim, hidden)) 
            # glorot(self.convs[-1].weight)
            if use_bn:
                self.bns.append(BatchNorm1d(hidden))
            else:
                self.bns.append(Identity())
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index, edge_weight=None):
        # print('Inside Model:  num graphs: {}, device: {}'.format(
        #     data.num_graphs, data.batch.device))
        # x, edge_index = data.x, data.edge_index
        for i in range(self.layer_num):
            # x = self.convs[i](x, edge_index, edge_weight)
            # print(i, x.dtype, self.convs[i].lin.weight.dtype)
            x = self.bns[i](self.convs[i](x, edge_index))
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
            # x = self.activation(self.convs[i](x, edge_index, edge_weight))
            # x = self.bns[i](x)
            # x = self.activation(self.bns[i](self.convs[i](x, edge_index)))
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            if self.use_bn:
                self.bns[i].reset_parameters()


@register.encoder_register
class GAT_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128, activation="relu", dropout=0.5, use_bn=True, last_activation=True):
        super(GAT_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation

        self.convs = ModuleList()
        self.bns = ModuleList()
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(GATConv(input_dim, hidden)) 
            for i in range(layer_num-2):
                self.convs.append(GATConv(hidden, hidden))
                # glorot(self.convs[i].weight) # initialization
            self.convs.append(GATConv(hidden, hidden))
            # glorot(self.convs[-1].weight)
            for i in range(layer_num-1):
                self.bns.append(BatchNorm1d(hidden))
                # self.acts.append(self.activation) 
            self.bns.append(BatchNorm1d(hidden))
            # self.acts.append(self.activation) 
        else: # one layer gcn
            self.convs.append(GATConv(input_dim, hidden)) 
            # glorot(self.convs[-1].weight)
            self.bns.append(BatchNorm1d(hidden))
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index, edge_weight=None):
        # print('Inside Model:  num graphs: {}, device: {}'.format(
        #     data.num_graphs, data.batch.device))
        # x, edge_index = data.x, data.edge_index
        for i in range(self.layer_num):
            # x = self.convs[i](x, edge_index, edge_weight)
            # print(i, x.dtype, self.convs[i].lin.weight.dtype)
            x = self.bns[i](self.convs[i](x, edge_index))
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            self.bns[i].reset_parameters()

    
@register.encoder_register 
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128, activation="relu", use_bn=True):
        super().__init__()
        assert layer_num >= 2
        self.layer_num = layer_num
        self.convs = ModuleList()
        self.bns = ModuleList()
        for i in range(layer_num-1):
            if use_bn:
                self.bns.append(BatchNorm1d(hidden))
            else:
                self.bns.append(Identity())
            if i == 0:
                self.convs.append(GCNConv(input_dim, hidden))
            else:
                self.convs.append(GCNConv(hidden, hidden))
        self.conv_mu = GCNConv(hidden, hidden)
        self.conv_logstd = GCNConv(hidden, hidden)
        self.activation = get_activation(activation)

    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.layer_num-1):
            x = self.activation(self.bns[i](self.convs[i](x, edge_index, edge_weight)))
        return self.conv_mu(x, edge_index, edge_weight), self.conv_logstd(x, edge_index, edge_weight)
    
    
@register.encoder_register    
class G2CL_GCNEncoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128, activation="relu") -> None:
        super().__init__()
        assert layer_num >= 2
        self.layer_num = layer_num
        self.convs = ModuleList()
        for i in range(layer_num-1):
            if i == 0:
                self.convs.append(GCNConv(input_dim, hidden))
            else:
                self.convs.append(GCNConv(hidden, hidden))
        self.conv_mu = GCNConv(hidden, hidden)
        self.conv_sigma = GCNConv(hidden, hidden)
        self.activation = get_activation(activation)
    
    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.layer_num-1):
            x = self.activation(self.convs[i](x, edge_index, edge_weight))
        mu, sigma = self.conv_mu(x, edge_index, edge_weight), self.conv_sigma(x, edge_index, edge_weight)
        sigma = softplus(sigma) + 1e-7
        # mu = self.activation(mu) # delete
        
        return Independent(Normal(loc=mu, scale=sigma), 1)
    
@register.encoder_register
class PMLP_GCN(nn.Module): 
    def __init__(self, input_dim, layer_num=2, hidden=128, activation="relu"):
        super(PMLP_GCN, self).__init__()
        # self.dropout = args.dropout
        self.num_layers = layer_num
        self.ff_bias = True  # Use bias for FF layers in default

        self.bns = nn.BatchNorm1d(hidden, affine=False, track_running_stats=False)
        self.activation = get_activation(activation)

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(input_dim, hidden, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden, hidden, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden, hidden, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.0)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, edge_weight=None, use_conv=True):
        for i in range(self.num_layers - 1):
            x = x @ self.fcs[i].weight.t() 
            if use_conv: x = gcn_conv(x, edge_index, edge_weight)  # Optionally replace 'gcn_conv' with other conv functions in conv.py
            if self.ff_bias: x = x + self.fcs[i].bias
            x = self.activation(self.bns(x))
            # x = F.dropout(x, p=self.dropout, training=self.training)

        x = x @ self.fcs[-1].weight.t() 
        if use_conv: x = gcn_conv(x, edge_index, edge_weight)
        if self.ff_bias: x = x + self.fcs[-1].bias
        return x
    

