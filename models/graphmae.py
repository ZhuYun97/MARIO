import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from functools import partial
from torch_geometric.data import Data
import copy
from easyGOOD.utils.register import register
from torch_geometric.nn import GCNConv


# loss function: sce
def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


# loss function: sig
def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss


def mask_edge(graph, mask_prob):
    E = graph.num_edges
    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


# graph transformation: drop edge
def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edge_index[0]
    dst = graph.edge_index[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = Data(edge_index=torch.stack((nsrc, ndst), 0))
    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


@register.model_register
class GraphMAE(nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, use_bn=True, last_activation=False, **args_dicts):
        super(GraphMAE, self).__init__()
        mask_rate = 0.75 # cora: 0.75
        drop_edge_rate = 0.25 # cora: 0.25
        alpha_l = 2
        
        self._mask_rate = mask_rate
        self._encoder_type = 'GCN'
        self._decoder_type = 'GCN'
        self._drop_edge_rate = drop_edge_rate
        
        self._output_hidden_size = hidden
        self._concat_hidden = False
        self._replace_rate = 0.0
        self._mask_token_rate = 1 - self._replace_rate

        # build encoder
        dec_in_dim = hidden
        # dec_num_hidden = num_hidden
        self.encoder = register.encoders[args_dicts['encoder_name']](input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)

        # build decoder
        # self.decoder = register.encoders[args_dicts['encoder_name']](hidden, 1, input_dim, activation)
        self.decoder = GCNConv(hidden, input_dim)
        

        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        if self._concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * layer_num, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        # setup loss function
        self.criterion = self.setup_loss_fn('sce', alpha_l)
        self.classifier = torch.nn.Linear(hidden, output_dim)
    
    def reset_classifier(self):
        self.classifier.reset_parameters()
    
    def forward(self, x, edge_index, edge_weight=None, frozen=False):
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                out = self.encoder(x, edge_index, edge_weight)
        else:
            out = self.encoder(x, edge_index, edge_weight)
        out = self.classifier(out)
        # out = self.classifier(out, edge_index)
        return out
    
    def pretrain(self, **kwargs):
        g = kwargs['data']
        x = kwargs['x']
        loss = self.mask_attr_prediction(g, x)
        return loss

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes
        perm = torch.randperm(num_nodes, device=x.device)
        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]
            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0
        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()
        return use_g, out_x, (mask_nodes, keep_nodes)

    def mask_attr_prediction(self, g, x):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)
        if self._drop_edge_rate > 0:
            use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g
        enc_rep = self.encoder(x=use_x, edge_index=use_g.edge_index)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)
        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)
        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0
        if self._decoder_type in ("mlp", "linear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(x=rep, edge_index=pre_use_g.edge_index)
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]
        loss = self.criterion(x_rec, x_init)
        return loss

    def embed(self, g, x):
        rep = self.encoder(x=x, edge_index=g.edge_index)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
    
    def update_prototypes(self, **kwargs):
        pass