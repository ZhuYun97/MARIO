from models.encoder import GCN_Encoder
import torch
from torch_geometric.utils import (add_self_loops, negative_sampling,
                                   remove_self_loops)
from easyGOOD.utils.register import register

EPS = 1e-15
MAX_LOGSTD = 10


class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def forward(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj
    
# class Classicial_GAE(GAE):
#     def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", decoder=None):
#         super().__init__(GCN_Encoder(input_dim, layer_num, hidden, activation), decoder)
        
#         self.classifier = torch.nn.Linear(hidden, output_dim)
        
#     def forward(self, x, edge_index, edge_weight=None, frozen=False):
#         if frozen:
#             with torch.no_grad():
#                 self.encoder.eval()
#                 out = self.encoder(x, edge_index, edge_weight)
#         else:
#             out = self.encoder(x, edge_index, edge_weight)
#         out = self.classifier(out)
#         return out
    
#     def pretrain(self, x, edge_index, edge_weight=None):
#         z = self.encode(x, edge_index, edge_weight=edge_weight)
#         loss = self.recon_loss(z, edge_index) # use all existing edges as training samples 
#         return loss
    
#     def reset_classifier(self):
#         self.classifier.reset_parameters()
        

@register.model_register   
class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, use_bn=True, last_activation=False, decoder=None, **args_dicts):
        super().__init__()
        self.encoder = register.encoders[args_dicts['encoder_name']](input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        self.classifier = torch.nn.Linear(hidden, output_dim)

    def reset_classifier(self):
        self.classifier.reset_parameters()

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss
    
    def forward(self, x, edge_index, edge_weight=None, frozen=False):
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                out = self.encoder(x, edge_index, edge_weight)
        else:
            out = self.encoder(x, edge_index, edge_weight)
        out = self.classifier(out)
        return out
    
    def pretrain(self, **kwargs):
        x, edge_index, edge_weight = kwargs['x'], kwargs['edge_index'], kwargs['edge_weight']
        z = self.encode(x, edge_index, edge_weight=edge_weight)
        loss = self.recon_loss(z, edge_index) # use all existing edges as training samples 
        return loss
    
    def update_prototypes(self, **kwargs):
        pass
    
    