import torch
from typing import Optional, Tuple
from torch import Tensor
from torch.nn import Module
from .gae import GAE
from .encoder import VariationalGCNEncoder
from easyGOOD.utils.register import register

EPS = 1e-15
MAX_LOGSTD = 10


@register.model_register
class VGAE(GAE):
    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, decoder=None, use_bn=True, last_activation=False, **args_dicts):
        super().__init__(input_dim, layer_num, hidden, output_dim, activation, dropout, use_bn, last_activation, decoder, **args_dicts)
        self.encoder = VariationalGCNEncoder(input_dim, layer_num, hidden, activation, use_bn) # replace the encoder in GAE
        
    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs) -> Tensor:
        """"""
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (torch.Tensor, optional): The latent space for :math:`\mu`. If
                set to :obj:`None`, uses the last computation of :math:`\mu`.
                (default: :obj:`None`)
            logstd (torch.Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`. (default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
        
    def forward(self, x, edge_index, edge_weight=None, frozen=False):
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                out = self.encode(x, edge_index, edge_weight)
        else:
            out = self.encode(x, edge_index, edge_weight)
        out = self.classifier(out)
        return out
    
    def pretrain(self, **kwargs):
        x, edge_index, edge_weight = kwargs['x'], kwargs['edge_index'], kwargs['edge_weight']
        z = self.encode(x, edge_index, edge_weight=edge_weight)
        loss = self.recon_loss(z, edge_index) # use all existing edges as training samples 
        loss += self.kl_loss() * (1/x.shape[0])
        return loss
    
    def update_prototypes(self, **kwargs):
        pass