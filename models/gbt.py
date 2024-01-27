from models.encoder import GCN_Encoder
import torch
from torch_geometric.nn import GCNConv
from easyGOOD.utils.register import register
from models.mlp import Two_MLP_BN, Two_MLP
import torch.nn.functional as F


@register.model_register
class GBT(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, use_bn=True, last_activation=False, **args_dicts) -> None:
        super().__init__()
        
        self.encoder = register.encoders[args_dicts['encoder_name']](input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)
        self.projector =Two_MLP_BN(hidden, hidden, hidden)
        self.classifier = torch.nn.Linear(hidden, output_dim)
        assert args_dicts['tau'] > 0
        self.tau = args_dicts['tau']
        
    def reset_classifier(self):
        self.classifier.reset_parameters()
        
    def forward(self, x, edge_index, edge_weight=None, frozen=False):
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                out = self.encoder(x, edge_index, edge_weight=edge_weight)
        else:
            out = self.encoder(x, edge_index, edge_weight=edge_weight)
        out = self.classifier(out)
        return out
    
    def pretrain(self, **kwargs):
        x1, x2, edge_index1, edge_index2, edge_weight1, edge_weight2 = kwargs['x1'], kwargs['x2'], kwargs['edge_index1'], kwargs['edge_index2'], kwargs['edge_weight1'], kwargs['edge_weight2']
        h1 = self.encoder(x1, edge_index1, edge_weight=edge_weight1)
        h2 = self.encoder(x2, edge_index2, edge_weight=edge_weight2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        loss = self.bt_loss(z1, z2, None)
        return loss
    
    def bt_loss(self, h1: torch.Tensor, h2: torch.Tensor, lambda_, batch_norm=True, eps=1e-15, *args, **kwargs):
        batch_size = h1.size(0)
        feature_dim = h1.size(1)

        if lambda_ is None:
            lambda_ = 1. / feature_dim

        if batch_norm:
            z1_norm = (h1 - h1.mean(dim=0)) / (h1.std(dim=0) + eps)
            z2_norm = (h2 - h2.mean(dim=0)) / (h2.std(dim=0) + eps)
            c = (z1_norm.T @ z2_norm) / batch_size
        else:
            c = h1.T @ h2 / batch_size

        off_diagonal_mask = ~torch.eye(feature_dim).bool()
        loss = (1 - c.diagonal()).pow(2).sum()
        loss += lambda_ * c[off_diagonal_mask].pow(2).sum()

        return loss    
        
        
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())) 
        
    def update_prototypes(self, **kwargs):
        pass
        
def diag(sigma, shape):
    
    if shape[0] > shape[1]:
        a = torch.diag_embed(sigma)
        b = torch.zeros(size=(shape[0]-shape[1], sigma.shape[0]), device=sigma.device)
        return torch.cat([a,b], dim=0)
    elif shape[0] < shape[1]:
        a = torch.diag_embed(sigma)
        b = torch.zeros(size=(sigma.shape[0], shape[1]-shape[0]), device=sigma.device)
        return torch.cat([a,b], dim=1)
    else:
        return torch.diag_embed(sigma)