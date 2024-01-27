from models.encoder import GCN_Encoder, PMLP_GCN, GAT_Encoder
from models.mlp import Two_MLP_BN
import torch
import torch.nn.functional as F
from easyGOOD.utils.register import register


@register.model_register
class GRACE(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, use_bn=True, last_activation=False, **args_dicts) -> None:
        super().__init__()
        
        # self.encoder = GCN_Encoder(input_dim, layer_num, hidden, activation)
        # self.encoder = GAT_Encoder(input_dim, layer_num, hidden, activation)
        self.encoder = register.encoders[args_dicts['encoder_name']](input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)
        self.projector =Two_MLP_BN(hidden, hidden, hidden)
        self.classifier = torch.nn.Linear(hidden, output_dim)
        # self.classifier = GCNConv(hidden, output_dim)
        assert args_dicts['tau'] > 0
        self.tau = args_dicts['tau']
        
    def reset_classifier(self):
        self.classifier.reset_parameters()
        
    def forward(self, x, edge_index, edge_weight=None, frozen=False):
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                out = self.encoder(x, edge_index, edge_weight)
                # # low-rank approximization
                # U, S, VT = torch.linalg.svd(out.cpu())
                # # print(S)
                # S_low = torch.clamp(S, min=10)
                # low_out = U@diag(S_low, shape=(U.shape[0], VT.shape[0]))@VT
                # out = low_out.cuda()
        else:
            out = self.encoder(x, edge_index, edge_weight)
        out = self.classifier(out)
        # out = self.classifier(out, edge_index)
        return out
    
    def pretrain(self, x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2):
        h1 = self.encoder(x1, edge_index1, edge_weight=edge_weight1)
        h2 = self.encoder(x2, edge_index2, edge_weight=edge_weight2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        l1 = self.semi_loss(z1, z2)
        l2 = self.semi_loss(z2, z1)
        ret = (l1 + l2) * 0.5
        return ret.mean()
        # return ret
    
    def pretrain_arcl(self, x_list, edge_index_list, edge_weight_list):
        f = lambda x: torch.exp(x / self.tau)
        num_views = len(x_list)
        z_list = []
        for i in range(num_views):
            z_i = self.projector(self.encoder(x_list[i], edge_index_list[i], edge_weight_list[i]))
            z_list.append(z_i)
        
        worst_sim = torch.ones(x_list[0].shape[0], device=x_list[0].device) # (node_num, 1)
        for i in range(num_views):
            for j in range(i, num_views):
                sim_ij = torch.cosine_similarity(z_list[i], z_list[j])   
                worst_sim[sim_ij < worst_sim] = sim_ij[sim_ij < worst_sim] # update
        # print(worst_sim)
        # print(torch.isnan(worst_sim).sum())
        worst_sim = f(worst_sim)
        
        refl_sim = f(self.sim(z_list[0], z_list[0]))
        between_sim = f(self.sim(z_list[0], z_list[1]))
        return -torch.log(
            worst_sim
            /(refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())).mean()
        
        
        
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