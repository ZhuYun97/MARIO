
import torch
import torch.nn.functional as F
from models.mlp import Two_MLP, Two_MLP_BN
import torch
import torch.nn.functional as F
from easyGOOD.utils.register import register


@register.model_register
class COSTA(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, use_bn=True, last_activation=False, **args_dicts):
        super(COSTA, self).__init__()
        self.encoder = register.encoders[args_dicts['encoder_name']](input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)

        self.projector = Two_MLP(hidden, hidden, hidden)
        self.classifier = torch.nn.Linear(hidden, output_dim)
        
        assert args_dicts['tau'] > 0
        self.tau = args_dicts['tau']
        
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
    
    def pretrain(self, x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2):
        h1 = self.encoder(x1, edge_index1, edge_weight=edge_weight1)
        h2 = self.encoder(x2, edge_index2, edge_weight=edge_weight2)
        
        k = torch.tensor(int(h1.shape[0] * 0.5))
        p = (1/torch.sqrt(k))*torch.randn(k, h1.shape[0], device=h1.device)
        
        h1 = p @ h1
        h2 = p @ h2 
        
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        
        l1 = self.infonce(z1, z2)
        l2 = self.infonce(z2, z1)
        loss = 0.5*(l1+l2)
        return loss.mean()

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    def infonce(self, anchor, sample):
        sim = self.sim(anchor, sample) / self.tau
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob.diag()
        return -loss.mean()

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())


            