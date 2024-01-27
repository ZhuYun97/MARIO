from models.encoder import GCN_Encoder, PMLP_GCN, GAT_Encoder
from models.mlp import Two_MLP_BN
import torch
import torch.nn.functional as F
from easyGOOD.utils.register import register
from torch import Tensor


@register.model_register
class MARIO(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, use_bn=True, last_activation=False, **args_dicts) -> None:
        super().__init__()
        
        # self.encoder = GCN_Encoder(input_dim, layer_num, hidden, activation)
        # self.encoder = GAT_Encoder(input_dim, layer_num, hidden, activation)
        self.encoder = register.encoders[args_dicts['encoder_name']](input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)
        self.projector = Two_MLP_BN(hidden, hidden, hidden)
        self.classifier = torch.nn.Linear(hidden, output_dim)
        self.weight = torch.nn.Parameter(torch.Tensor(hidden, hidden))

        assert args_dicts['tau'] > 0 and args_dicts['num_clusters'] > 0
        self.tau = args_dicts['tau']
        self.prototypes = torch.nn.Linear(hidden, args_dicts['num_clusters'], bias=False)
        
        self.prototypes_lr = args_dicts['prototypes_lr']
        self.prototypes_ites = args_dicts['prototypes_iters']
        self.cmi_coefficient = args_dicts['cmi_coefficient']
        torch.nn.init.xavier_uniform_(self.prototypes.weight.data)
        
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
        return out
    
    
    def update_prototypes(self, x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2):
        with torch.no_grad():
            h1 = self.encoder(x1, edge_index1, edge_weight=edge_weight1)
            h2 = self.encoder(x2, edge_index2, edge_weight=edge_weight2)
        
        
        # inner step, optimize prototypes
        pro_optimizer = torch.optim.Adam([{'params': self.prototypes.parameters(), 'lr': self.prototypes_lr, 'weight_decay':1e-5}, # 1e-3 for GOOD-Twitch
                                           {'params': list(self.projector.parameters()), 'lr': self.prototypes_lr}])
        for i in range(self.prototypes_ites):
            pro_optimizer.zero_grad()
            z1 = self.projector(h1)
            z2 = self.projector(h2)
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            # normalize the prototypes
            with torch.no_grad():
                w = self.prototypes.weight.data.clone()
                w = torch.nn.functional.normalize(w, dim=1, p=2)
                # print("w:{}".format(w))
                self.prototypes.weight.copy_(w)
            
            scores1 = self.prototypes(z1) # B*K
            scores2 = self.prototypes(z2) # B*K
        
            # compute assignments
            with torch.no_grad():
                q1 = self.sinkhorn(scores1)
                q2 = self.sinkhorn(scores2)
            online_clus_loss = -0.5*(q1*F.log_softmax(scores2/self.tau, dim=1) + q2*F.log_softmax(scores1/self.tau, dim=1)) 
            # + 0.1*self.constraint() # add orthognal loss
            online_clus_loss = torch.sum(online_clus_loss, dim=1).mean()
            
            online_clus_loss.backward()
            pro_optimizer.step()
        # normalize the prototypes
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = torch.nn.functional.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)
    
    def pretrain(self, **kwargs):   
        x1, x2, edge_index1, edge_index2, edge_weight1, edge_weight2 = kwargs['x1'], kwargs['x2'], kwargs['edge_index1'], kwargs['edge_index2'], kwargs['edge_weight1'], kwargs['edge_weight2']
        h1 = self.encoder(x1, edge_index1, edge_weight=edge_weight1)
        h2 = self.encoder(x2, edge_index2, edge_weight=edge_weight2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        
        l1 = self.semi_loss(z1, z2)
        l2 = self.semi_loss(z2, z1)
        ret = (l1 + l2) * 0.5
        loss = ret
        # re-compute
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        scores1 = self.prototypes(z1) # B*K
        scores2 = self.prototypes(z2) # B*K
        
        # compute assignments
        with torch.no_grad():
            q1 = self.sinkhorn(scores1)
            q2 = self.sinkhorn(scores2)

        ccl_loss = self.weak_cmi(z1, z2, q1, q2)
        loss += -self.cmi_coefficient*ccl_loss
        return loss.mean()
        # return ret

        
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
        
    def sinkhorn(self, out, epsilon=0.1, n_iters=3):
        Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        # dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            # dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        return Q.t()
    
    
    def constraint(self):
        return torch.norm(torch.mm(self.prototypes.weight,self.prototypes.weight.T)-
                          torch.eye(self.prototypes.weight.shape[0], device=self.prototypes.weight.device))
    
    def weak_cmi(self, z1, z2, q1, q2):
        N = z1.shape[0]
        EPS = 1e-5
        f = lambda x: torch.exp(x / 1)
        # f = lambda x: torch.exp(x / self.tau)
        # refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        
        # # normalize?
        # q1 = self.prototypes(z1) # B*K
        # q2 = self.prototypes(z2) # B*K
        
        # y1, y2 = q1.argmax(dim=1), q2.argmax(dim=1)
        p1, y1 = torch.max(q1, dim=1)
        p2, y2 = torch.max(q2, dim=1)

        mask = (y1 == y2)
        
        conditional_mask = (y1.repeat(N,1) == y2.reshape(-1,1).repeat(1, N))
        conditional_mask += torch.eye(N, device=z1.device).bool() # enforce the diagnoal elements belong to the same cluster
        # print(conditional_mask.sum(dim=1), conditional_mask.shape)
        # between_sim[~conditional_mask] = EPS
        
        neg_sim = torch.sum(torch.mul(between_sim, conditional_mask), dim=1)
        
        ccl = -torch.log(
            between_sim.diag()
            / neg_sim)
        return ccl*mask
    
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