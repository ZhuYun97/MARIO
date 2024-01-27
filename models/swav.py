from models.encoder import GCN_Encoder, PMLP_GCN
from models.mlp import Two_MLP_BN, Two_MLP
import torch
import torch.nn.functional as F
from easyGOOD.utils.register import register


def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result

@register.model_register
class SWAV(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, use_bn=True, last_activation=False, **args_dicts) -> None:
        super().__init__()
        self.encoder = register.encoders[args_dicts['encoder_name']](input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)
        # self.encoder = GCN_Encoder(input_dim, layer_num, hidden, activation)
        self.projector =Two_MLP_BN(hidden, hidden, hidden)
        self.classifier = torch.nn.Linear(hidden, output_dim)
        # self.classifier = GCNConv(hidden, output_dim)
        assert args_dicts['tau'] > 0
        self.tau = args_dicts['tau']
        # cora use 1000,
        self.prototypes = torch.nn.Linear(hidden, 1000, bias=False) # weight shape: K*D
        self.init_emb()
        # normalize for prototypes
        
    def reset_classifier(self):
        self.classifier.reset_parameters()
        
    def init_emb(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
        
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
            # q1 = self._batched_semi_emd_loss(scores1)
            # q2 = self._batched_semi_emd_loss(scores2)
            
        # p1 = F.softmax(scores1 / self.tau, dim=0)
        # p2 = F.softmax(scores2 / self.tau, dim=0)
        
        loss = -0.5*(q1*F.log_softmax(scores2/self.tau, dim=1) + q2*F.log_softmax(scores1/self.tau, dim=1)) 
        # + 0.1*self.constraint() # need to add orthognal loss
        loss = torch.sum(loss, dim=1)

        # l1 = self.semi_loss(z1, z2)
        # l2 = self.semi_loss(z2, z1)
        # contra_loss = (l1 + l2) * 0.5
        return loss.mean()
        
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
    
    def _batched_semi_emd_loss(self, scores, lamb=20):
        # assert out1.shape[0] == out2.shape[0] and avg_out1.shape == avg_out2.shape

        # x reps from sparse to dense
        cost_matrix = 1-scores # (N,K)
        N, K = cost_matrix.shape
        cost_matrix = cost_matrix.unsqueeze(0)

        # Sinkhorn iteration
        iter_times = 5
        with torch.no_grad():
            # u, r, c = zeros(K), ones(K) / K, ones(B) / B
            r = torch.ones(size=(1, N, 1), device=cost_matrix.device) / N
            c = torch.ones(size=(1, K, 1), device=cost_matrix.device) / K
            # r = torch.bmm(out1, avg_out2.transpose(1,2)) # (B,N,1), normalize? otherwise the accumulated pd does not equal to one
            # r[r<=0] = 1e-8 # max(ri, 0)
            # r = r / r.sum(dim=1, keepdim=True) # the accumulated pd equals to one, or we can use softmax
            # # r = (torch.ones_like(r)/r.shape[1]) # uniform marginal weights
            # c = torch.bmm(out2, avg_out1.transpose(1,2)) # (B,M,1), normalize?
            # c[c<=0] = 1e-8 # max(ci, 0)
            # c = c / c.sum(dim=1, keepdim=True)
            # # c = (torch.ones_like(c)/c.shape[1]) # uniform
            P = torch.exp(-1*lamb*cost_matrix) # (B,N,K)
            u = (torch.ones_like(c)/c.shape[1]) # (B,M,1)
            for i in range(iter_times):
                v = torch.div(r, torch.bmm(P, u))
                u = torch.div(c, torch.bmm(P.transpose(1,2), v))
            u = u.squeeze(dim=-1)
            v = v.squeeze(dim=-1)
            transport_matrix = torch.bmm(torch.bmm(matrix_diag(v), P), matrix_diag(u))
        assert cost_matrix.shape == transport_matrix.shape
        transport_matrix = transport_matrix.squeeze(0)
        transport_matrix /= transport_matrix.sum(dim=1, keepdim=True)
        # S = torch.mul(transport_matrix, 1-cost_matrix).sum(dim=1).sum(dim=1, keepdim=True) #
        # loss_emd = 2-2*S # maybe mean function should not be used here
        return transport_matrix
    