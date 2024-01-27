from models.encoder import GCN_Encoder, GAT_Encoder
import torch
from easyGOOD.utils.register import register
from torch.autograd import grad
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch import nn
from torch_geometric.data import Data


@register.model_register
class EERM(nn.Module):
    r"""
    EERM implementation adapted from https://github.com/qitianwu/GraphOOD-EERM.
    """
    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, use_bn=True, last_activation=False, **args_dicts):
        config = args_dicts['config']
        self.config = config
        super(EERM, self).__init__()
        # self.encoder = GCN_Encoder(config)
        self.encoder = register.encoders[args_dicts['encoder_name']](input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)
        self.p = 0.2
        self.K = config.ood.extra_param[0]
        self.T = config.ood.extra_param[1]
        self.num_sample = config.ood.extra_param[2]
        self.classifier = torch.nn.Linear(hidden, output_dim)

        self.gl = Graph_Editer(self.K, config.dataset.num_train_nodes, config.device)
        self.gl.reset_parameters()
        self.gl_optimizer = torch.optim.Adam(self.gl.parameters(), lr=config.ood.extra_param[3])

    def reset_parameters(self):
        self.encoder.reset_parameters()
        if hasattr(self, 'graph_est'):
            self.gl.reset_parameters()

    def forward(self, *args, **kwargs):
        data = kwargs.get('data')
        loss_func = self.config.metric.loss_func

        # --- K fold ---
        if self.training:
            edge_index, _ = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)
            x = data.x[data.train_mask]
            y = data.y[data.train_mask]

            # --- check will orig_edge_index change? ---
            orig_edge_index = edge_index
            for t in range(self.T):
                Loss, Log_p = [], 0
                for k in range(self.K):
                    edge_index, log_p = self.gl(orig_edge_index, self.num_sample, k)
                    raw_pred = self.classifier(self.encoder(x, edge_index))

                    loss = loss_func(raw_pred, y)
                    Loss.append(loss.view(-1))
                    Log_p += log_p
                Var, Mean = torch.var_mean(torch.cat(Loss, dim=0))
                reward = Var.detach()
                inner_loss = - reward * Log_p
                self.gl_optimizer.zero_grad()
                inner_loss.backward()
                self.gl_optimizer.step()
            return Var, Mean
        else:
            out = self.classifier(self.encoder(kwargs['x'], kwargs['edge_index']))
            return out
        
    def loss_postprocess(self, loss, data, mask, config):
        beta = 10 * config.ood.ood_param * config.train.epoch / config.train.max_epoch \
               + config.ood.ood_param * (1 - config.train.epoch / config.train.max_epoch)
        Var, Mean = loss
        loss = Var + beta * Mean
        self.mean_loss = Mean
        self.spec_loss = Var
        return loss
    
    def reset_classifier(self):
        # for i in range(self.layer_num):
        #     self.convs[i].reset_parameters()
        #     self.bns[i].reset_parameters()
        # self.classifier.reset_parameters()
        torch.nn.init.xavier_uniform_(self.classifier.weight.data)
        torch.nn.init.constant_(self.classifier.bias.data, 0)
        
    def output_postprocess(self, model_output):
        return model_output


class Graph_Editer(nn.Module):
    r"""
    EERM's graph editer adapted from https://github.com/qitianwu/GraphOOD-EERM.
    """
    def __init__(self, K, n, device):
        super(Graph_Editer, self).__init__()
        self.B = nn.Parameter(torch.FloatTensor(K, n, n))
        self.n = n
        self.device = device

    def reset_parameters(self):
        nn.init.uniform_(self.B)

    def forward(self, edge_index, num_sample, k):
        n = self.n
        Bk = self.B[k]
        A = to_dense_adj(edge_index, max_num_nodes=n)[0].to(torch.int)
        A_c = torch.ones(n, n, dtype=torch.int).to(self.device) - A
        P = torch.softmax(Bk, dim=0)
        S = torch.multinomial(P, num_samples=num_sample)  # [n, s]
        M = torch.zeros(n, n, dtype=torch.float).to(self.device)
        col_idx = torch.arange(0, n).unsqueeze(1).repeat(1, num_sample)
        M[S, col_idx] = 1.
        C = A + M * (A_c - A)
        edge_index = dense_to_sparse(C)[0]

        log_p = torch.sum(
            torch.sum(Bk[S, col_idx], dim=1) - torch.logsumexp(Bk, dim=0)
        )

        return edge_index, log_p
    
    

