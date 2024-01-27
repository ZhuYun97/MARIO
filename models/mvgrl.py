from models.encoder import GCN_Encoder, PMLP_GCN, GAT_Encoder
from models.mlp import Two_MLP_BN
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Identity
from torch.nn import ModuleList
from easyGOOD.utils.register import register
from  torch_geometric.transforms import GDC
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import copy
import numpy as np
import networkx as nx
from scipy.linalg import fractional_matrix_power, inv
import scipy.sparse as sp
from torch_geometric.utils import to_networkx, from_scipy_sparse_matrix, remove_self_loops, to_undirected
from torch_sparse import SparseTensor
import os


def corrupt(*args):
    x, edges, edge_weights = args
    size = x.shape[0]
    shuffle_idx = torch.randperm(size)
    x = x[shuffle_idx]
    return x, edges, edge_weights

def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    diff_adj = alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1
    A = sp.csr_matrix(diff_adj)
    A = from_scipy_sparse_matrix(A)
    return A


@register.model_register
class MVGRL(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, use_bn=True, last_activation=False, **args_dicts) -> None:
        super().__init__()
        
        # self.encoder = GCN_Encoder(input_dim, layer_num, hidden, activation)
        # self.encoder = GAT_Encoder(input_dim, layer_num, hidden, activation)
        self.dataset_desc = args_dicts['dataset']
        self.encoder = register.encoders[args_dicts['encoder_name']](input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)
        self.encoder2 = Diff_Encoder(input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)
        # self.encoder2 = copy.deepcopy(self.encoder)
        self.read = Readout()
        self.sigm = torch.nn.Sigmoid()
        self.disc = Discriminator(hidden)
        # cache for diffusion graph
        self.diff_edge = None
        self.diff_weight = None
        self.diff_path = f"./storage/datasets/{self.dataset_desc['dataset_name']}/{self.dataset_desc['domain']}/processed"
        self.diff_name = f"diffusion_g.pt"
        self.diff_name = os.path.join(self.diff_path, self.diff_name)
        print(self.diff_name)
        if os.path.exists(self.diff_name):
            print('Loading precomputed diffusion graph...')
            tmp = torch.load(self.diff_name, map_location='cuda')
            self.diff_edge = tmp[0]
            self.diff_weight = tmp[1]
        self.classifier = torch.nn.Linear(hidden, output_dim)
        assert args_dicts['tau'] > 0
        self.tau = args_dicts['tau']
        self.b_xent = torch.nn.BCEWithLogitsLoss()
        
    def reset_classifier(self):
        self.classifier.reset_parameters()
        
    def forward(self, x, edge_index, edge_weight=None, frozen=False):
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                self.encoder2.eval()
                out = self.encoder(x, edge_index, edge_weight)
                # use off-the-shelf modules 
                if self.diff_edge is None:
                    print("It is the first time to use diffusion adjacent matrix. Computing...")
                    # view2: diffusion graph
                    data = Data(x=x, edge_index=edge_index)
                    nx_g = to_networkx(data)
                    # diff_g = self.gen(data)
                    diff_g = compute_ppr(nx_g)
                    self.diff_edge = diff_g[0].to(data.x.device)
                    self.diff_weight = diff_g[1].to(data.x.device)
                # diff_g = self.gen(g)
                out2 =self.encoder2(x, self.diff_edge, self.diff_weight.float())
                
                out = out + out2
        else:
            out = self.encoder(x, edge_index, edge_weight)
            if self.diff_edge is None:
                print("It is the first time to use diffusion adjacent matrix. Computing...")
                # view2: diffusion graph
                nx_g = to_networkx(data)
                # diff_g = self.gen(data)
                diff_g = compute_ppr(nx_g)
                self.diff_edge = diff_g[0].to(data.x.device)
                self.diff_weight = diff_g[1],float().to(data.x.device)
            out2 =self.encoder2(x, self.diff_edge, self.diff_weight.float())
            
            out = out + out2
        out = self.classifier(out)
        # out = self.classifier(out, edge_index)
        return out
    
    def pretrain(self, **kwargs):
        data, x, edge_index, edge_weight = kwargs['data'], kwargs['x'], kwargs['edge_index'], kwargs['edge_weight']
        # view1: original graph
        h1 = self.encoder(x, edge_index, edge_weight=edge_weight)
        c1 = self.read(h1)
        c1 = self.sigm(c1)
        if self.diff_edge is None:
            print("It is the first time to use diffusion adjacent matrix. Computing...")
            edge_index, _ = remove_self_loops(to_undirected(data.edge_index))
            data.edge_index = edge_index
            # view2: diffusion graph
            nx_g = to_networkx(data)
            # diff_g = self.gen(data)
            diff_g = compute_ppr(nx_g)
            torch.save(diff_g, self.diff_name)
            self.diff_edge = diff_g[0].to(data.x.device)
            self.diff_weight = diff_g[1].float().to(data.x.device)
        h2 = self.encoder2(x, self.diff_edge, self.diff_weight.float())
        c2 = self.read(h2)
        c2 = self.sigm(c2)
        
        # create  negative samples
        x_neg, _, _ = corrupt(x, edge_index, edge_weight)
        h3 = self.encoder(x_neg, edge_index, edge_weight)
        h4 = self.encoder2(x_neg, self.diff_edge, self.diff_weight.float())
        # z1 = self.projector(h1)
        # z2 = self.projector(h2)
        logits = self.disc(c1, c2, h1, h2, h3, h4)
        lbl_1 = torch.ones(h1.shape[0] * 2, device=h1.device)
        lbl_2 = torch.zeros(h1.shape[0] * 2, device=h1.device)
        lbl = torch.cat((lbl_1, lbl_2), 0)
        ret = self.b_xent(logits, lbl)
        return ret.mean()
    
    def update_prototypes(self, **kwargs):
        pass
 
    
# Borrowed from https://github.com/PetarV-/DGI
class Discriminator(torch.nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = torch.nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c1, c2, h1, h2, h3, h4, s_bias1=None, s_bias2=None):
        c_x1 = torch.unsqueeze(c1, 0)
        c_x1 = c_x1.expand_as(h1).contiguous()
        c_x2 = torch.unsqueeze(c2, 0)
        c_x2 = c_x2.expand_as(h2).contiguous()

        # positive
        sc_1 = torch.squeeze(self.f_k(h2, c_x1), 1)
        sc_2 = torch.squeeze(self.f_k(h1, c_x2), 1)

        # negetive
        sc_3 = torch.squeeze(self.f_k(h4, c_x1), 1)
        sc_4 = torch.squeeze(self.f_k(h3, c_x2), 1)
        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 0)
        return logits
    
    
# Borrowed from https://github.com/PetarV-/DGI
class Readout(torch.nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 0)
    
    
class Diff_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128, activation="relu", dropout=0.5, use_bn=True, last_activation=True):
        super(Diff_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden
        self.input_dim = input_dim
        self.activation = F.relu
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation

        self.convs = ModuleList()
        self.bns = ModuleList()
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(GCNConv(input_dim, hidden, add_self_loops=False, normalize=False)) 
            for i in range(layer_num-2):
                self.convs.append(GCNConv(hidden, hidden, add_self_loops=False, normalize=False))
                # glorot(self.convs[i].weight) # initialization
            self.convs.append(GCNConv(hidden, hidden, add_self_loops=False, normalize=False))
            # glorot(self.convs[-1].weight)
            for i in range(layer_num):
                if use_bn:
                    self.bns.append(BatchNorm1d(hidden))
                else:
                    self.bns.append(Identity())

        else: # one layer gcn
            self.convs.append(GCNConv(input_dim, hidden, add_self_loops=False, normalize=False)) 
            # glorot(self.convs[-1].weight)
            if use_bn:
                self.bns.append(BatchNorm1d(hidden))
            else:
                self.bns.append(Identity())
            # self.acts.append(self.activation) 
    
    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.layer_num):
            x = self.bns[i](self.convs[i](x, edge_index, edge_weight))
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