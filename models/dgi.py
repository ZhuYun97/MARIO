from torch_geometric.nn import DeepGraphInfomax
import torch
from .encoder import GCN_Encoder
from easyGOOD.utils.register import register


def summary(x, *args, **kwargs):
    x = x.mean(dim=0)
    return x

def corrupt(*args):
    x, edges, edge_weights = args
    size = x.shape[0]
    shuffle_idx = torch.randperm(size)
    x = x[shuffle_idx]
    return x, edges, edge_weights

@register.model_register
class DGI(DeepGraphInfomax):
    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, use_bn=True, last_activation=False, **args_dicts):
        encoder = register.encoders[args_dicts['encoder_name']](input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)
        super().__init__(hidden, encoder, summary, corrupt)
        self.classifier = torch.nn.Linear(hidden, output_dim)
    
    def forward(self, x, edge_index, edge_weight=None, frozen=False):
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                out = self.encoder(x, edge_index, edge_weight)
        else:
            out = self.encoder(x, edge_index, edge_weight)
        out = self.classifier(out)
        return out
    
    def pretrain(self, x, edge_index, edge_weight=None):
        pos_z = self.encoder(x, edge_index, edge_weight)
        cor = self.corruption(x, edge_index, edge_weight)
        neg_z = self.encoder(*cor)
        summary = self.summary(pos_z)
        loss = self.loss(pos_z, neg_z, summary)
        return loss
    
    def reset_classifier(self):
        self.classifier.reset_parameters()