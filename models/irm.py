from models.encoder import GCN_Encoder, GAT_Encoder
import torch
from easyGOOD.utils.register import register
from torch.autograd import grad
import numpy as np


@register.model_register
class IRM(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, use_bn=True, last_activation=False, **args_dicts):
        super(IRM, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden
        self.input_dim = input_dim

        # self.encoder = GCN_Encoder(input_dim, layer_num, hidden, activation)
        self.encoder = register.encoders[args_dicts['encoder_name']](input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)
        self.classifier = torch.nn.Linear(hidden, output_dim)
        
        # For IRM
        self.dummy_w = torch.nn.Parameter(torch.Tensor([1.0]))
    
    def forward(self, x, edge_index, edge_weight=None, frozen=False, **kwargs):
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                x = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        else:
            x = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.classifier(x)
        return x
    
    def output_postprocess(self, model_output):
        raw_pred = self.dummy_w * model_output
        return raw_pred
    
    def loss_postprocess(self, loss, data, mask, config):
        spec_loss_list = []
        for i in range(config.dataset.num_envs):
            env_idx = data.env_id == i
            if loss[env_idx].shape[0] > 0:
                grad_all = torch.sum(
                    grad(loss[env_idx].sum() / mask[env_idx].sum(), self.dummy_w, create_graph=True)[0].pow(2))
                spec_loss_list.append(grad_all)
        spec_loss = config.ood.ood_param * sum(spec_loss_list) / len(spec_loss_list)
        if torch.isnan(spec_loss):
            spec_loss = 0
        mean_loss = loss.sum() / mask.sum()
        loss = spec_loss + mean_loss
        # self.mean_loss = mean_loss
        # self.spec_loss = spec_loss
        return loss
    
    def reset_classifier(self):
        # for i in range(self.layer_num):
        #     self.convs[i].reset_parameters()
        #     self.bns[i].reset_parameters()
        # self.classifier.reset_parameters()
        torch.nn.init.xavier_uniform_(self.classifier.weight.data)
        torch.nn.init.constant_(self.classifier.bias.data, 0)
        
        
# class GCN_Classifier(torch.nn.Module):
#     def __init__(self, hidden, num_classes):
#         super(GCN_Classifier, self).__init__()
#         self.classifier = GCNConv(hidden, num_classes)