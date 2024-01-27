from models.encoder import GCN_Encoder, GAT_Encoder
import torch
from easyGOOD.utils.register import register

@register.model_register
class ERM(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, use_bn=True, last_activation=False, **args_dicts):
        super(ERM, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden
        self.input_dim = input_dim

        # self.encoder = GCN_Encoder(input_dim, layer_num, hidden, activation)
        self.encoder = register.encoders[args_dicts['encoder_name']](input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)
        self.classifier = torch.nn.Linear(hidden, output_dim)
    
    def forward(self, x, edge_index, edge_weight=None, frozen=False, **kwargs):
        if frozen:
            with torch.no_grad():
                self.encoder.eval()
                x = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        else:
            x = self.encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        x = self.classifier(x)
        return x
    
    def reset_classifier(self):
        # for i in range(self.layer_num):
        #     self.convs[i].reset_parameters()
        #     self.bns[i].reset_parameters()
        # self.classifier.reset_parameters()
        torch.nn.init.xavier_uniform_(self.classifier.weight.data)
        torch.nn.init.constant_(self.classifier.bias.data, 0)
        
    def output_postprocess(self, model_output):
        return model_output
    
    def loss_postprocess(self, loss, data, mask, config):
        mean_loss = loss.sum() / mask.sum()
        return mean_loss
        
        
# class GCN_Classifier(torch.nn.Module):
#     def __init__(self, hidden, num_classes):
#         super(GCN_Classifier, self).__init__()
#         self.classifier = GCNConv(hidden, num_classes)