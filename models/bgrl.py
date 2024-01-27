import copy
from models.encoder import GCN_Encoder
import torch
from models.mlp import Two_MLP
from torch.nn.functional import cosine_similarity
from easyGOOD.utils.register import register


@register.model_register
class BGRL(torch.nn.Module):
    r"""BGRL architecture for Graph representation learning.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """
    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, use_bn=True, last_activation=False, **args_dicts):
        super().__init__()
        # online network
        self.online_encoder = register.encoders[args_dicts['encoder_name']](input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)
        self.predictor = Two_MLP(hidden, hidden, hidden)

        # target network
        self.target_encoder = copy.deepcopy(self.online_encoder)

        # reinitialize weights
        self.target_encoder.reset_parameters()
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.classifier = torch.nn.Linear(hidden, output_dim)

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())
    
    def reset_classifier(self):
        self.classifier.reset_parameters()

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def pretrain(self, x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2):
        # forward online network
        online_y1 = self.online_encoder(x1, edge_index1, edge_weight1)

        # prediction
        online_q1 = self.predictor(online_y1)

        # forward target network
        with torch.no_grad():
            target_y2 = self.target_encoder(x2, edge_index2, edge_weight2).detach()
            
        # calculate symmetrical part
        online_y2 = self.online_encoder(x2, edge_index2, edge_weight2)

        # prediction
        online_q2 = self.predictor(online_y2)

        # forward target network
        with torch.no_grad():
            target_y1 = self.target_encoder(x1, edge_index1, edge_weight1).detach()
            
        loss = 2 - cosine_similarity(online_q1, target_y2.detach(), dim=-1).mean() - cosine_similarity(online_q2, target_y1.detach(), dim=-1).mean()
        return loss
    
    def forward(self, x, edge_index, edge_weight=None, frozen=False):
        if frozen:
            with torch.no_grad():
                self.online_encoder.eval()
                out = self.online_encoder(x, edge_index, edge_weight)
        else:
            out = self.online_encoder(x, edge_index, edge_weight)
        out = self.classifier(out)
        return out


def load_trained_encoder(encoder, ckpt_path, device):
    r"""Utility for loading the trained encoder."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(checkpoint['model'], strict=True)
    return encoder.to(device)


def compute_representations(net, dataset, device):
    r"""Pre-computes the representations for the entire dataset.

    Returns:
        [torch.Tensor, torch.Tensor]: Representations and labels.
    """
    net.eval()
    reps = []
    labels = []

    for data in dataset:
        # forward
        data = data.to(device)
        with torch.no_grad():
            reps.append(net(data))
            labels.append(data.y)

    reps = torch.cat(reps, dim=0)
    labels = torch.cat(labels, dim=0)
    return [reps, labels]
