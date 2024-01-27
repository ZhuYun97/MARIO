import torch
from torch import nn

class Two_MLP_BN(torch.nn.Module):
    r"""
    Applies a non-linear transformation to contrastive space from representations.

        Args:
            hidden size of encoder, mlp hidden size, mlp output size
    """
    def __init__(self, hidden, mlp_hid, mlp_out):

        super(Two_MLP_BN, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden, mlp_hid), 
            nn.BatchNorm1d(mlp_hid),
            nn.ReLU(),
            nn.Linear(mlp_hid, mlp_out)
        )

    def forward(self, feat):
        return self.proj(feat)
    
class Two_MLP(nn.Module):
    r"""MLP used for predictor. The MLP has one hidden layer.

    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()