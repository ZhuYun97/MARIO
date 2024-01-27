import torch
import torch.nn as nn
from random import sample
from models.encoder import GCN_Encoder
from models.mlp import Two_MLP_BN, Two_MLP
from easyGOOD.utils.register import register
import copy


@register.model_register
class MOCO(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", dropout=0.5, use_bn=True, last_activation=False, **args_dicts):
        """
        dim: feature dimension (default: 128)
        r: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: softmax temperature 
        mlp: whether to use mlp projection
        """
        super(MOCO, self).__init__()
        assert args_dicts['tau'] > 0
        assert args_dicts['queue_size'] > 0
        assert args_dicts['mm'] <= 1 and args_dicts['mm'] > 0
        self.tau = args_dicts['tau']
        self.r = args_dicts['queue_size']
        self.m = args_dicts['mm']
        self.T = args_dicts['tau']

        # create the encoders
        # num_classes is the output fc dimension
        self.online_encoder = register.encoders[args_dicts['encoder_name']](input_dim, layer_num, hidden, activation, dropout, use_bn, last_activation)
        self.target_encoder = copy.deepcopy(self.online_encoder)

        self.online_projector = Two_MLP_BN(hidden, hidden, hidden) # w/o BN
        self.target_projector = Two_MLP_BN(hidden, hidden, hidden)

        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
        for param_q, param_k in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(hidden, self.r))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.contrastive_criterion = nn.CrossEntropyLoss()
        
        self.classifier = torch.nn.Linear(hidden, output_dim)

    @torch.no_grad()
    def _momentum_update_key_params(self):
        """
        Momentum update of the key encoder and projector
        """
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
        for param_q, param_k in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size < self.r:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            batch_size = self.r-ptr
            self.queue[:, ptr:] = keys[:batch_size].T
        ptr = (ptr + batch_size) % self.r  # move pointer


        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def pretrain(self, x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            is_eval: return momentum embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples
        Output:
            logits, targets, proto_logits, proto_targets
        """
        
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_params()  # update the key encoder and projector

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.target_encoder(x2, edge_index2, edge_weight2)  # keys: NxC
            k = self.target_projector(k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute query features
        q = self.online_encoder(x1, edge_index1, edge_weight1)  # queries: NxC
        q = self.online_projector(q)
        q = nn.functional.normalize(q, dim=1)
        
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: Nxr
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+r)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        
        loss = self.contrastive_criterion(logits, labels)
        
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
    
    def reset_classifier(self):
        self.classifier.reset_parameters()

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output