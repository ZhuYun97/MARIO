from models.encoder import GCN_Encoder
from models.mlp import Two_MLP_BN
import torch
import torch.nn.functional as F
from easyGOOD.utils.register import register
from models.moco import MOCO
from random import sample
# import faiss
import torch.nn as nn
import numpy as np


@register.model_register
class PGCL(MOCO):
    def __init__(self, input_dim, layer_num=2, hidden=128, output_dim=70, activation="relu", **args_dicts) -> None:
        super().__init__(input_dim, layer_num, hidden, output_dim, activation, **args_dicts)
        # self.num_cluster = [10] if not args_dicts['num_cluster'] else args_dicts['num_cluster'] # array
        self.num_cluster = [10]
        # self.proto = torch.nn.Parameter(torch.normal(mean=0, std=0.01, size=(10, hidden)))
        
    def forward(self, x, edge_index, edge_weight=None, frozen=False):
        if frozen:
            with torch.no_grad():
                self.online_encoder.eval()
                out = self.online_encoder(x, edge_index, edge_weight)
        else:
            out = self.online_encoder(x, edge_index, edge_weight)
        out = self.classifier(out)
        return out
    
    def pretrain(self, x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2, cluster_result=None, index=None):
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
            k = torch.nn.functional.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute query features
        q = self.online_encoder(x1, edge_index1, edge_weight1)  # queries: NxC
        q = self.online_projector(q)
        q = torch.nn.functional.normalize(q, dim=1)
        
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
        
        # prototypical contrast
        if cluster_result is not None:  
            proto_labels = []
            proto_logits = []
            for n, (im2cluster,prototypes,density) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'],cluster_result['density'])):
                # get positive prototypes
                pos_proto_id = im2cluster[index]
                pos_prototypes = prototypes[pos_proto_id]    
                
                # sample negative prototypes
                all_proto_id = [i for i in range(im2cluster.max()+1)]       
                neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist())
                neg_proto_id = sample(neg_proto_id,self.r) #sample r negative prototypes 
                neg_prototypes = prototypes[neg_proto_id]    

                proto_selected = torch.cat([pos_prototypes,neg_prototypes],dim=0)
                
                # compute prototypical logits
                logits_proto = torch.mm(q,proto_selected.t())
                
                # targets for prototype assignment
                labels_proto = torch.linspace(0, q.size(0)-1, steps=q.size(0)).long().cuda()
                
                # scaling temperatures for the selected prototypes
                temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).cuda()],dim=0)]  
                logits_proto /= temp_proto
                
                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)
            
            for proto_out,proto_target in zip(proto_logits, proto_labels):
                loss_proto += self.contrastive_criterion(proto_out, proto_target)  
                loss_proto /= len(self.num_cluster)
            
            loss += loss_proto
        return loss
        
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
    

    def pcl_loss(self, v_ins, pad_mask, tau_ii=1, tau_ip=1, alpha=1):
        loss = 0.
        num_classes, num, dim = v_ins.shape # C, num, D
        m = 0.2
        # instance-prototype loss  
        sim_mat = torch.exp(self.pcl_sim(v_ins, self.proto)/tau_ip)
        # print(self.pcl_sim(v_ins, self.proto)[:,0,:])
        # sim_mat = torch.exp(self.pcl_sim(v_ins, self.projection(self.proto))/tau_ip)
        num = sim_mat.shape[1]
        
        for i in range(num):
            pos_score = torch.diag(sim_mat[:,i,:]).pow(alpha)
            neg_score = (sim_mat[:,i,:].pow(alpha).sum(1) - pos_score)
            loss += - (torch.log(pos_score / (pos_score+neg_score))).sum() # mask
        loss = loss / (num * self.num_classes * self.num_classes)

        # instance-instance loss
        loss_ins = 0.
        for i in range(v_ins.shape[0]):
            sim_instance = torch.exp(self.pcl_sim(v_ins, v_ins[i])/tau_ii)
            pos_ins = sim_instance[i].pow(alpha)
            neg_ins = (sim_instance.pow(alpha).sum(0) - pos_ins).sum(0)
            loss_ins += - torch.log(pos_ins / (pos_ins+neg_ins)).sum()
        loss_ins = loss_ins / (num * self.num_classes * num * self.num_classes)

        loss = loss*2 + loss_ins

        return loss


def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset),args.low_dim).cuda()
    for i, (images, index) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            feat = model(images,is_eval=True) 
            features[index] = feat 
    return features.cpu()
    

def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}
    
    for seed, num_cluster in enumerate(args.num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = args.gpu    
        index = faiss.GpuIndexFlatL2(res, d, cfg)  

        clus.train(x, index)   

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
        
        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        density = args.temperature*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    

        im2cluster = torch.LongTensor(im2cluster).cuda()               
        density = torch.Tensor(density).cuda()
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results