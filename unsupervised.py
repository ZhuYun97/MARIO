from easyGOOD.data.good_datasets.good_cora import GOODCora
from easyGOOD.data import create_dataloader, load_dataset
from easyGOOD.utils.initial import reset_random_seed
from easyGOOD.utils.config_reader import config_summoner
from easyGOOD.utils.args import args_parser
from easyGOOD.utils.logger import load_logger
from eval_utils import nan2zero_get_mask, evaluate, evaluate_all_with_scores, k_fold
from augmentation import drop_feature, adversarial_aug_train
from writer import write_all_in_pic, write_res_in_log, save_ckpts, load_ckpts
from optimizers import CosineDecayScheduler
from models import load_model

import torch
from tqdm import tqdm
from torch_geometric.utils import dropout_adj
from datetime import datetime
import numpy as np
from rich.progress import track


def train_best_linear_head(model, config, ood_train=False):
    model.reset_classifier()
    classifier_optimizer = torch.optim.Adam(model.classifier.parameters(), lr=config.train.linear_head_lr)
    best_id_val, best_id_test, best_ood_val, best_ood_test = 0, 0, 0, 0
    for e in track(range(config.train.linear_head_epochs)):
        for data in loader['train']:
            model.classifier.train()
            data = data.to(device)
            node_norm = data.get('node_norm') if config.model.model_level == 'node' else None
            node_norm = torch.ones(data.x.shape[0], device=data.x.device) if node_norm == None else node_norm
            edge_weight = data.get('edge_norm') if config.model.model_level == 'node' else None

            mask, targets = nan2zero_get_mask(data, 'train', config)
            preds = model(data.x, data.edge_index, edge_weight=None, frozen=True)

            loss = criterion(preds, targets) * mask
            loss = loss * node_norm * mask.sum() if config.model.model_level == 'node' else loss # normalization
            loss = loss.mean() / mask.sum()
            # loss = loss.sum() / mask.sum()
            loss.backward()
            classifier_optimizer.step()
            classifier_optimizer.zero_grad()
        # early stop
        train_acc, id_val, id_test, ood_val, ood_test = evaluate_all_with_scores(model, loader, criterion, config, device)
        if id_val > best_id_val:
            best_id_val = id_val
            best_id_test = id_test
        if ood_val > best_ood_val:
            best_ood_val = ood_val
            best_ood_test = ood_test
    return train_acc, best_id_val, best_id_test, best_ood_val, best_ood_test 

def train_linear_head(model, config, ood_train=False):
    model.reset_classifier()
    classifier_optimizer = torch.optim.Adam(model.classifier.parameters(), lr=config.train.linear_head_lr)
    for e in track(range(config.train.linear_head_epochs)):
        for data in loader['train']:
            model.classifier.train()
            data = data.to(device)
            node_norm = data.get('node_norm') if config.model.model_level == 'node' else None
            node_norm = torch.ones(data.x.shape[0], device=data.x.device) if node_norm == None else node_norm
            edge_weight = data.get('edge_norm') if config.model.model_level == 'node' else None
            # training mask
            mask, targets = nan2zero_get_mask(data, 'train', config)
            preds = model(data.x, data.edge_index, edge_weight=None, frozen=True) # you can also assign edge_weight

            loss = criterion(preds, targets) * mask
            loss = loss * node_norm * mask.sum() if config.model.model_level == 'node' else loss # normalization
            loss = loss.mean() / mask.sum()
            loss.backward()
            classifier_optimizer.step()
            classifier_optimizer.zero_grad()
            
            
def pretrain(data, model, config):
    model.train()
    
    # multi-views
    if config.model.model_name in ['GRACE', 'BGRL', 'G2CL', 'MOCO', 'PGCL', 'UNPMLP'
                                   , 'SWAV', 'MARIO', 'PROJ_GRACE', 'COSTA']:
        
        # node_norm = data.get('node_norm').to(device) if data.get('node_norm') != None else torch.ones(data.x.shape[0])

        # data augmentation, drop features and drop edges used in this repo
        x1, x2 = drop_feature(data.x, config.aug.mask_feat1), drop_feature(data.x, config.aug.mask_feat2)
        edge_index1, edge_norm1 = dropout_adj(edge_index=data.edge_index, p=config.aug.mask_edge1)
        edge_index2, edge_norm2 = dropout_adj(edge_index=data.edge_index, p=config.aug.mask_edge2)
        x1, edge_index1, x2, edge_index2 = x1.to(device), edge_index1.to(device), x2.to(device), edge_index2.to(device)

        # adversarial augmentation
        if config.aug.ad_aug:
            model.update_prototypes(x1=x1, edge_index1=edge_index1, edge_weight1=None, x2=x2, edge_index2=edge_index2, edge_weight2=None)
                
            def node_attack(perturb):
                x1_noise = x1 + perturb
                return model.pretrain(x1=x1_noise, edge_index1=edge_index1, edge_weight1=None, x2=x2, edge_index2=edge_index2, edge_weight2=None)

            loss = adversarial_aug_train(model, node_attack, x1.shape, 1e-3, 3, device)
        else:
            model.update_prototypes(x1=x1, edge_index1=edge_index1, edge_weight1=None, x2=x2, edge_index2=edge_index2, edge_weight2=None)
            loss = model.pretrain(x1=x1, edge_index1=edge_index1, edge_weight1=None, x2=x2, edge_index2=edge_index2, edge_weight2=None)
            
    # only one view
    elif config.model.model_name in ['GAE', 'VGAE', 'DGI', 'GraphMAE', "MVGRL"]:
        if config.aug.ad_aug:
            raise NotImplementedError(f'{config.model.model_name} can not use adversarial augmentation now!')
        # x, edge_index, edge_weight = data.x.to(device), data.edge_index.to(device), data.edge_norm.to(device)
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        if config.model.model_name == 'GraphMAE':
            loss = model.pretrain(data.to(device), x)
        elif config.model.model_name == 'MVGRL':
            loss = model.pretrain(data.to(device), x, edge_index)
        else:
            loss = model.pretrain(x, edge_index)
    else:
        raise NotImplementedError(f'{config.model.model_name} is not implemented!')
    return loss


if __name__ == '__main__':
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # args
    args = args_parser()
    config = config_summoner(args) 
    reset_random_seed(config)
    dataset = load_dataset(config.dataset.dataset_name, config)
    loader = create_dataloader(dataset, config) # use sampling graph instead
    # training process, used for training linear head
    if dataset.num_classes > 2: # multi-label classification
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    else: # binary classification
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    model = load_model(config.model.model_name, config).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    if config.model.model_name in ['MARIO']:
        params = []
        for k, v in model.named_parameters():
            if 'projector' in k or 'prototypes' in k:
                continue
            else:
                params.append(v)
        optimizer = torch.optim.Adam(params, lr=config.train.lr)
    
    
    if config.model.model_name == 'BGRL':
        lr_scheduler = CosineDecayScheduler(config.train.lr, config.train.warmup_steps, config.train.max_epoch)
        mm_scheduler = CosineDecayScheduler(1 - config.train.mm, 0, config.train.max_epoch)
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.train.mile_stones,
                                                        gamma=0.1)
    # load checkpoint if specified
    if config.model.load_checkpoint:
        load_ckpts(model, config)

    ebar = tqdm(range(1, config.train.max_epoch+1))
    best_id_val, best_id_id_test, best_id_ood_test, best_ood_val, best_ood_ood_test = 0, 0, 0, 0, 0
    train_acc, id_val, id_test, ood_val, ood_test = 0, 0, 0, 0, 0
    
    train_list, id_val_list, id_test_list, ood_val_list, ood_test_list = [], [], [], [], []
    
    if config.train.best_linear_head:
        print("Note: We will use the best linear head.")
    for e in ebar:
        if config.model.load_checkpoint: # use pre-trained model
            print("Load checkpoint, skip pre-training...")
            break
        # pbar = tqdm(enumerate(loader['train']), total=len(loader['train']))
        epoch_loss = 0
        epoch_node_cnt = 0
        # for index, data in pbar:
        for data in loader['train']:
            # if data.batch is not None and (data.batch[-1] < config.train.train_bs - 1):
            #     continue
            if config.model.model_name == 'BGRL':
                # update learning rate
                lr = lr_scheduler.get(e)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                # update momentum
                mm = 1 - mm_scheduler.get(e)
            
            optimizer.zero_grad()
            loss = pretrain(data, model, config)
            # if config.aug.ad_aug:
            #     optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update target network
            if config.model.model_name == 'BGRL':
                model.update_target_network(mm)
            else:
                lr_scheduler.step()
            epoch_loss += loss.detach().item() * data.x.shape[0]
            epoch_node_cnt += data.x.shape[0]
        
        # train the classifier and eval
        if e % config.train.eval_step == 0:
            if e == config.train.max_epoch:
                break # we will evaluate model out of the loop
            # ID linear head if ood_train is True else OOD linear head
            if config.dataset.ood_train_set:
                train_acc, ood_val , ood_test = train_eval_ood_linear_head(model, config, k=config.dataset.ood_split_fold)
                id_val, id_test = ood_val, 0
            else:
                if config.train.best_linear_head:
                    train_acc, id_val, id_test, ood_val, ood_test = train_best_linear_head(model, config)
                    # print(f"Epoch {e}, ID test: {id_test:.4f}, OOD test: {ood_test:.4f}")
                else:
                    train_linear_head(model, config)
                    # eval
                    train_acc, id_val, id_test, ood_val, ood_test = evaluate_all_with_scores(model, loader, criterion, config, device)
                    train_list.append(train_acc)
                    id_val_list.append(id_val)
                    id_test_list.append(id_test)
                    ood_val_list.append(ood_val)
                    ood_test_list.append(ood_test)
            
            # id val
            if id_val > best_id_val:
                best_id_val, best_id_id_test, best_id_ood_test = id_val, id_test, ood_test
            # ood val
            if ood_val > best_ood_val:
                best_ood_val, best_ood_ood_test = ood_val, ood_test
        
        # print(f'train acc: {train_acc} , id test: {id_test} ood test: {ood_test}')
        # ebar.set_postfix({'Train Loss': epoch_loss/epoch_node_cnt, 'train acc': train_acc,
        #                     'id val': id_val, 'id test': id_test,
        #                     'ood val': ood_val, 'ood test': ood_test})
        ebar.set_postfix({'train acc': train_acc,
                            'id test': id_test,
                            'ood test': ood_test})
        accs = [train_acc, id_val, id_test, ood_val, ood_test]
        write_all_in_pic(current_time, config, accs, e) # the information of tensorboard is recorded in /storage/tensorboard 
    ##############################################################################
    # evaluate out of the loop
    if config.dataset.ood_train_set:
        train_acc, ood_val , ood_test = train_eval_ood_linear_head(model, config, k=config.dataset.ood_split_fold)
        id_val, id_test = ood_val, 0
    else:
        if config.train.best_linear_head:
            train_acc, id_val, id_test, ood_val, ood_test = train_best_linear_head(model, config)
        else:
            train_linear_head(model, config)
            # eval
            train_acc, id_val, id_test, ood_val, ood_test = evaluate_all_with_scores(model, loader, criterion, config, device)
            train_list.append(train_acc)
            id_val_list.append(id_val)
            id_test_list.append(id_test)
            ood_val_list.append(ood_val)
            ood_test_list.append(ood_test)
    # id val
    if id_val > best_id_val:
        best_id_val, best_id_id_test, best_id_ood_test = id_val, id_test, ood_test
    # ood val
    if ood_val > best_ood_val:
        best_ood_val, best_ood_ood_test = ood_val, ood_test
    ################################################################################
    # save checkpoint
    if config.train.save_checkpoint:
        save_ckpts(model, config)
    print(f"\nFinal results: id-id: {best_id_id_test:.4f}, id-ood: {best_id_ood_test:.4f}, ood-ood: {best_ood_ood_test:.4f}")
    write_res_in_log([best_id_id_test, best_id_ood_test, best_ood_ood_test], config) # write results in /storage/log 
    
    tmp = torch.tensor([train_list, id_val_list, id_test_list, ood_val_list, ood_test_list])
    torch.save(tmp, f=f"./storage/records/result-{args.config_path.split('/')[5:]}-{config.random_seed}")