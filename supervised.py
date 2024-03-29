from easyGOOD.data.good_datasets.good_cora import GOODCora
from easyGOOD.data import create_dataloader, load_dataset
from easyGOOD.utils.initial import reset_random_seed
from easyGOOD.utils.config_reader import config_summoner
from easyGOOD.utils.args import args_parser
from easyGOOD.utils.logger import load_logger
from eval_utils import nan2zero_get_mask, evaluate_all_with_scores, k_fold
from models import GCN, GAT, GCN_Encoder, GRACE, load_sup_model
from writer import write_all_in_pic, write_res_in_log

import torch
from tqdm import tqdm
import numpy as np


def train_linear_head(model, epoch, config, ood_train=False):
    model.reset_classifier()
    model.classifier.train()
    for e in range(epoch):
        classifier_optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
        for data in loader['train']:
            model.classifier.train()
            data = data.to(device)
            node_norm = data.get('node_norm') if config.model.model_level == 'node' else None
            edge_weight = data.get('edge_norm') if config.model.model_level == 'node' else None
            if ood_train:
                assert config.dataset.ood_train_set
                mask, targets = nan2zero_get_mask(data, 'ood_train', config) # select 10% from OOD test set as training set for OOD linear head
            else:
                mask, targets = nan2zero_get_mask(data, 'train', config)
            # preds = model(data=data, edge_weight=edge_weight)
            preds = model(data.x, data.edge_index, edge_weight=edge_weight, frozen=True)

            loss = criterion(preds, targets) * mask
            loss = loss * node_norm * mask.sum() if config.model.model_level == 'node' and not config.dataset.inductive else loss
            loss = loss.mean() / mask.sum()
            # loss = loss.sum() / mask.sum()
            loss.backward()
            classifier_optimizer.step()
            classifier_optimizer.zero_grad()
            
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # args
    args = args_parser()
    config = config_summoner(args)
    reset_random_seed(config)

    dataset = load_dataset(config.dataset.dataset_name, config)
    loader = create_dataloader(dataset, config)
    # training process
    if dataset.num_classes > 2: # multi-label classification
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
    else: # binary classification
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        
    # criterion = config.metric.loss_func
    model = load_sup_model(config.model.model_name, config).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.train.mile_stones,
                                                        gamma=0.1)

    # trainng
    ebar = tqdm(range(1, config.train.max_epoch+1))
    best_id_val, best_id_id_test, best_id_ood_test, best_ood_val, best_ood_ood_test = 0, 0, 0, 0, 0
    
    train_list, id_val_list, id_test_list, ood_val_list, ood_test_list = [], [], [], [], []
    for e in ebar:
        # pbar = tqdm(enumerate(loader['train']), total=len(loader['train']))
        epoch_loss = 0
        epoch_node_cnt = 0
        # for index, data in pbar:
        for data in loader['train']:
            optimizer.zero_grad()
            model.train()
            data = data.to(device)
            node_norm = data.get('node_norm') if config.model.model_level == 'node' else None
            edge_weight = data.get('edge_norm') if config.model.model_level == 'node' else None

            mask, targets = nan2zero_get_mask(data, 'train', config)
            preds = model(x=data.x, edge_index=data.edge_index, edge_weight=edge_weight, data=data)
            preds = model.output_postprocess(preds)
            if config.model.model_name != 'EERM':
                loss = criterion(preds, targets) * mask
                loss = loss * node_norm * mask.sum() if (config.model.model_level == 'node' and not config.dataset.inductive) else loss
            else:
                loss = preds
            loss = model.loss_postprocess(loss, data, mask, config)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item() * mask.sum().item()
            epoch_node_cnt += mask.sum().item()
        
        scheduler.step()
        
        # train the linear classifier and eval
        if (e % config.train.eval_step == 0) or (e == config.train.max_epoch):
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
                
            ebar.set_postfix({'Train Loss': epoch_loss/epoch_node_cnt, 'train acc': train_acc,
                                'id val': id_val, 'id test': id_test,
                                'ood val': ood_val, 'ood test': ood_test})
            accs = [train_acc, id_val, id_test, ood_val, ood_test]
        # write_all_in_pic(current_time, config, accs, e) # the information of tensorboard is recorded in /storage/tensorboard 
        
    print(f"\nFinal results: id-id: {best_id_id_test:.4f}, id-ood: {best_id_ood_test:.4f}, ood-ood: {best_ood_ood_test:.4f}")
    write_res_in_log([best_id_id_test, best_id_ood_test, best_ood_ood_test], config) # write results in /storage/log 
