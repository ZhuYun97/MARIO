from torch.utils.tensorboard import SummaryWriter
import os
from easyGOOD.definitions import STORAGE_DIR
import torch

def write_all_in_pic(current_time, config, accs, epoch):
    assert len(accs) == 5
    acc_keys = ['train', 'id_val', 'id_test', 'ood_val', 'ood_test']
    for i, key in enumerate(acc_keys):
        with SummaryWriter(log_dir=os.path.join(config.tensorboard_logdir, f'{config.log_file}_{current_time}/{key}')) as writer:
            writer.add_scalar('Acc', accs[i], epoch)
            
def write_res_in_log(res, config):
    assert len(res) == 3
    best_id_id_test, best_id_ood_test, best_ood_ood_test = res
    # record results in log files
    log_path = os.path.join(STORAGE_DIR, f"log/{config.dataset.dataset_name}/{config.dataset.domain}-{config.dataset.shift_type}")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(os.path.join(log_path, f"{config.model.model_name}-{config.model.encoder_name}.log"))
    if not os.path.exists(log_file):
        os.system(f'touch {log_file}')
    with open(log_file, 'a') as f:
        if config.random_seed == 0:
            f.write("=====================\n")
        f.write(f"Inductive: {config.dataset.inductive}, OOD train: {config.dataset.ood_train_set}-{config.dataset.ood_split_fold}, Epochs: {config.train.max_epoch} Lr: {config.train.lr} Tau: {config.model.tau} MaskFeat 1,2: {config.aug.mask_feat1, config.aug.mask_feat2} MaskEdge1: {config.aug.mask_edge1, config.aug.mask_edge2} Ad aug: {config.aug.ad_aug} Dropout: {config.model.dropout_rate} use_bn: {config.model.use_bn} last_act: {config.model.last_activation}")
        f.write(f"---lin epochs {config.train.linear_head_epochs} lin lr {config.train.linear_head_lr} eval step {config.train.eval_step}")
        if config.model.model_name == 'BRGL':
            f.write(f" mm: {config.train.mm} warmup: {config.train.warmup_steps}")
        elif config.model.model_name == 'REGCL':
            f.write(f" num_clusters: {config.model.num_clusters} pro_lr: {config.model.prototypes_lr} pro_iters: {config.model.prototypes_iters} cmi_cof: {config.model.cmi_coefficient}")
        f.write("\n")
        
        f.write(f"Results: {best_id_id_test*100:.2f} {best_id_ood_test*100:.2f} {best_ood_ood_test*100:.2f}\n")
        
def save_ckpts(model, config):
    inductive = f'inductive' if config.dataset.inductive else "transudctive"
    # ln_head = f'ood-head-{config.dataset.ood_split_fold}' if config.dataset.ood_train_set else 'id-head'
    # ln_head = f'ood-head' if config.dataset.ood_train_set else 'id-head' # ood-head is used for evaluate
    ckpt_path = os.path.join(STORAGE_DIR, f"checkpoints/{config.dataset.dataset_name}/{config.dataset.domain}-{config.dataset.shift_type}/{inductive}/{config.model.model_name}/{config.model.encoder_name}")
    args = f"Epochs:{config.train.max_epoch}-Lr:{config.train.lr}-Tau:{config.model.tau}-MaskFeat:{config.aug.mask_feat1, config.aug.mask_feat2}-MaskEdge:{config.aug.mask_edge1, config.aug.mask_edge2}-Ad_aug:{config.aug.ad_aug}"
    ckpt_path = os.path.join(ckpt_path, args)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    
    file_name = os.path.join(ckpt_path, f'{config.random_seed}.pth')
    torch.save(model.state_dict(), file_name)
    print(f"Save model to {file_name}")
    
def load_ckpts(model, config):
    inductive = f'inductive' if config.dataset.inductive else "transudctive"
    # ln_head = f'ood-head-{config.dataset.ood_split_fold}' if config.dataset.ood_train_set else 'id-head'
    # ln_head = f'ood-head' if config.dataset.ood_train_set else 'id-head'
    ckpt_path = os.path.join(STORAGE_DIR, f"checkpoints/{config.dataset.dataset_name}/{config.dataset.domain}-{config.dataset.shift_type}/{inductive}/{config.model.model_name}/{config.model.encoder_name}")
    args = f"Epochs:{config.train.max_epoch}-Lr:{config.train.lr}-Tau:{config.model.tau}-MaskFeat:{config.aug.mask_feat1, config.aug.mask_feat2}-MaskEdge:{config.aug.mask_edge1, config.aug.mask_edge2}-Ad_aug:{config.aug.ad_aug}/{config.random_seed}.pth"
    file_name = os.path.join(ckpt_path, args)
    
    model.load_state_dict(torch.load(file_name))