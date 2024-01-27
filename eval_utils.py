import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, KFold


def nan2zero_get_mask(data, task, config):
    r"""
    Training data filter masks to process NAN.

    Args:
        data (Batch): input data
        task (str): mask function type
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.model_level`)

    Returns (Tensor):
        [mask (Tensor) - NAN masks for data formats, targets (Tensor) - input labels]

    """
    if config.model.model_level == 'node':
        # if 'train' in task:
        if task in ['train', 'eval_train']:
            mask = data.train_mask
        elif task == 'id_val':
            mask = data.get('id_val_mask')
        elif task == 'id_test':
            mask = data.get('id_test_mask')
        elif task == 'val':
            mask = data.val_mask
        elif task == 'test':
            mask = data.test_mask
        elif task == 'ood_train':
            assert config.dataset.ood_train_set
            mask = data.ood_train_mask
        else:
            raise ValueError(f'Task should be train/id_val/id_test/val/test, but got {task}.')
    else:
        mask = ~torch.isnan(data.y)
    if mask is None:
        return None, None
    targets = torch.clone(data.y).detach()
    targets[~mask] = 0

    return mask, targets

def eval_score(pred_all, target_all, config):
    r"""
    Calculate metric scores given preprocessed prediction values and ground truth values.

    Args:
        pred_all (Union[List[np.ndarray], List[List[np.ndarray]]]): Prediction value list. It is a list of output pred
            of :func:`eval_data_preprocess`.
        target_all (Union[List[np.ndarray], List[List[np.ndarray]]]): Ground truth value list. It is a list of output
            target of :func:`eval_data_preprocess`.
        config (Union[CommonArgs, Munch]): The required config is ``config.metric.score_func`` that is a function for
            score calculation (*e.g.*, :func:`GOOD.utils.metric.Metric.acc`).

    Returns:
        A float score value.
    """
    np.seterr(invalid='ignore')
    assert type(pred_all) is list, 'Wrong prediction input.'
    if type(pred_all[0]) is list:
        # multi-task
        all_task_preds = []
        all_task_targets = []
        for task_i in range(len(pred_all[0])):
            preds = []
            targets = []
            for pred, target in zip(pred_all, target_all):
                preds.append(pred[task_i])
                targets.append(target[task_i])
            all_task_preds.append(np.concatenate(preds))
            all_task_targets.append(np.concatenate(targets))

        scores = []
        for i in range(len(all_task_preds)):
            if all_task_targets[i].shape[0] > 0:
                scores.append(np.nanmean(config.metric.score_func(all_task_targets[i], all_task_preds[i])))
        score = np.nanmean(scores)
    else:
        pred_all = np.concatenate(pred_all)
        target_all = np.concatenate(target_all)
        # print(pred_all)
        score = np.nanmean(config.metric.score_func(target_all, pred_all))
    return score

def eval_data_preprocess(y, raw_pred, mask, config):
    r"""
    Preprocess data for evaluations by converting data into np.ndarray or List[np.ndarray] (Multi-task) format.
    When the task of the dataset is not multi-task, data is converted into np.ndarray.
    When it is multi-task, data is converted into List[np.ndarray] in which each np.ndarray in the list represents
    one task. For example, GOOD-PCBA is a 128-task binary classification dataset. Therefore, the output list will
    contain 128 elements.

    Args:
        y (torch.Tensor): Ground truth values.
        raw_pred (torch.Tensor): Raw prediction values without softmax or sigmoid.
        mask (torch.Tensor): Ground truth NAN mask for removing empty label.
        config (Union[CommonArgs, Munch]): The required config is
            ``config.metric.dataset_task``

    Returns:
        Processed prediction values and ground truth values.

    """
    if config.metric.dataset_task == 'Binary classification':
        pred_prob = raw_pred.sigmoid()
        if y.shape[1] > 1:
            # multi-task
            preds = []
            targets = []
            for i in range(y.shape[1]):
                # pred and target per task
                preds.append(pred_prob[:, i][mask[:, i]].detach().cpu().numpy())
                targets.append(y[:, i][mask[:, i]].detach().cpu().numpy())
            return preds, targets
        pred = pred_prob[mask].reshape(-1).detach().cpu().numpy()
    elif config.metric.dataset_task == 'Multi-label classification':
        pred_prob = raw_pred.softmax(dim=1)
        pred = pred_prob[mask].detach().cpu().numpy()
    elif 'Regression' in config.metric.dataset_task:
        pred = raw_pred[mask].reshape(-1).detach().cpu().numpy()
    else:
        raise ValueError('Dataset task value error.')

    target = y[mask].reshape(-1).detach().cpu().numpy()

    return pred, target        
# testing
@torch.no_grad()
def evaluate(model, loader, split, criterion, config, device):
    r"""
    This function is design to collect data results and calculate scores and loss given a dataset subset.
    (For project use only)

    Args:
        split (str): A split string for choosing the corresponding dataloader. Allowed: 'train', 'id_val', 'id_test',
            'val', and 'test'.

    Returns:
        A score and a loss.

    """
    stat = {'score': None, 'loss': None}
    if loader.get(split) is None:
        return stat
    model.eval()

    loss_all = []
    mask_all = []
    pred_all = []
    target_all = []
    # pbar = tqdm(loader[split], desc=f'Eval {split.capitalize()}', total=len(loader[split]))
    # for data in pbar:
    for data in loader[split]:
        data = data.to(device)

        mask, targets = nan2zero_get_mask(data, split, config)
        if mask is None:
            return stat
        node_norm = torch.ones((data.num_nodes,),
                                device=device) if config.model.model_level == 'node' else None
        edge_weight = data.get('edge_norm') if config.model.model_level == 'node' else None
        raw_preds = model(x=data.x, edge_index=data.edge_index, edge_weight=None, frozen=config.evaluation.frozen)

        # --------------- Loss collection ------------------
        loss: torch.tensor = criterion(raw_preds, targets) * mask
        mask_all.append(mask)
        loss_all.append(loss)

        # ------------- Score data collection ------------------
        pred, target = eval_data_preprocess(data.y, raw_preds, mask, config)
        # pred = raw_preds.cpu()
        # target = targets.cpu()
        pred_all.append(pred)
        target_all.append(target)

    # ------- Loss calculate -------
    loss_all = torch.cat(loss_all)
    mask_all = torch.cat(mask_all)
    stat['loss'] = loss_all.sum() / mask_all.sum()

    # --------------- Metric calculation including ROC_AUC, Accuracy, AP.  --------------------
    stat['score'] = eval_score(pred_all, target_all, config)

    # print(f'#IN#\n{split.capitalize()} {config.metric.score_name}: {stat["score"]:.4f}\n'
    #         f'{split.capitalize()} Loss: {stat["loss"]:.4f}')

    model.train()

    return {'score': stat['score'], 'loss': stat['loss']}

def evaluate_all_with_scores(model, loader, criterion, config, device):
    epoch_train_stat = evaluate(model, loader, 'eval_train', criterion, config, device)
    id_val_stat = evaluate(model, loader, 'id_val', criterion, config, device)
    id_test_stat = evaluate(model, loader, 'id_test', criterion, config, device)
    val_stat = evaluate(model, loader, 'val', criterion, config, device)
    test_stat = evaluate(model, loader, 'test', criterion, config, device)
    return epoch_train_stat['score'], id_val_stat['score'], id_test_stat['score'], val_stat['score'], test_stat['score']


def k_fold(data, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)
    # skf = KFold(n_splits=folds, shuffle=True, random_state=12345)
    labels = data.y[data.test_mask]
    from collections import Counter
    frequence = Counter(labels.numpy().tolist())
    # print(frequence)
    test_idx = torch.nonzero(data.test_mask).squeeze()

    test_indices, val_indices, train_indices = [], [], []
    for relative_test_idx, relative_train_idx in skf.split(torch.zeros(len(labels)), labels):
        train_mask = torch.zeros(data.y.shape[0], device=data.y.device, dtype=torch.bool)
        val_mask = torch.zeros(data.y.shape[0], device=data.y.device, dtype=torch.bool)
        test_mask = torch.zeros(data.y.shape[0], device=data.y.device, dtype=torch.bool)
        
        abs_train_idx = test_idx[relative_train_idx]
        abs_val_idx = test_idx[relative_test_idx]
        abs_test_idx = test_idx[relative_test_idx]
        
        train_mask[abs_train_idx] = True
        val_mask[abs_val_idx] = True
        test_mask[abs_test_idx] = True
        
        train_indices.append(train_mask)
        val_indices.append(val_mask)
        test_indices.append(test_mask)
        # print("train", train_mask.sum(), test_mask, test_mask.sum())
    return train_indices, val_indices, test_indices