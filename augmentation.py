import torch

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def adversarial_aug_train(model, node_attack, perturb_shape, step_size, m, device):
    model.train()

    perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device)
    perturb.requires_grad_()
    
    loss = node_attack(perturb)
    loss /= m

    for i in range(m-1):
        loss.backward()
        perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        loss = node_attack(perturb)
        loss /=  m

    return loss