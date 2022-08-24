import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

'''https://github.com/facebookresearch/mixup-cifar10'''

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_step(model,data_loader,opt=None,atk_method=None,*args):

    if opt:
        model.train()
    else:
        model.eval()

    total_loss, total_correct = 0., 0.
    total_param_grad_norm = 0.

    for X, Y in tqdm(data_loader):
        X, Y = X.cuda(), Y.cuda()

        if atk_method:
            delta = atk_method(model, X, Y, *args)
        else:
            delta = 0.

        X = X + delta

        X, Y_a, Y_b, lam = mixup_data(X, Y)
        X, Y_a, Y_b = map(torch.autograd.Variable, (X, Y_a, Y_b))

        Y_pred = model(X)
        loss = mixup_criterion(nn.CrossEntropyLoss(), Y_pred, Y_a, Y_b, lam)

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

            diff_norm_list = []
            for name, parameter in model.named_parameters():
                diff_norm_list.append(torch.norm(parameter.grad, p='fro').cpu().numpy())
            diff_norm = np.linalg.norm(np.array(diff_norm_list))
            total_param_grad_norm += diff_norm * X.shape[0]
        
        total_loss += loss.item() * X.shape[0]
        total_correct += lam * (Y_pred.max(dim=1)[1] == Y_a).sum().item() + (1 - lam) * (Y_pred.max(dim=1)[1] == Y_b).sum().item()

    avg_loss = total_loss / len(data_loader.dataset)
    avg_acc = total_correct / len(data_loader.dataset) * 100
    avg_param_grad_norm = total_param_grad_norm / len(data_loader.dataset)

    return avg_loss, avg_acc, avg_param_grad_norm

