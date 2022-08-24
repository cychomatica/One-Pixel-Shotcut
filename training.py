import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

def step(model,data_loader,opt=None,atk_method=None,*args):
    
    if opt:
        model.train()
    else:
        model.eval()

    total_loss, total_error = 0., 0.
    total_param_grad_norm = 0.

    for X, y in tqdm(data_loader):
        X, y = X.cuda(), y.cuda()

        if atk_method:
            delta = atk_method(model, X, y, *args)
        else:
            delta = 0.

        y_pred = model(X + delta)
        loss = nn.CrossEntropyLoss()(y_pred, y)

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

            diff_norm_list = []
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    diff_norm_list.append(torch.norm(parameter.grad, p='fro').cpu().numpy())
            diff_norm = np.linalg.norm(np.array(diff_norm_list))
            total_param_grad_norm += diff_norm * X.shape[0]
        
        total_loss += loss.item() * X.shape[0]
        total_error += (y_pred.max(dim=1)[1] != y).sum().item()

    avg_loss = total_loss / len(data_loader.dataset)
    avg_acc = 100 - total_error / len(data_loader.dataset) * 100
    avg_param_grad_norm = total_param_grad_norm / len(data_loader.dataset)

    return avg_loss, avg_acc, avg_param_grad_norm