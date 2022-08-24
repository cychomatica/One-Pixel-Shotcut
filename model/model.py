import torch.nn as nn
import numpy as np


def summary(model:nn.Module):

    total_params = 0

    for x in filter(lambda p: p.requires_grad, model.parameters()):
        total_params += np.prod(x.data.cpu().numpy().shape)

    total_layers = len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, model.parameters())))

    try:
        print('Model name:', model.model_name)
    except:
        print('Model name:','Unknown')
        
    print('Total number of params:', total_params)
    print('Total layers:', total_layers)

    # return model.model_name, total_params, total_layers