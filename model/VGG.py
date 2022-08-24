'''
VGG Repository: 
https://github.com/chengyangfu/pytorch-vgg-cifar10
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, model_name='VGG', num_classes=10):
        super(VGG, self).__init__()

        self.model_name = model_name

        self.features = features

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}

def VGG11(batch_norm=True, num_classes=10):
    '''VGG 11-layer model (configuration "A")'''
    return VGG(make_layers(cfg['A'], batch_norm=batch_norm), 'VGG11', num_classes)

def VGG13(batch_norm=True, num_classes=10):
    '''VGG 13-layer model (configuration "B")'''
    return VGG(make_layers(cfg['B'], batch_norm=batch_norm), 'VGG13', num_classes)

def VGG16(batch_norm=True, num_classes=10):
    '''VGG 16-layer model (configuration "D")'''
    return VGG(make_layers(cfg['D'], batch_norm=batch_norm), 'VGG16', num_classes)

def VGG19(batch_norm=True, num_classes=10):
    '''VGG 19-layer model (configuration 'E')'''
    return VGG(make_layers(cfg['E'], batch_norm=batch_norm), 'VGG19', num_classes)