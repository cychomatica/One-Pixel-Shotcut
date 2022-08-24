import torch
import torchvision

import numpy as np

from PIL import Image


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, data, target, transform):

        '''data format: np.ndarray, float32 range from 0 to 1, H x W x C'''
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index: int):

        img, target = self.data[index], self.target[index]
        img = np.uint8(img * 255)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
        
class Perturbed_Dataset(torch.utils.data.Dataset):

    def __init__(self, data, perturbation, target, transform, pert=1) -> None:
        super().__init__()

        '''Clean Examples if 'pert' is False'''
    
        '''data format: np.ndarray, float32 range from 0 to 1, H x W x C'''

        self.data = data
        self.perturbation = perturbation
        self.target = target
        self.transform = transform
        self.pert = pert

        '''Perturbation mode: S for sample-wise, C for class-wise, U for universal'''

        if len(self.perturbation.shape) == 4:
            if self.perturbation.shape[0] == len(self.target):
                self.mode = 'S'
            else:
                self.mode = 'C'
        else:
            self.mode = 'U'

    def __len__(self):

        return len(self.target)

    def __getitem__(self, index:int):
        
        if self.pert == 1:
            if self.mode == 'S':
                img_p, target = self.data[index] + self.perturbation[index], self.target[index]
            elif self.mode == 'C':
                img_p, target = self.data[index] + self.perturbation[self.target[index]], self.target[index]
            else:
                img_p, target = self.data[index] + self.perturbation, self.target[index]

        elif self.pert == 2:
            img_p, target = self.perturbation[index], self.target[index]
            
        else:
            img_p, target = self.data[index], self.target[index]

        img_p = np.clip(img_p, 0, 1)
        img_p = np.uint8(img_p * 255)
        img_p = Image.fromarray(img_p)
        
        if self.transform is not None:
            img_p = self.transform(img_p)
            
        return img_p, target
        

def net_param_diff_norm(model:torch.nn.Module, state_dict_init, p='fro'):

    diff_norm_list = []
    for name, parameter in model.named_parameters():
        diff_norm_list.append(torch.norm(parameter.data - state_dict_init[name], p=p).cpu().numpy())

    diff_norm = np.linalg.norm(np.array(diff_norm_list))
    return diff_norm

def load_cifar10_data(path,download=True,transform_train=None,transform_test=None):

    '''return torchvision.datasets.CIFAR10'''

    traindata = torchvision.datasets.CIFAR10(root=path,train=True,download=download,transform=transform_train)
    testdata = torchvision.datasets.CIFAR10(root=path,train=False,download=download,transform=transform_test)
    traindata.targets = np.array(traindata.targets)
    testdata.targets = np.array(testdata.targets)
    return traindata, testdata

def load_cifar100_data(path,download=True,transform_train=None,transform_test=None):

    '''return torchvision.datasets.CIFAR10'''

    traindata = torchvision.datasets.CIFAR100(root=path,train=True,download=download,transform=transform_train)
    testdata = torchvision.datasets.CIFAR100(root=path,train=False,download=download,transform=transform_test)
    traindata.targets = np.array(traindata.targets)
    testdata.targets = np.array(testdata.targets)
    return traindata, testdata

def save_img(imgs:list,save_path:str,nrow=8):

    img_save = torchvision.utils.make_grid(torch.cat(imgs,dim=0),nrow,pad_value=1)
    img_save = img_save.permute(1,2,0) * 255
    img_save = img_save.cpu().numpy()
    img_save = Image.fromarray(img_save.astype('uint8'))
    img_save.save(save_path)

