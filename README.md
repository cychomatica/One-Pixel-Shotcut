# One-Pixel Shortcut
Official Pytorch implementation of paper **One-Pixel Shortcut: on the Learning Preference of Deep Neural Networks**

Accepted to **ICLR 2023** as a **spotlight** paper.

By Shutong Wu, Sizhe Chen, Cihang Xie and Xiaolin Huang

Paper Link: https://arxiv.org/abs/2205.12141

## Requirements

Here are the versions of packages we use for the implementation of experiments.


|Library         | Version |
|----------------------|----|
|`Python`|  `3.7.7`|
|`pytorch`|  `1.7.1`|
|`torchvision`|  `0.8.2`|
|`numpy`|  `1.20.3`|
|`tqdm`| `4.62.2`|

## Run the main One-Pixel Shortcut training and evaluation script
For example, here is the command to train a ResNet-18 on OPS data: 
```console
python main.py \
--data_path=the location of your dataset \
--save_path=the saving location and name of this experiment \
--pert=OPS \
--model=RN18 \
--data_aug=Standard \
--sparsity=1 \
--at_pgd_step=0 \
```

if you want to train a ResNet-18 on CIFAR-10-S data, run: 
```console
python main.py \
--data_path=the location of your dataset \
--save_path=the saving location and name of this experiment \
--pert=CIFAR10-S \
--em_path=the location of EM noise file \
--model=RN18 \
--data_aug=Standard \
--sparsity=1 \
--at_pgd_step=0 \
```
Due to the limitation of file size, we do not include our pre-generated EM noise here. Please see https://github.com/HanxunH/Unlearnable-Examples for details of EM noise generation. Then set _**em_path**_ to the location where you save the EM noise file.
