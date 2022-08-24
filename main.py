import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

import argparse
import numpy as np

from model.model import summary
import utils
import training
import perturb
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./', help='cifar10 data path')
    parser.add_argument('--save_path', type=str, default='./', help='records saving path')

    parser.add_argument('--pert', type=str, default=None, choices=['None', 'EM', 'OPS', 'CIFAR10-S'])
    parser.add_argument('--em_path', type=str, default=None, help='location of EM noise file (if used)')
    parser.add_argument('--model', type=str, default='RN18', choices=['LeNet', 'RN18', 'WRN-28-10', 'DN121', 'CVT', 'CCT'])

    parser.add_argument('--data_aug', type=str, default='Standard')
    parser.add_argument('--n_holes', type=int, default=1)
    parser.add_argument('--length', type=int, default=16)
    parser.add_argument('--reorg_n', type=int, default=4)
    parser.add_argument('--reorg_m', type=int, default=4)

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--num_classes', type=int, default=10)

    parser.add_argument('--sparsity', type=int, default=1)

    parser.add_argument('--at_norm', type=str, default='linf')
    parser.add_argument('--at_pgd_step', type=int, default=0, help='set to 0 if not using AT')
    parser.add_argument('--eps', type=float, default=8/255)
    parser.add_argument('--alpha', type=float, default=2/255)

    args = parser.parse_args()

    '''data augmentation'''
    trans = {
            'aug': transforms.Compose([
                                        transforms.ToTensor(), 
                                        ]),
            'clean': transforms.Compose([
                                        transforms.ToTensor(),
                                        ])
            }

    clean_train_data, clean_test_data = utils.load_cifar10_data(args.data_path,trans['aug'],trans['clean'])

    if args.data_aug == 'Standard':
        trans['aug'] = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, 4),
                                        transforms.ToTensor()])
    elif args.data_aug == 'Cutout':
        trans['aug'] = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, 4),
                                        transforms.ToTensor()])
        from augmentation.Cutout import Cutout
        trans['aug'].transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
    elif args.data_aug == 'RandAug':
        trans['aug'] = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, 4),
                                        transforms.ToTensor()])
        from augmentation.RandAugment import RandAugment
        trans['aug'].transforms.insert(2,RandAugment())

    print('data augment:\t{}\n'.format(args.data_aug))


    '''dataset preparation'''
    clean_train_data, clean_test_data = utils.load_cifar10_data(args.data_path,trans['aug'],trans['clean'])

    if args.pert == 'OPS':
        datapack = {'image': clean_train_data.data / 255,
                    'perturbation': perturb.pixel_search(clean_train_data, sparsity=args.sparsity),
                    'target': np.array(clean_train_data.targets)}

        pert_train_data = utils.Perturbed_Dataset(datapack['image'], datapack['perturbation'], datapack['target'], trans['aug'])

        train_loader = DataLoader(pert_train_data,batch_size=args.batch_size,shuffle=True,num_workers=8)
        print('\nusing OPS training set\n')
    elif args.pert == 'EM':
        datapack = {'image': clean_train_data.data / 255,
                    'perturbation': torch.load(args.em_path).permute(0,2,3,1).numpy(),
                    'target': np.array(clean_train_data.targets)}

        pert_train_data = utils.Perturbed_Dataset(datapack['image'], datapack['perturbation'], datapack['target'], trans['aug'])

        train_loader = DataLoader(pert_train_data,batch_size=args.batch_size,shuffle=True,num_workers=8)
        print('\nusing EM training set\n')
    elif args.pert == 'CIFAR10-S':
        datapack = {'image': clean_train_data.data / 255,
                    'perturbation': perturb.pixel_search(clean_train_data, 
                                                        pert_init=torch.load(args.em_path).permute(0,2,3,1).numpy(),
                                                        sparsity=args.sparsity),
                    'target': np.array(clean_train_data.targets)}

        pert_train_data = utils.Perturbed_Dataset(datapack['image'], datapack['perturbation'], datapack['target'], trans['aug'])

        train_loader = DataLoader(pert_train_data,batch_size=args.batch_size,shuffle=True,num_workers=8)
        print('\nusing CIFAR10-S training set\n')
    else:
        train_loader = DataLoader(clean_train_data,batch_size=args.batch_size,shuffle=True,num_workers=8)
        print('\nusing clean training set\n')

    clean_test_loader = DataLoader(clean_test_data,batch_size=args.batch_size,shuffle=False,num_workers=8)

    '''saving sample images'''
    num_per_class = 10
    idx_class = np.zeros((10,num_per_class), dtype=int)
    for i in range(10):
        idx_class[i] = np.where(np.array(datapack['target']) == i)[0][:num_per_class]

        imgs_class = datapack['image'][idx_class[i]]
        imgs_pert_class = datapack['image'][idx_class[i]] + datapack['perturbation'][idx_class[i]]
        pert_class = datapack['perturbation'][idx_class[i]]

        img_class_save = [torch.from_numpy(imgs_class).permute(0,3,1,2), 
                        torch.from_numpy(imgs_pert_class).permute(0,3,1,2),
                        torch.from_numpy(pert_class).permute(0,3,1,2)]

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        utils.save_img(img_class_save, os.path.join(args.save_path, 'sample_class_{}.png'.format(i)), nrow=num_per_class)

    pert_train_data = utils.Perturbed_Dataset(datapack['image'], datapack['perturbation'], datapack['target'], trans['aug'])


    '''model'''
    if args.model == 'LeNet':
        from model.LeNet import LeNet
        Net = LeNet(num_classes=args.num_classes)
    elif args.model == 'RN18':
        from model.ResNet import ResNet18
        Net = ResNet18(num_classes=args.num_classes)
    elif args.model == 'WRN-28-10':
        from model.WideResNet import Wide_ResNet
        Net = Wide_ResNet(depth=28,widen_factor=10,dropout_rate=0,num_classes=args.num_classes)
    elif args.model == 'DN121':
        from model.DenseNet import Densenet121
        Net = Densenet121(num_classes=args.num_classes)
    elif args.model == 'CVT':
        from model.Compact_Transformer.cvt import cvt_7_4_32
        Net = cvt_7_4_32(num_classes=args.num_classes)
    elif args.model == 'CCT':
        from model.Compact_Transformer.cct import cct_7_3x1_32
        Net = cct_7_3x1_32(num_classes=args.num_classes)

    Net = Net.cuda()
    summary(Net)

    '''use AdamW optimizer for CCT&CVT'''
    if args.model in ['CCT', 'CVT']:
        Opt = optim.AdamW(Net.parameters(), lr=args.lr, betas=[0.9, 0.999], weight_decay=args.weight_decay)
        LrSch = torch.optim.lr_scheduler.CosineAnnealingLR(Opt,T_max=args.epoch,eta_min=1e-5)
    else: 
        Opt = optim.SGD(Net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        LrSch = torch.optim.lr_scheduler.MultiStepLR(Opt,milestones=[args.epoch/2,args.epoch*3/4])

    from copy import deepcopy
    state_init = deepcopy(Net.state_dict())

    '''training'''
    for n in range(args.epoch):

        if args.at_pgd_step:   
            if args.data_aug == 'Mixup':
                from augmentation.Mixup import mixup_step
                train_loss, train_acc, param_grad_norm = mixup_step(Net,train_loader,Opt,perturb.pgd_adv,True,args.at_pgd_step,args.eps,args.alpha,args.at_norm)
            else:
                train_loss, train_acc, param_grad_norm = training.step(Net,train_loader,Opt,perturb.pgd_adv,True,args.at_pgd_step,args.eps,args.alpha,args.at_norm)  #single step AT
            val_loss, val_acc, _ = training.step(Net,clean_test_loader)
            LrSch.step()

            param_diff_norm = utils.net_param_diff_norm(Net, state_init)

            print(
            'epoch {}\n'.format(n+1), 
            'param_diff_norm:{:.6f}\t'.format(param_diff_norm),
            'param_grad_norm:{:.6f}\t'.format(param_grad_norm),
            'adv_train_loss:{:.6f}\t'.format(train_loss), 
            'adv_train_acc:{:.4f}%\t'.format(train_acc), 
            'val_loss:{:.6f}\t'.format(val_loss), 
            'val_acc:{:.4f}%\t'.format(val_acc),
            'lr:{:.6f}\t'.format(Opt.param_groups[0]['lr'])
            )   


        else:
            if args.data_aug == 'Mixup':
                from augmentation.Mixup import mixup_step
                train_loss, train_acc, param_grad_norm = mixup_step(Net,train_loader,Opt)
            else:
                train_loss, train_acc, param_grad_norm = training.step(Net,train_loader,Opt)
            val_loss, val_acc, _ = training.step(Net,clean_test_loader)
            LrSch.step()

            param_diff_norm = utils.net_param_diff_norm(Net, state_init)

            print(
            'epoch {}\n'.format(n+1), 
            'param_diff_norm:{:.6f}\t'.format(param_diff_norm),
            'param_grad_norm:{:.6f}\t'.format(param_grad_norm),
            'normal_train_loss:{:.6f}\t'.format(train_loss), 
            'normal_train_acc:{:.4f}%\t'.format(train_acc), 
            'val_loss:{:.6f}\t'.format(val_loss), 
            'val_acc:{:.4f}%\t'.format(val_acc),
            'lr:{:.6f}\t'.format(Opt.param_groups[0]['lr'])
            )   

    '''saving the final checkpoint'''
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    torch.save(Net.state_dict(),os.path.join(args.save_path, '{}-{}epochs.pkl'.format(Net.model_name, args.epoch)))
    print('model saved')

