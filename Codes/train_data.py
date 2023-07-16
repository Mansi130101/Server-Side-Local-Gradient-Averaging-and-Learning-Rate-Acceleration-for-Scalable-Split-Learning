import argparse
import os
import shutil
import time
import errno
import random
import math
import kornia
from kornia import augmentation as augs
from kornia import filters
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
#from torchlars import LARS
import networks.resnet
from networks.losses import M_CosineLoss, E_CosineLoss, DistanceLoss, AMCLoss, LE_CosineLoss
from data_utils import sort_indices, create_dataloader, assert_distribution, sort_multi_indices, divide_indices

parser = argparse.ArgumentParser(description='InfoPro-PyTorch')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset: [cifar10|stl10|svhn]')

parser.add_argument('--model', default='resnet', type=str,
                    help='resnet is supported currently')

parser.add_argument('--layers', default=0, type=int,
                    help='total number of layers (have to be explicitly given!)')

parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')

parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.set_defaults(augment=True)

parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--name', default='', type=str,
                    help='name of experiment')
parser.add_argument('--no', default='1', type=str,
                    help='index of the experiment (for recording convenience)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    help='print frequency (default: 10)')


# Cosine learning rate
parser.add_argument('--cos_lr', dest='cos_lr', action='store_true',
                    help='whether to use cosine learning rate')
parser.set_defaults(cos_lr=False)


# InfoPro
parser.add_argument('--local_module_num', default=1, type=int,
                    help='number of local modules (1 refers to end-to-end training)')

parser.add_argument('--balanced_memory', dest='balanced_memory', action='store_true',
                    help='whether to split local modules with balanced GPU memory (InfoPro* in the paper)')
parser.set_defaults(balanced_memory=False)

parser.add_argument('--aux_net_config', default='1c2f', type=str,
                    help='architecture of auxiliary classifier / contrastive head '
                         '(default: 1c2f; 0c1f refers to greedy SL)'
                         '[0c1f|0c2f|1c1f|1c2f|1c3f|2c2f]')

parser.add_argument('--local_loss_mode', default='contrast', type=str,
                    help='ways to estimate the task-relevant info I(x, y)'
                         '[contrast|cross_entropy]')

parser.add_argument('--aux_net_widen', default=1.0, type=float,
                    help='widen factor of the two auxiliary nets (default: 1.0)')

parser.add_argument('--aux_net_feature_dim', default=0, type=int,
                    help='number of hidden features in auxiliary classifier / contrastive head '
                         '(default: 128)')

# The hyper-parameters \lambda_1 and \lambda_2 for 1st and (K-1)th local modules.
# Note that we assume they change linearly between these two modules.
# (The last module always uses standard end-to-end loss)
# See our paper for more details.

parser.add_argument('--alpha_1', default=0.0, type=float,)   # \alpha_1 for 1st local module
parser.add_argument('--beta_1', default=0.0, type=float,)   # \beta_1 for 1st local module
parser.add_argument('--beta_2', default=0.0, type=float,)   # \beta_2 for 1st local module
parser.add_argument("--gamma_1", default=0.0, type=float,)   # gamma_1 of contrastive loss
parser.add_argument("--gamma_2", default=0.0, type=float,)   # gamma_2 of contrastive loss

parser.add_argument('--ixx_2', default=0.0, type=float,)   # \lambda_1 for (K-1)th local module
parser.add_argument('--ixy_2', default=0.0, type=float,)   # \lambda_2 for (K-1)th local module

parser.add_argument('--skewness_ratio', default=0.5, type=float,)  # Skewness between labels

parser.add_argument("--branch", dest="branch", action = "store_true", help="Whether to branch Chunk 2.")  # Whether to branch Chunk 2
parser.set_defaults(branch=False)

parser.add_argument("--contrastive_loss_function", type=str, required=True, help="Which type of contrastive loss to use. opetions available: ecosine, mcosine, distance, amc")
parser.add_argument("--positive_weight", type=float, default=1.0, help="weight of positive loss")
parser.add_argument("--negative_weight", type=float, default=1.0, help="weight of negative loss")
parser.add_argument("--contrastive_loss_margin", type=float, default = 0.5, help="margin used in contrastive loss function")



args = parser.parse_args()

# augmentation utils
class RandomApply(nn.Module):
  def __init__(self, fn, p):
      super().__init__()
      self.fn = fn                 # Transform function to apply
      self.p = p                   # Probability of application
  def forward(self, x):
      if random.random() > self.p:
          return x
      return self.fn(x)

def augmented(img):
  transform_fn = nn.Sequential(RandomApply(augs.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8), 
                             augs.RandomGrayscale(p=0.3), 
                             augs.RandomHorizontalFlip(), 
                             RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.2), 
                             augs.RandomResizedCrop((32, 32)))  #RandomGaussianNoise, apply_adjust_brightness, apply_adjust_contrast
  
  aug_img = transform_fn(img)
  return aug_img

if args.contrastive_loss_function == "distance":
  loss_fn = DistanceLoss(args.contrastive_loss_margin, a=args.negative_weight, b=args.positive_weight)
elif args.contrastive_loss_function == "amc":
  loss_fn = AMCLoss(args.contrastive_loss_margin, a=args.negative_weight, b=args.positive_weight)
elif args.contrastive_loss_function == "ecosine":
  loss_fn = E_CosineLoss(args.contrastive_loss_margin, a=args.negative_weight, b=args.positive_weight)
elif args.contrastive_loss_function == "mcosine":
  loss_fn = M_CosineLoss(args.contrastive_loss_margin, a=args.negative_weight, b=args.positive_weight)
elif args.contrastive_loss_function == "lecosine":
  loss_fn = LE_CosineLoss(args.contrastive_loss_margin, a=args.negative_weight, b=args.positive_weight)
else:
  raise NotImplementedError("Loss function not Implemented")

# Configurations adopted for training deep networks.
training_configurations = {
    'resnet': {
        'epochs': 150,
        "skewness_ratio" : args.skewness_ratio, 
        'batch_size': 128 if args.dataset in ['cifar10', 'svhn'] else 128,
        'initial_learning_rate': 2e-3 if args.dataset in ['cifar10', 'svhn',"fashion_mnist"] else 0.1,
        'changing_lr': [80, 120],
        'lr_decay_rate': 1e-3,
        'momentum': 0.9,         
        'nesterov': False,
        'weight_decay': 1e-6,
    }
}

record_path = './' \
              + ('InfoPro*_' if args.balanced_memory else 'InfoPro_') \
              + str(args.dataset) \
              + '_' + str(args.model) + str(args.layers) \
              + '_K_' + str(args.local_module_num) \
              + '_' + str(args.name) \
              + '/baseline/cifar10/' \
              + args.contrastive_loss_function\
              + "/"\
              + 'no_' + str(args.no) \
              + '_aux_net_config_' + str(args.aux_net_config) \
              + '_local_loss_mode_' + str(args.local_loss_mode) \
              + '_aux_net_widen_' + str(args.aux_net_widen) \
              + '_aux_net_feature_dim_' + str(args.aux_net_feature_dim) \
              + '_alpha_1_' + str(args.alpha_1) \
              + '_beta_1_' + str(args.beta_1) \
              + '_beta_2_' + str(args.beta_2) \
              + '_ixx_2_' + str(args.ixx_2) \
              + '_ixy_2_' + str(args.ixy_2) \
              + ('_cos_lr_' if args.cos_lr else '')\
              + '_gamma_1_' + str(args.gamma_1)\
              + '_gamma_2_' + str(args.gamma_2)\
              + '_skewness_rate_' + str(args.skewness_ratio)\
              + '_m_'+ str(args.contrastive_loss_margin)\
              + '_a_' + str(args.negative_weight)\
              + '_b_' + str(args.positive_weight)\
              + "_epochs_" + str(training_configurations['resnet']["epochs"])\
              + "_lr_" + str(training_configurations['resnet']["initial_learning_rate"])

print("Storing in ...", record_path)

record_file1 = record_path + '/training_process1.txt'
accuracy_file1 = record_path + '/accuracy_epoch1.txt'
loss_file1 = record_path + '/loss_epoch1.txt'
check_point1 = os.path.join(record_path, args.checkpoint)
test_file1 = record_path + "/test1.txt"
record_file2 = record_path + '/training_process2.txt'
accuracy_file2 = record_path + '/accuracy_epoch2.txt'
loss_file2 = record_path + '/loss_epoch2.txt'
check_point2 = os.path.join(record_path, args.checkpoint)
test_file2 = record_path + "/test2.txt"

pretrain = True  # While using this script/folder we keep pretrain True which can be made false while fine tuning on colab


def main():
    global best_prec1
    best_prec1 = 0
    global val_acc1
    val_acc1 = []
    global best_prec2
    best_prec2 = 0
    global val_acc2
    val_acc2 = []
    global best_avg_prec
    best_avg_prec = 0
    start_time = time.time()

    class_num = args.dataset in ['cifar10', 'sl10', 'svhn'] and 10 or 100

    if 'cifar' in args.dataset:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        kwargs_dataset_train = {'train': True}
        kwargs_dataset_test = {'train': False}
    else:
        normalize = transforms.Normalize(mean=[x / 255 for x in [127.5, 127.5, 127.5]],
                                         std=[x / 255 for x in [127.5, 127.5, 127.5]])
        kwargs_dataset_train = {'split': 'train'}
        kwargs_dataset_test = {'split': 'test'}

    if args.augment:
        if 'cifar' in args.dataset:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4, 4, 4, 4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            image_size = 32
        else:
            raise NotImplementedError

    else:
        transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                    ])
        image_size = 32

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': False}

    #**************************************added***********
    train_data=datasets.CIFAR10(root="/content/images", train=True, download=True, transform=transform_train)
    test_data=datasets.CIFAR10(root="/content/images", train=False, download=True, transform=transform_test)
    
    '''
    dataset_size = "xxs"
    train_indices, val_indices = sort_indices(train_data, 2, dataset_size)   # Sort indices for 2 labels
    test_indices, _ = sort_indices(test_data, 2, "xl")     # Sort indices for 2 labels   
    '''

    train_indices, val_indices = sort_multi_indices(train_data, 200, 10, 'train')
    test_indices, _ = sort_multi_indices(test_data, 1000, 10, "test")
    
    # BASELINE
    #np.random.seed(10)
    #train1_indices = train_indices.copy()
    #train2_indices = train_indices.copy()
    #np.random.shuffle(train1_indices)
    #np.random.shuffle(train2_indices)
    
    # PROPOSED
    train_worker_indices = divide_indices(train_indices, 200, 10, workers=2)
    assert_distribution(train_data, train_worker_indices, None)
    assert_distribution(train_data, None, val_indices)
    train_loaders, val_loader, test_loader = create_dataloader(train_data, test_data, 
    train_worker_indices, val_indices, test_indices,
    training_configurations[args.model]["batch_size"])
    
    train_loader1 = train_loaders[0]
    train_loader2 = train_loaders[1] 
    print("DataLoader 1 samples = ", len(train_loader1)*training_configurations[args.model]["batch_size"])
    print("DataLoader 2 samples = ",len(train_loader2)*training_configurations[args.model]["batch_size"])
    print("Sample Batch...\n")
    tmp = next(iter(train_loader1))
    print(tmp[1][:20])
    tmp = next(iter(train_loader2))
    print(tmp[1][:20])
    
    #train_loader = torch.utils.data.DataLoader(
    #    datasets.__dict__[args.dataset.upper()]('./data', download=True, transform=transform_train,
    #                                            **kwargs_dataset_train),
    #    batch_size=training_configurations[args.model]['batch_size'], shuffle=True, **kwargs)
    #val_loader = torch.utils.data.DataLoader(
    #    datasets.__dict__[args.dataset.upper()]('./data', transform=transform_test,
    #                                            **kwargs_dataset_test),
    #    batch_size=training_configurations[args.model]['batch_size'], shuffle=False, **kwargs)
    #**************************************added***********
    
    # create model
    if args.model == 'resnet':
        model1 = eval('networks.resnet.resnet' + str(args.layers))\
            (local_module_num=args.local_module_num,
             batch_size=training_configurations[args.model]['batch_size'],
             image_size=image_size,
             balanced_memory=args.balanced_memory,
             dataset=args.dataset,
             class_num=class_num,
             wide_list=(16, 16, 32, 64),
             dropout_rate=args.droprate,
             aux_net_config=args.aux_net_config,
             local_loss_mode=args.local_loss_mode,
             aux_net_widen=args.aux_net_widen,
             aux_net_feature_dim=args.aux_net_feature_dim)
        model2 = eval('networks.resnet.resnet' + str(args.layers))\
            (local_module_num=args.local_module_num,
             batch_size=training_configurations[args.model]['batch_size'],
             image_size=image_size,
             balanced_memory=args.balanced_memory,
             dataset=args.dataset,
             class_num=class_num,
             wide_list=(16, 16, 32, 64),
             dropout_rate=args.droprate,
             aux_net_config=args.aux_net_config,
             local_loss_mode=args.local_loss_mode,
             aux_net_widen=args.aux_net_widen,
             aux_net_feature_dim=args.aux_net_feature_dim)
    else:
        raise NotImplementedError

    if not os.path.isdir(check_point1):
        mkdir_p(check_point1)
    if not os.path.isdir(check_point2):
        mkdir_p(check_point2)

    cudnn.benchmark = True

    optimizer1 = torch.optim.SGD(model1.parameters(),
                                lr=training_configurations[args.model]['initial_learning_rate'],
                                momentum=training_configurations[args.model]['momentum'],
                                nesterov=training_configurations[args.model]['nesterov'],
                                weight_decay=training_configurations[args.model]['weight_decay'])#,
                                #eps = 1e-8, trust_coef = 0.001)
    if torch.cuda.is_available():
      model1 = torch.nn.DataParallel(model1).cuda()

    optimizer2 = torch.optim.SGD(model2.parameters(),
                                lr=training_configurations[args.model]['initial_learning_rate'],
                                momentum=training_configurations[args.model]['momentum'],
                                nesterov=training_configurations[args.model]['nesterov'],
                                weight_decay=training_configurations[args.model]['weight_decay'])#, 
                                #eps = 1e-8, trust_coef = 0.001)
    if torch.cuda.is_available():
      model2 = torch.nn.DataParallel(model2).cuda()

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        val_acc = checkpoint['val_acc']
        best_prec1 = checkpoint['best_acc']
        np.savetxt(accuracy_file, np.array(val_acc))
    else:
        start_epoch = 0

    for epoch in range(start_epoch, training_configurations[args.model]['epochs']):

        #adjust_learning_rate(optimizer1, epoch + 1)
        #adjust_learning_rate(optimizer2, epoch + 1)
        
        # train for one epoch
        chunk2_loss = train(train_loader1, train_loader2, model1, model2, optimizer1, optimizer2, epoch)
        #print("Final Chunk 2 Loss = ", chunk2_loss)
        
        # evaluate on validation set
        prec1, prec2 = validate(val_loader, model1, model2, epoch)
        
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model1.state_dict(),
            'best_acc': best_prec1,
            'optimizer': optimizer1.state_dict(),
            'val_acc': val_acc1,
        }, is_best, checkpoint=check_point1)
        np.savetxt(accuracy_file1, np.array(val_acc1))

        # remember best prec@1 and save checkpoint
        is_best = prec2 > best_prec2
        best_prec2 = max(prec2, best_prec2)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model2.state_dict(),
            'best_acc': best_prec2,
            'optimizer': optimizer2.state_dict(),
            'val_acc': val_acc2,
        }, is_best, checkpoint=check_point2)

        avg_prec = (prec1+prec2)/2
        best_avg_prec = max(best_avg_prec, avg_prec)
        print('Best Prec 1: ', best_prec1)
        print('Best Prec 2: ', best_prec2)
        print('Best avg accuracy: ', best_avg_prec)
        np.savetxt(accuracy_file2, np.array(val_acc2))
        #if epoch % 99 == 0:
        #  test_prec1, test_prec2 = validate(test_loader, model1, model2, training_configurations[args.model]["epochs"], "test",start_time)
        #  print("After 100 epochs of Training...\n Test prec 1 = ", test_prec1, " Test Prec 2 = ", test_prec2, " Avg Test Prec = ", (test_prec1+test_prec2)/2)

    test_prec1, test_prec2 = validate(test_loader, model1, model2, training_configurations[args.model]["epochs"], "test", start_time)
    print("After end of Training...\n Test prec 1 = ", test_prec1, " Test Prec 2 = ", test_prec2, " Avg Test Prec = ", (test_prec1+test_prec2)/2)
    print("*****************************************************************************************")




def train(train_loader1, train_loader2, model1, model2, optimizer1, optimizer2, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()

    losses1 = AverageMeter()
    losses2 = AverageMeter()

    top1 = AverageMeter()
    top2 = AverageMeter()

    train_batches_num = len(train_loader1)

    # switch to train mode
    model1.train()
    model2.train()

    end = time.time()
    for i, ((x1, target1), (x2, target2)) in enumerate(zip(train_loader1, train_loader2)):
        if torch.cuda.is_available():
          target1 = target1.cuda()
          target2 = target2.cuda()
          x1 = x1.cuda()
          x2 = x2.cuda()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        output1, chunk1_loss1, chunk2_loss1, hidden1, hidden12 = model1(img=x1,
                            target=target1,
                            alpha_1=args.alpha_1,
                            beta_1=args.beta_1,
                            beta_2=args.beta_2,
                            ixx_2=args.ixx_2,
                            ixy_2=args.ixy_2,
                            gamma_1 = args.gamma_1,
                            gamma_2 = args.gamma_2)

        # measure accuracy and record loss
        prec1 = accuracy(output1.data, target1, topk=(1,))[0]
        losses1.update(chunk2_loss1.data.item(), x1.size(0))
        top1.update(prec1.item(), x1.size(0))

        output2, chunk1_loss2, chunk2_loss2, hidden2, hidden22 = model2(img=x2,
                            target=target2,
                            alpha_1=args.alpha_1,
                            beta_1=args.beta_1,
                            beta_2=args.beta_2,
                            ixx_2=args.ixx_2,
                            ixy_2=args.ixy_2,
                            gamma_1 = args.gamma_1,
                            gamma_2 = args.gamma_2)
        
        # measure accuracy and record loss
        prec2 = accuracy(output2.data, target2, topk=(1,))[0]
        losses2.update(chunk2_loss2.data.item(), x2.size(0))
        top2.update(prec2.item(), x2.size(0))
          
        loss1 = chunk1_loss1 + chunk2_loss1
        loss2 = chunk1_loss2 + chunk2_loss2 
        if args.gamma_1 != 0:
          contra_loss_chunk1 = args.gamma_1*loss_fn(hidden1, hidden2, target1, target2)
          #print("Chunk 1 contrastive :", contra_loss_chunk1.item())
          #print("Total Loss = ", loss1)
          loss1 += contra_loss_chunk1
          
        if args.gamma_2 != 0:
          #contra_loss_chunk2 = args.gamma_2*Bicontrastive(hidden12, hidden22)
          contra_loss_chunk2 = args.gamma_2*loss_fn(hidden12, hidden22, target1, target2)
          loss2 += contra_loss_chunk2
          #print("Chunk 2 Contrastive Loss = ", contra_loss_chunk2.item())
          
        loss1.backward(retain_graph = True) 
        loss2.backward()
        optimizer1.step()
        optimizer2.step()    

        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            # print(discriminate_weights)
            fd = open(record_file1, 'a+')
            string = ('W1 Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                       epoch, i+1, train_batches_num, batch_time=batch_time,
                       loss=losses1, top1=top1))
            print(string)
            fd.write(string + '\n')
            fd.close()

            fd = open(record_file2, "a+")
            string = ('W2 Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                       epoch, i+1, train_batches_num, batch_time=batch_time,
                       loss=losses2, top1=top2))
            print(string)
            fd.write(string + '\n')
            fd.close()
    return 


def validate(val_loader, model1, model2, epoch, mode="val",start_time=0):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    
    losses1 = AverageMeter()
    losses2 = AverageMeter()

    top1 = AverageMeter()
    top2 = AverageMeter()

    train_batches_num = len(val_loader)

    # switch to evaluate mode
    model1.eval()
    model2.eval()

    end = time.time()
    for i, (x, target) in enumerate(val_loader):
        if torch.cuda.is_available():
          target = target.cuda()
          x = x.cuda()
        input_x = torch.autograd.Variable(x)
        target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            output1, loss1 = model1(img=input_x,
                                 target=target_var,
                                 beta_2 = args.beta_2)
            output2, loss2 = model2(img=input_x,
                                 target=target_var,
                                 beta_2 = args.beta_2)
        
        # measure accuracy and record loss
        prec1 = accuracy(output1.data, target, topk=(1,))[0]
        losses1.update(loss1.data.item(), input_x.size(0))
        top1.update(prec1.item(), input_x.size(0))

        prec2 = accuracy(output2.data, target, topk=(1,))[0]
        losses2.update(loss2.data.item(), input_x.size(0))
        top2.update(prec2.item(), input_x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    
    if mode=="val":
      fd = open(record_file1, 'a+')
      string = ('W1 Val: [{0}][{1}/{2}]\t'
                'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
          epoch, (i + 1), train_batches_num, batch_time=batch_time,
          loss=losses1, top1=top1))
      print(string)
      fd.write(string + '\n')
      fd.close()

      fd = open(record_file2, 'a+')
      string = ('W2 Val: [{0}][{1}/{2}]\t'
                'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
          epoch, (i + 1), train_batches_num, batch_time=batch_time,
          loss=losses2, top1=top2))
      print(string)
      fd.write(string + '\n')
      fd.close()
      val_acc1.append(top1.ave)
      val_acc2.append(top2.ave)
    else:
      print("While Testing, Loading best model state...")
      c1 = torch.load(check_point1+"/checkpoint.pth.tar")
      c2 = torch.load(check_point2+"/checkpoint.pth.tar")
      model1.load_state_dict(c1["state_dict"])
      model2.load_state_dict(c2["state_dict"])
      print(f'Loaded best checkpoint from epoch {c1["epoch"]} !!')
      fd = open(test_file1, 'a+')
      string = ('W1 Test: [{0}][{1}/{2}]\t'
                'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
          epoch, (i + 1), train_batches_num, batch_time=batch_time,
          loss=losses1, top1=top1))
      print(string)
      fd.write(string + '\n ' + str(time.time()-start_time))
      fd.close()
      
      fd = open(test_file2, 'a+')
      string = ('W2 Test: [{0}][{1}/{2}]\t'
                'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
          epoch, (i + 1), train_batches_num, batch_time=batch_time,
          loss=losses2, top1=top2))
      print(string)
      fd.write(string + '\n' + str(time.time()-start_time))
      fd.close()
      
    return top1.ave, top2.ave


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    if not args.cos_lr:
        if epoch in training_configurations[args.model]['changing_lr']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= training_configurations[args.model]['lr_decay_rate']
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

    else:
        for param_group in optimizer.param_groups:
            if epoch <= 10:
                param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate']\
                                * (1 + math.cos(math.pi * epoch / training_configurations[args.model]['epochs'])) * (epoch - 1) / 10 + 0.01 * (11 - epoch) / 10
            else:
                param_group['lr'] = 0.5 * training_configurations[args.model]['initial_learning_rate']\
                                    * (1 + math.cos(math.pi * epoch / training_configurations[args.model]['epochs']))
        print('lr:')
        for param_group in optimizer.param_groups:
            print(param_group['lr'])


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()