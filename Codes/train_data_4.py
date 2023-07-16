import argparse
import os
import shutil
import time
import errno
import random
import math
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
import networks.resnet_2
from networks.losses import *
from data_utils import *

parser = argparse.ArgumentParser(description='InfoPro-PyTorch')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset: [cifar10|stl10|svhn|fmnist]')

parser.add_argument('--model', default='resnet', type=str,
                    help='resnet is supported currently')

parser.add_argument('--layers', default=0, type=int,
                    help='total number of layers (have to be explicitly given!)')

parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')

parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.set_defaults(augment=False)

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

parser.add_argument('--skewness_ratio', default=1/3, type=float,)  # Skewness between labels

parser.add_argument("--contrastive_loss_function", type=str, default='distance', help="Which type of contrastive loss to use. opetions available: ecosine, mcosine, distance, amc, lecosine") #required=True
parser.add_argument("--positive_weight", type=float, default=1.0, help="weight of positive loss")
parser.add_argument("--negative_weight", type=float, default=500.0, help="weight of negative loss")
parser.add_argument("--contrastive_loss_margin", type=float, default = 1.5, help="margin used in contrastive loss function")
parser.add_argument("--workers", type=int, default = 2, help="number of workers")
parser.add_argument("--epoch", type=int, default = 60, help="number of epochs")

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
        'epochs': args.epoch,
        "skewness_ratio" : args.skewness_ratio, 
        'batch_size': 128 if args.dataset in ['cifar10', 'svhn'] else 128,
        'initial_learning_rate': 2e-4 if args.dataset in ['cifar10', 'svhn'] else 0.1,
        'changing_lr': [80, 120],
        'lr_decay_rate': 1e-6,
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
              + '/single_proj_head/multiworker/dynamicrampupdown/' + str(args.contrastive_loss_function)\
              + '/worker_' + str(args.workers) + '/' \
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
              + ('_cos_lr_' if args.cos_lr else '') \
              + '_gamma_1_' + str(args.gamma_1) \
              + '_gamma_2_' + str(args.gamma_2) \
              + '_margin_' + str(args.contrastive_loss_margin) \
              + '_a_' + str(args.negative_weight) \
              + '_b_' + str(args.positive_weight) \
              + '_skewness_rate_' + str(args.skewness_ratio) \
              + '_epochs_' + str(args.epoch) \
              + '_run2'
print("Storing in ...", record_path)

record_file=[]
accuracy_file_=[]
loss_file=[]
check_point=[]
ph_check_point = []
test_file=[]

for num in range(args.workers):
    record_file.append(record_path + '/training_process{}.txt'.format(num+1))
    accuracy_file_.append(record_path + '/accuracy_epoch{}.txt'.format(num+1))
    loss_file.append(record_path + '/loss_epoch{}.txt'.format(num+1))
    cp = args.checkpoint + '_' + str(num)
    check_point.append(os.path.join(record_path, cp))
    test_file.append(record_path + "/test{}.txt".format(num+1))

for num in range(args.local_module_num):
    cp = args.checkpoint + '_ph_' + str(num)
    ph_check_point.append(os.path.join(record_path, cp))



pretrain = True  # While using this script/folder we keep pretrain True which can be made false while fine tuning on colab


def main():
    global best_avg_prec
    best_avg_prec = 0
    global best_prec_
    best_prec_=[]
    global val_acc_
    val_acc_=[]

    for num in range(args.workers):
      best_prec_.append(0)
      val_acc_.append([])

    start_time = time.time()

    class_num = args.dataset in ['cifar10', 'sl10', 'svhn','fmnist'] and 10 or 100

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
                    transforms.Resize((32,32)),
                      
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    normalize,
                    ])
        image_size=32

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32,32)),

        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': False}

    if args.dataset=='cifar':
      train_data=datasets.CIFAR10(root="/content/images", train=True, download=True, transform=transform_train)
      test_data=datasets.CIFAR10(root="/content/images", train=False, download=True, transform=transform_test)
    elif args.dataset=='fmnist':
      train_data=datasets.FashionMNIST(root="/content/images", train=True, download=True, transform=transform_train)
      test_data=datasets.FashionMNIST(root="/content/images", train=False, download=True, transform=transform_test)
    
    train_indices, val_indices = sort_multi_indices(train_data, args.workers*25, 10, 'train',args.dataset)
    test_indices, _ = sort_multi_indices(test_data, 1000, 10, "test",args.dataset)
  
    
    # PROPOSED
    train_worker_indices = divide_indices(train_indices, args.workers*25, 10, workers=args.workers)
    assert_distribution(train_data, train_worker_indices, None)
    assert_distribution(train_data, None, val_indices)
    train_loaders, val_loader, test_loader = create_dataloader(train_data, test_data, 
    train_worker_indices, val_indices, test_indices,
    training_configurations[args.model]["batch_size"])
    
    #train_loader1 = train_loaders[0]
    #train_loader2 = train_loaders[1] 
    #print("DataLoader 1 samples = ", len(train_loader1)*training_configurations[args.model]["batch_size"])
    #print("DataLoader 2 samples = ",len(train_loader2)*training_configurations[args.model]["batch_size"])
    #print("Sample Batch...\n")
    #tmp = next(iter(train_loader1))
    #print(tmp[1][:20])
    #tmp = next(iter(train_loader2))
    #print(tmp[1][:20])
    
    workers=[]
    
    if args.model == 'resnet':
        for num in range(args.workers):
            
            model = eval('networks.resnet_2.no_proj_resnet' + str(args.layers))\
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
            workers.append(model)
        assert(len(workers)==args.workers)
    else:
        raise NotImplementedError

    projection_heads = []
    
    projection_head_1 = nn.Sequential(
      nn.Linear(in_features=8192, out_features=1024),
      nn.ReLU(),
      nn.Linear(in_features=1024, out_features=128)
    )
    projection_head_2 = nn.Sequential(
      nn.Linear(in_features=64, out_features=1024),
      nn.ReLU(),
      nn.Linear(in_features=1024, out_features=128)
    )

    if torch.cuda.is_available():
      projection_head_1 = projection_head_1.cuda()
      projection_head_2 = projection_head_2.cuda()

    projection_heads.append(projection_head_1)
    projection_heads.append(projection_head_2)


    for num in range(args.workers):
      if not os.path.isdir(check_point[num]):
        mkdir_p(check_point[num])
    for num in range(args.local_module_num):
      if not os.path.isdir(ph_check_point[num]):
        mkdir_p(ph_check_point[num])

    cudnn.benchmark = True

    optimizers=[]
     
    for num in range(args.workers):
        optimizer = torch.optim.SGD(workers[num].parameters(),
                                    lr=training_configurations[args.model]['initial_learning_rate'],
                                    momentum=training_configurations[args.model]['momentum'],
                                    nesterov=training_configurations[args.model]['nesterov'],
                                    weight_decay=training_configurations[args.model]['weight_decay'])#,
                                    #eps = 1e-8, trust_coef = 0.001)
        optimizers.append(optimizer)
        if torch.cuda.is_available():
          workers[num] = workers[num].cuda()
    



    proj_head_optimizers = []
    proj_head_optimizers.append(torch.optim.SGD(projection_head_1.parameters(), lr=5e-5, weight_decay=1e-6))
    proj_head_optimizers.append(torch.optim.SGD(projection_head_2.parameters(), lr=5e-5, weight_decay=1e-6))

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
        best_prec_[0] = checkpoint['best_acc']
        np.savetxt(accuracy_file, np.array(val_acc))
    else:
        start_epoch = 0

    for epoch in range(start_epoch, training_configurations[args.model]['epochs']):
        #adjust_learning_rate(optimizer1, epoch + 1)
        #adjust_learning_rate(optimizer2, epoch + 1)
        
        # train for one epoch
        train(train_loaders, workers, projection_heads, args.workers, optimizers, proj_head_optimizers, epoch)
        
        # evaluate on validation set
        prec = validate(val_loader, workers, projection_heads, args.workers, epoch)
        
        avg_prec=0
        for num in range(args.workers):
            # remember best prec@1 and save checkpoint
            is_best = prec[num] > best_prec_[num]
            avg_prec += prec[num]
            best_prec_[num] = max(prec[num], best_prec_[num])
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': workers[num].state_dict(),
                'best_acc': best_prec_[num],
                'optimizer': optimizers[num].state_dict(),
                'val_acc': val_acc_[num],
            }, is_best, checkpoint=check_point[num])
            np.savetxt(accuracy_file_[num], np.array(val_acc_[num]))
            print('Best Prec {}: '.format(num+1), best_prec_[num])
            
            if num==args.workers -1:
                avg_prec /= args.workers
                is_avg_best = avg_prec > best_avg_prec
                best_avg_prec = max(best_avg_prec, avg_prec)      
                print('Best avg accuracy: ', best_avg_prec) 
        for num in range(args.local_module_num):
            save_checkpoint({
              'epoch' : epoch+1,
              'state_dict' : projection_heads[num].state_dict(),
              'best_avg_acc' : best_avg_prec,
              'optimizer' : proj_head_optimizers[num].state_dict(),
              'val_acc' : avg_prec
            }, is_avg_best, checkpoint = ph_check_point[num])

    test_prec = validate(test_loader, workers, projection_heads, args.workers, training_configurations[args.model]["epochs"], "test", start_time)
    print("After end of Training...\n ") #Test prec 1 = ", test_prec1, " Test Prec 2 = ", test_prec2, " Test Prec 3 = ", test_prec3, " Avg Test Prec = ", (test_prec1+test_prec2+test_prec3)/3)
    avg_test = 0
    for num in range(args.workers):
        print('\nTest Prec {}: '.format(num+1), test_prec[num]) 
        avg_test += test_prec[num]
    print("Average Test Accuracy = ", avg_test/len(test_prec))
    print("*****************************************************************************************")


def train(train_loaders, workers, projection_heads, num_workers, optimizers, proj_head_optimizers, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses=[]
    top=[]

    first_flag=15
    second_flag=40+(args.workers-2)*20
    if epoch <=first_flag:
      gamma_1 = args.gamma_1*np.exp(-5*(1-epoch/first_flag)**2)
      gamma_2 = args.gamma_2*np.exp(-5*(1-epoch/first_flag)**2)
    elif epoch>=second_flag:
      gamma_1 = args.gamma_1*np.exp(-12.5*(1-(args.epoch - epoch)/second_flag)**2)
      gamma_2 = args.gamma_2*np.exp(-12.5*(1-(args.epoch - epoch)/second_flag)**2)
    else:
      gamma_1 = args.gamma_1
      gamma_2 = args.gamma_2

    for num in range(num_workers):
      losses.append(AverageMeter())
      top.append(AverageMeter())

    train_batches_num = len(train_loaders[0])
    for num in range(num_workers):
        workers[num].train()

    end = time.time()
    #**********************************************EDIT FOR EACH WORKER******************
    for i,batch in enumerate(zip(*train_loaders)):
          x=[batch[i][0] for i in range(0, len(train_loaders))]
          target=[batch[i][1] for i in range(0, len(train_loaders))]

          if torch.cuda.is_available():
              for num in range(num_workers):
                target[num] = target[num].cuda()
                x[num] = x[num].cuda()

          for num in range(num_workers):
              optimizers[num].zero_grad()
          proj_head_optimizers[0].zero_grad()
          proj_head_optimizers[1].zero_grad()

          
          output=[]
          chunk1_loss=[]
          chunk2_loss=[]
          hidden1=[]
          hidden2=[]
          prec=[]
          for num in range(num_workers):
              output1, chunk1_loss1, chunk2_loss1, hidden11, hidden12 = workers[num](img=x[num],
                                  target=target[num],
                                  alpha_1=args.alpha_1,
                                  beta_1=args.beta_1,
                                  beta_2=args.beta_2,
                                  ixx_2=args.ixx_2,
                                  ixy_2=args.ixy_2,
                                  gamma_1 = args.gamma_1,
                                  gamma_2 = args.gamma_2)
              output.append(output1)
              chunk1_loss.append(chunk1_loss1)
              chunk2_loss.append(chunk2_loss1)

              proj_hidden11 = projection_heads[0](hidden11)
              proj_hidden12 = projection_heads[1](hidden12)

              hidden1.append(proj_hidden11)
              hidden2.append(proj_hidden12)

              prec.append(accuracy(output[num].data, target[num], topk=(1,))[0])
              losses[num].update(chunk2_loss[num].data.item(), x[num].size(0))
              top[num].update(prec[num].item(), x[num].size(0))

          loss=[]
          contra_loss_chunk1=0
          contra_loss_chunk2=0

          for num in range(num_workers):
              loss.append(chunk1_loss[num]+chunk2_loss[num])

         # Chunk 1 Contrastive Loss Calculation 
          c1l=np.zeros([num_workers,num_workers])
          if args.gamma_1 != 0:
            for num in range(num_workers):
              for no in range(num_workers):
                n=num_workers-no-1
                if num==n:
                  break
                else:
                  c1l[num][n]=gamma_1*loss_fn(hidden1[num], hidden1[n], target[num], target[n])
                  c1l[n][num]=c1l[num][n]
            
            for num in range(num_workers):
              for n in range(num_workers):
                  l1=loss[num].item()
                  loss[num] += (c1l[num][n])/(num_workers-1)
                  l2=loss[num].item()
                  print('Contrastive Loss Added = ', l2-l1,'Other Losses = ', l1)
          
          # Chunk 2 Contrastive Loss Calculation
          c2l=np.zeros([num_workers,num_workers])
          if args.gamma_2 != 0:
            for num in range(num_workers):
              for no in range(num_workers):
                n=num_workers-no-1
                if num==n:
                  break
                else:
                  c2l[num][n]=gamma_2*loss_fn(hidden2[num], hidden2[n], target[num], target[n])
                  c2l[n][num]=c2l[num][n]
            
            for num in range(num_workers):
              for n in range(num_workers):
                  l1=loss[num].item()
                  loss[num] += (c2l[num][n])/(num_workers-1)
                  l2=loss[num].item()
                  print('Contrastive Loss Added = ', l2-l1,'Other Losses = ', l1)

          '''
          if args.gamma_1 != 0:
            for num in range(num_workers):
              for no in range(num_workers):
                n=num_workers-no-1
                if num==n:
                  break
                else:
                  contra_loss_chunk1+=args.gamma_1*loss_fn(hidden1[num], hidden1[n], target[num], target[n])

            #print("Chunk 1 contrastive :", contra_loss_chunk1.item())
            #print("Total Loss = ", loss1)
            loss[0] += contra_loss_chunk1/num_workers
            
          if args.gamma_2 != 0:
            for num in range(num_workers):
              for no in range(num_workers):
                n=num_workers-no-1
                if num==n:
                  break
                else:
                  contra_loss_chunk2+=args.gamma_2*loss_fn(hidden2[num], hidden2[n], target[num], target[n])

            loss[1] += contra_loss_chunk2/num_workers
            #print("Chunk 2 Contrastive Loss = ", contra_loss_chunk2.item())
          '''
          for num in range(num_workers-1):
            loss[num].backward(retain_graph = True) 
          loss[num_workers-1].backward()

          
          for num in range(num_workers):
            optimizers[num].step()  
          proj_head_optimizers[0].step()
          proj_head_optimizers[1].step()
          
          batch_time.update(time.time() - end)
          end = time.time()

          if (i+1) % args.print_freq == 0:
            # print(discriminate_weights)
            for num in range(num_workers):  
              fd = open(record_file[num], 'a+')
              string = ('W{0} Epoch: [{1}][{2}/{3}]\t'
                        'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                        'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                        'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(num+1,
                        epoch, i+1, train_batches_num, batch_time=batch_time,
                        loss=losses[num], top1=top[num]))
              print(string)
              fd.write(string + '\n')
              fd.close()

    return 


def validate(val_loader, workers, projection_heads, num_workers, epoch, mode="val",start_time=0):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses=[]
    top=[]
    for num in range(num_workers):
      losses.append(AverageMeter())
      top.append(AverageMeter())

    train_batches_num = len(val_loader)

    if mode != 'val':
      print("While Testing, Loading best model state...")
      c=[]
      for num in range(num_workers):
        c.append(torch.load(check_point[num]+"/model_best.pth.tar"))
      
      for num in range(num_workers):
        workers[num].load_state_dict(c[num]["state_dict"])
        print("Loaded best checkpoint from epoch ", c[num]['epoch'])
    
    # switch to evaluate mode
    for num in range(num_workers):
      workers[num].eval()
    
    end = time.time()
    for i, (x, target) in enumerate(val_loader):
        if torch.cuda.is_available():
          target = target.cuda()
          x = x.cuda()
        input_x = torch.autograd.Variable(x)
        target_var = torch.autograd.Variable(target)

        loss=[]
        output=[]
        with torch.no_grad():
          for num in range(num_workers):
            output1, loss1 = workers[num](img=input_x,
                                 target=target_var,
                                 beta_2 = args.beta_2)
            output.append(output1)
            loss.append(loss1)
            
        # measure accuracy and record loss
        prec=[]
        for num in range(num_workers):
          prec.append(accuracy(output[num].data, target, topk=(1,))[0])
          losses[num].update(loss[num].data.item(), input_x.size(0))
          top[num].update(prec[num].item(), input_x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    
    if mode=="val":
      for num in range(num_workers):
          fd = open(record_file[num], 'a+')
          string = ('W{0} Val: [{1}][{2}/{3}]\t'
                    'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                    'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                    'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(num+1,
              epoch, (i + 1), train_batches_num, batch_time=batch_time,
              loss=losses[num], top1=top[num]))
          print(string)
          fd.write(string + '\n')
          fd.close()

      for num in range(num_workers):
        val_acc_[num].append(top[num].ave)     
    else:      
      for num in range(num_workers):
        fd = open(test_file[num], 'a+')
        string = ('W{0} Test: [{1}][{2}/{3}]\t'
                  'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                  'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                  'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(num+1,
            epoch, (i + 1), train_batches_num, batch_time=batch_time,
            loss=losses[num], top1=top[num]))
        print(string)
        fd.write(string + '\n ' + str(time.time()-start_time))
        fd.close()

    return [top[num].ave for num in range(num_workers)]


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