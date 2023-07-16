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
import networks.resnet
from networks.losses import *
from data_utils import *

parser = argparse.ArgumentParser(description='InfoPro-PyTorch')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset: [cifar10|stl10|svhn|fmnist]')

parser.add_argument('--model', default='resnet', type=str,
                    help='resnet is supported currently')

parser.add_argument('--layers', default=16, type=int,
                    help='total number of layers (have to be explicitly given!)')

parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')

parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.set_defaults(augment=False)

parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
#parser.add_argument('--resume', default='', type=str,
#                    help='path to latest checkpoint (default: none)')

parser.add_argument('--resume', default=0, type=int,
                    help='chunk number to resume from')
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
parser.add_argument('--local_module_num', default=2, type=int,
                    help='number of local modules (1 refers to end-to-end training)')

parser.add_argument('--balanced_memory', dest='balanced_memory', action='store_true',
                    help='whether to split local modules with balanced GPU memory (InfoPro* in the paper)')
parser.set_defaults(balanced_memory=False)

parser.add_argument('--aux_net_config', default='1c2f', type=str,
                    help='architecture of auxiliary classifier / contrastive head '
                         '(default: 1c2f; 0c1f refers to greedy SL)'
                         '[0c1f|0c2f|1c1f|1c2f|1c3f|2c2f]')

parser.add_argument('--local_loss_mode', default='cross_entropy', type=str,
                    help='ways to estimate the task-relevant info I(x, y)'
                         '[contrast|cross_entropy]')

parser.add_argument('--aux_net_widen', default=1.0, type=float,
                    help='widen factor of the two auxiliary nets (default: 1.0)')

parser.add_argument('--aux_net_feature_dim', default=128, type=int,
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
parser.add_argument("--contrastive_loss_function", type=str, default='distance', help="Which type of contrastive loss to use. opetions available: ecosine, mcosine, distance, amc, lecosine") #required=True
parser.add_argument("--positive_weight", type=float, default=1.0, help="weight of positive loss")
parser.add_argument("--negative_weight", type=float, default=1.0, help="weight of negative loss")
parser.add_argument("--contrastive_loss_margin", type=float, default = 1.0, help="margin used in contrastive loss function")

parser.add_argument("--workers", type=int, help="number of workers")
parser.add_argument("--epochs", default=[200], type=int,nargs="*", help="epochs [s1,s2,c1,c2]")
parser.add_argument("--run", type=int, help="run count")

parser.add_argument("--random", type=int, default=0, help="random contrastive loss")
parser.add_argument("--randomint", type=int, default=0, help="no of pairs of random contrastive loss")
parser.add_argument("--rampup", type=int, default=0, help="ramp up variable")
parser.add_argument("--flag", type=int, default=10, help="start flag")

parser.add_argument("--contra_diff", type=int, default=0, help="0==s1(c1)s2(c2), 1==s1c1s2c2, 2==s1s2c1c2, 3==c1s1c2s2, 4==c1c2s1s2")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
parser.add_argument("--skewed", type=int, default=0, help="varied dataset")
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

if args.contra_diff==0:
  epochsC1=0
  epochsC2=0
  if len(args.epochs)==1:
    epochsS1=args.epochs[0]
    epochsS2=args.epochs[0]
  elif len(args.epochs)==2:
    epochsS1=args.epochs[0]
    epochsS2=args.epochs[1]

elif args.contra_diff>0:
  if len(args.epochs)==1:
    epochsS1=args.epochs[0]
    epochsC1=args.epochs[0]
    epochsC2=args.epochs[0]
    epochsS2=args.epochs[0]
  elif len(args.epochs)==2:
    epochsS1=args.epochs[0]
    epochsC1=args.epochs[1]
    epochsC2=args.epochs[1]
    epochsS2=args.epochs[0]
  elif len(args.epochs)==3:
    epochsS1=args.epochs[0]
    epochsC1=args.epochs[1]
    epochsC2=args.epochs[2]
    epochsS2=args.epochs[0]
  elif len(args.epochs)==4:
    epochsS1=args.epochs[0]
    epochsC1=args.epochs[1]
    epochsC2=args.epochs[2]
    epochsS2=args.epochs[3]

global epochs
epochs=[epochsS1,epochsS2,epochsC1,epochsC2]

# Configurations adopted for training deep networks.
training_configurations = {
    'resnet': {
        'epochs': epochs,
        "skewness_ratio" : args.skewness_ratio, 
        'batch_size': 128,
        'initial_learning_rate': args.lr,
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
              + ('/s1(c1)s2(c2)/' if args.contra_diff==0 else '') \
              + ('/s1c1s2c2/' if args.contra_diff==1 else '') \
              + ('/s1s2c1c2/' if args.contra_diff==2 else '') \
              + ('/c1s1c2s2/' if args.contra_diff==3 else '') \
              + ('/c1c2s1s2/' if args.contra_diff==4 else '') \
              + 'worker_' + str(args.workers) +'/' \
              + ('_rampup_' + str(args.flag) if args.rampup else '') \
              + ('_random_' + str(args.randomint) if args.random else '') \
              + ('_skewed_' if args.skewed else '') \
              + 'alpha_' + str(args.alpha_1) \
              + '_beta_1_' + str(args.beta_1) \
              + '_beta_2_' + str(args.beta_2) \
              + '_gamma_1_' + str(args.gamma_1) \
              + '_gamma_2_' + str(args.gamma_2) \
              + '_epochs_' + str(epochs) \
              + '_run' + str(args.run) \
              + '_exp'

print("Storing in ...", record_path)

record_file_={}
accuracy_file_={}
check_point_={}
cp_={}
test_file=[]

for n in range(2):
  record_file_[n]=[]
  for num in range(args.workers):
    record_file_[n].append(record_path + '/training_process_Chunk{}_{}.txt'.format(n,num))

for n in range(4):
  cp_[n]=[]
  check_point_[n]=[]
  accuracy_file_[n]=[]
  for num in range(args.workers):
    accuracy_file_[n].append(record_path + '/accuracy_epoch_Chunk{}_{}.txt'.format(n,num))
    cp_[n].append(args.checkpoint + '_{}_'.format(n) + str(num))
    check_point_[n].append(os.path.join(record_path, cp_[n][num]))

for num in range(args.workers):
  test_file.append(record_path + "/test{}.txt".format(num))

pretrain = True
    
def sup_contra_train(start_epoch,val_loader,train_loaders,workers,optimizers,mode,chunknum):
  if mode=='supervised':
    echunk=epochs[chunknum]
    ##training learning
    for epoch in range(start_epoch[chunknum], echunk):#training_configurations[args.model][echunk]):
      r=train(train_loaders, workers, args.workers, optimizers, epoch, chunknum,contra='supervised')
      #supervised learning
      prec = validate(val_loader, workers, args.workers, epoch, mode="val",chunknum=chunknum,contra='supervised')
      for num in range(args.workers):
        accuracy_file=accuracy_file_[chunknum]
        best_prec=best_prec_[chunknum]
        val_acc=val_acc_[chunknum]
        best_avg_prec=best_avg_prec_[chunknum]

        is_best = prec[num] > best_prec[num]
        best_prec[num] = max(prec[num], best_prec[num])
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': workers[num].state_dict(),
            'best_acc': best_prec[num],
            'optimizer': optimizers[num][chunknum].state_dict(),
            'val_acc': val_acc[num],
        }, is_best, checkpoint=check_point_[chunknum][num])
        
        np.savetxt(accuracy_file[num], np.array(val_acc[num]))
        print('Best Prec {}: '.format(num+1), best_prec[num])
      
        if num==args.workers -1:
          avg_prec=0
          for n in range(args.workers):
            avg_prec+=best_prec[n]/args.workers
            best_avg_prec = max(best_avg_prec, avg_prec)      
          print('Best avg accuracy: ', best_avg_prec,'for chunk',chunknum) 
  
  if mode=='contra':
    echunk=epochs[chunknum+2]
    ##contrastive learning
    for epoch in range(start_epoch[chunknum+2], echunk): #training_configurations[args.model][echunk]):
      r=train(train_loaders, workers, args.workers, optimizers, epoch, chunknum,contra='contra')
      #contrastive learning
      prec = validate(val_loader, workers, args.workers, epoch, mode="val",chunknum=chunknum,contra='contra')
      for num in range(args.workers):
        accuracy_file=accuracy_file_[chunknum+2]
        best_prec=best_prec_[chunknum+2]
        val_acc=val_acc_[chunknum+2]
        best_avg_prec=best_avg_prec_[chunknum+2]

        is_best = prec[num] > best_prec[num]
        best_prec[num] = max(prec[num], best_prec[num])
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': workers[num].state_dict(),
            'best_acc': best_prec[num],
            'optimizer': optimizers[num][chunknum+2].state_dict(),
            'val_acc': val_acc[num],
        }, is_best, checkpoint=check_point_[chunknum+2][num])
        
        np.savetxt(accuracy_file[num], np.array(val_acc[num]))
        print('Best Prec {}: '.format(num+1), best_prec[num])
      
        if num==args.workers -1:
          avg_prec=0
          for n in range(args.workers):
            avg_prec+=best_prec[n]/args.workers
            best_avg_prec = max(best_avg_prec, avg_prec)      
          print('Best avg accuracy: ', best_avg_prec,'for chunk',chunknum) 
  
  return echunk

def main():
    global best_avg_prec_
    global best_prec_
    global val_acc_

    best_avg_prec_={}
    best_prec_={}
    val_acc_={}

    for num in range(4):
      best_avg_prec_[num]=0
      best_prec_[num]=[]
      val_acc_[num]=[]
      for n in range(args.workers):
        best_prec_[num].append(0)
        val_acc_[num].append([])

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
    
    train_indices, val_indices = sort_multi_indices(train_data, args.workers*25*10, 10, 'train',args.dataset)
    test_indices, _ = sort_multi_indices(test_data, 1000, 10, "test",args.dataset)
    
    train_worker_indices = divide_indices(train_indices, args.workers*25, 10, workers=args.workers,skewed=args.skewed)
    train_loaders, val_loader, test_loader = create_dataloader(train_data, test_data, 
    train_worker_indices, val_indices, test_indices,
    training_configurations[args.model]["batch_size"])
    
    workers=[]
    
    if args.model == 'resnet':
        for num in range(args.workers):
      
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
            workers.append(model1)
    else:
        raise NotImplementedError

    for num in range(args.workers):
      for n in range(4):
        if not os.path.isdir(check_point_[n][num]):
          mkdir_p(check_point_[n][num])

    cudnn.benchmark = True

    optimizers=[]
     
    for num in range(args.workers):
      optimizer=[]
      for n in range(4):
        optimizer.append(torch.optim.SGD(workers[num].parameters(),
                                  lr=training_configurations[args.model]['initial_learning_rate'],
                                  momentum=training_configurations[args.model]['momentum'],
                                  nesterov=training_configurations[args.model]['nesterov'],
                                  weight_decay=training_configurations[args.model]['weight_decay']))
                                  #eps = 1e-8, trust_coef = 0.001)
      
      optimizers.append(optimizer)
      if torch.cuda.is_available():
        workers[num] = torch.nn.DataParallel(workers[num]).cuda()
  
    start_epoch=[]
    #if args.resume==0:
    for n in range(4):
        start_epoch.append(0)

    if args.resume:
      for n in range(4):
        try:
          #if args.resume==n: #s1,s2,c1,c2
          #n=n-1
          accuracy_file=accuracy_file_[n]
          best_prec=best_prec_[n]
          val_acc=val_acc_[n]

          # Load checkpoint.
          print('==> Resuming from checkpoint_{}..'.format(n))
          resumed_path_1=[]
          ckp_1=[]
          for num in range(args.workers):
            resumed_path_1.append('{}/{}/checkpoint.pth.tar'.format(record_path, cp_[n][num])) #os.path.join(record_path, cp[num]))
            assert os.path.isfile(resumed_path_1[num]), 'Error: no checkpoint directory found!'
            print(resumed_path_1)
            cp_[n][num] = os.path.dirname(resumed_path_1[num])
            ckp_1.append(torch.load(resumed_path_1[num]))
            start_epoch[n] = ckp_1[num]['epoch']
            workers[num].load_state_dict(ckp_1[num]['state_dict'])
            optimizers[num][n].load_state_dict(ckp_1[num]['optimizer'])
            val_acc[num]=ckp_1[num]['val_acc']
            best_prec[num] += ckp_1[num]['best_acc']
            np.savetxt(accuracy_file[num], np.array(val_acc[num]))
            
        except:
          #args.resume=n+2
          #if n==4:
          start_epoch[n] = 0
          print('nothing to restore in state {}\n'.format(n))
    
    print('---start_epoch',start_epoch)

    # 0-->s1(c1)s2(c2), 1-->s1c1s2c2, 2-->s1s2c1c2, 3-->c1s1c2s2, 4-->c1c2s1s2
    for chunknum in range(args.local_module_num): #2
      if args.contra_diff>2:
        echunk=sup_contra_train(start_epoch,val_loader,train_loaders,workers,optimizers,'contra',chunknum)

      if args.contra_diff<4:
        echunk=sup_contra_train(start_epoch,val_loader,train_loaders,workers,optimizers,'supervised',chunknum)

      if args.contra_diff==1:
        echunk=sup_contra_train(start_epoch,val_loader,train_loaders,workers,optimizers,'contra',chunknum)

    if args.contra_diff==2:
      for chunknum in range(args.local_module_num): #2
        echunk=sup_contra_train(start_epoch,val_loader,train_loaders,workers,optimizers,'contra',chunknum)

    if args.contra_diff==4:
      for chunknum in range(args.local_module_num): #2
        echunk=sup_contra_train(start_epoch,val_loader,train_loaders,workers,optimizers,'supervised',chunknum)
      
    test_prec = validate(test_loader, workers, args.workers, echunk, "test", start_time=start_time,chunknum=chunknum)
    print("After end of Training...\n")
    avg_test = 0
    for num in range(args.workers):
        print('\nTest Prec {}: '.format(num+1), test_prec[num],'for chunk',chunknum) 
        avg_test += test_prec[num]
    print("Average Test Accuracy = ", avg_test/len(test_prec),'for chunk',chunknum)
    print("*****************************************************************************************")
    print("*****************************************************************************************")
    print("*****************************************************************************************")

def train(train_loaders, workers, num_workers, optimizers, epoch, chunknum, contra='supervised'):
  """Train for one epoch on the training set"""
  #print('training ',train_loaders)
  batch_time = AverageMeter()
  losses=[]
  top=[]
  
  for num in range(num_workers):
    losses.append(AverageMeter())
    top.append(AverageMeter())
  
  for num in range(num_workers):
    workers[num].train()

  train_batches_num = len(train_loaders[0])

  if contra=='supervised':
    if chunknum==0:
      alpha_1=args.alpha_1
      beta_1=args.beta_1
      beta_2=0
      gamma_2=0
      gamma_1=0

      if args.contra_diff==0:
        gamma_1=args.gamma_1

        if args.rampup==1:
          first_flag=args.flag
          if epoch <=first_flag:
            gamma_1 = args.gamma_1*np.exp(-5*(1-epoch/first_flag)**2)
          else:
            gamma_1 = args.gamma_1

    if chunknum==1:
      alpha_1=0
      beta_1=0
      gamma_1=0
      beta_2=args.beta_2
      gamma_2=0
      
      if args.contra_diff==0:
        gamma_2=args.gamma_2
        
        if args.rampup==1:
          first_flag=args.flag
          if epoch <=first_flag:
            gamma_2 = args.gamma_2*np.exp(-5*(1-epoch/first_flag)**2)
          else:
            gamma_2 = args.gamma_2


  if contra=='contra':
    alpha_1=0
    beta_1=0
    beta_2=0
    gamma_1=0
    gamma_2=0

    if chunknum==0:
      gamma_1 = args.gamma_1

      if args.rampup==1:
        first_flag=args.flag
        if epoch <=first_flag:
          gamma_1 = args.gamma_1*np.exp(-5*(1-epoch/first_flag)**2)
        else:
          gamma_1 = args.gamma_1
          
    if chunknum==1:
      gamma_2=args.gamma_2
      
      if args.rampup==1:
        first_flag=args.flag
        if epoch <=first_flag:
          gamma_2 = args.gamma_2*np.exp(-5*(1-epoch/first_flag)**2)
        else:
          gamma_2 = args.gamma_2    
    
  end = time.time()
  for i,batch in enumerate(zip(*train_loaders)):
    x=[batch[i][0] for i in range(0, len(train_loaders))]
    target=[batch[i][1] for i in range(0, len(train_loaders))]

    if torch.cuda.is_available():
      for num in range(num_workers):
        target[num] = target[num].cuda()
        x[num] = x[num].cuda()

    for num in range(num_workers):
      if contra=='supervised':
        optimizers[num][chunknum].zero_grad()
      if contra=='contra':
        optimizers[num][chunknum+2].zero_grad()
      
    output=[]
    chunk1_loss=[]
    hidden1=[]
    prec=[]
    loss=[]

    if chunknum==0:
      gamma=gamma_1
    if chunknum==1:
      gamma=gamma_2

    if gamma!=0:
      c1l=np.zeros([num_workers,num_workers]) #chunkloss
      for num in range(num_workers):
        if args.random==0 or (args.random==1 and args.randomint>=num_workers-1):
          nlist=[x for x in range(num_workers) if x!=num]  
        elif args.random==1:
          nlist=[]
          while(len(nlist)!=args.randomint):
            a=random.randint(0,num_workers-1)
            if a !=num:
              if a not in nlist:
                nlist.append(a)
      
    for num in range(num_workers):
      output1, chunk1_loss1, chunk2_loss1, hidden11, hidden12= workers[num](img=x[num],
                          target = target[num],
                          alpha_1 = alpha_1, beta_1=beta_1,
                          gamma_1 = gamma_1, beta_2=beta_2,
                          gamma_2 = gamma_2, chunknum=chunknum)
                          
      output.append(output1)
      if chunknum==0:
        hidden1.append(hidden11)
        chunk1_loss.append(chunk1_loss1)
      if chunknum==1:
        hidden1.append(hidden12)
        chunk1_loss.append(chunk2_loss1)

      loss.append(chunk1_loss[num])
      prec.append(accuracy(output[num].data, target[num], topk=(1,))[0])
      losses[num].update(chunk1_loss[num].data.item(), x[num].size(0))
      top[num].update(prec[num].item(), x[num].size(0))
   
    if gamma!=0:
      for num in range(num_workers): 
          for n in nlist:
              c1l[num][n]=gamma*loss_fn(hidden1[num], hidden1[n], target[num], target[n])
              c1l[n][num]=c1l[num][n]
              l1=loss[num].item()
              loss[num] += (c1l[num][n])/(len(nlist))
              #chunk1_loss[num] += (c1l[num][n])/(len(nlist))
              l2=loss[num].item()
              if num==0:
                print('Contrastive Loss Added = ', l2-l1,'Other Losses = ', l1,'num',num,'n',n,'c',chunknum)
    
    for num in range(num_workers):
      if num==num_workers-1:
        loss[num_workers-1].backward() 
      else:
        loss[num].backward(retain_graph = True) 
      if contra=='supervised':
        optimizers[num][chunknum].step()
      if contra=='contra':
        optimizers[num][chunknum+2].step() 

    batch_time.update(time.time() - end)
    end = time.time()

    if (i+1) % args.print_freq == 0:
      for num in range(num_workers):  
        fd = open(record_file_[chunknum][num], 'a+')
        
        string = ('W{0} C{1} Epoch: [{2}][{3}/{4}]\t'
                  'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                  'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                  'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(num+1,
                  chunknum+1, epoch, i+1, train_batches_num, batch_time=batch_time,
                  loss=losses[num], top1=top[num]))
        print(string)
        fd.write(string + '\n')
        fd.close()
  
  return hidden1


def validate(val_loader, workers, num_workers, epoch, mode="val",start_time=0,chunknum=0,contra='supervised'):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses=[]
    top=[]
    for num in range(num_workers):
      losses.append(AverageMeter())
      top.append(AverageMeter())
    
    if contra=='contra':
      echunk=chunknum+2
    elif contra=='supervised':
      echunk=chunknum

    train_batches_num = len(val_loader)

    if mode != 'val':
      print("While Testing, Loading best model state...")
      c=[]
      for num in range(num_workers):
        c.append(torch.load(check_point_[echunk][num]+"/model_best.pth.tar"))
        
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
        prec=[]

        with torch.no_grad():
          for num in range(num_workers):
            output1, loss1 = workers[num](img=input_x, beta_1=args.beta_1,
                                 target=target_var, beta_2=args.beta_2)
            output.append(output1)
            loss.append(loss1)
            
            prec.append(accuracy(output[num].data, target, topk=(1,))[0])
            losses[num].update(loss[num].data.item(), input_x.size(0))
            top[num].update(prec[num].item(), input_x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if mode=="val":
      for num in range(num_workers):
        fd = open(record_file_[chunknum][num], 'a+')
      
        string = ('W{0} C{1} Val: [{2}][{3}/{4}]\t'
                    'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                    'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                    'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(num+1,
              chunknum+1, epoch, (i + 1), train_batches_num, batch_time=batch_time,
              loss=losses[num], top1=top[num]))
        print(string)
        fd.write(string + '\n')
        fd.close()

      for num in range(num_workers):
        val_acc_[echunk][num].append(top[num].ave)     
    
    else:      
      for num in range(num_workers):
        fd = open(test_file[num], 'a+')
        string = ('W{0} C{1} Test: [{2}][{3}/{4}]\t'
                  'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                  'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                  'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(num+1,
            chunknum+1, epoch, (i + 1), train_batches_num, batch_time=batch_time,
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