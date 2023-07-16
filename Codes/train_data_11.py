import argparse
import os
import shutil
import time
import errno
import random
import math
import numpy as np
import pandas as pd
import copy
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
import networks.resnet_3
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

parser.add_argument('--skewness_ratio', default=0.5, type=float,)  # Skewness between labels

parser.add_argument("--workers", type=int, help="number of workers")
parser.add_argument("--epochs", default=[200], type=int,nargs="*", help="epochs [s1,s2,c1,c2]")
parser.add_argument("--run", type=int, help="run count")

parser.add_argument("--random", type=int, default=0, help="random contrastive loss")
parser.add_argument("--randomint", type=int, default=0, help="no of pairs of random contrastive loss")
parser.add_argument("--rampup", type=int, default=0, help="ramp up variable")
parser.add_argument("--flag", type=int, default=10, help="start flag")

parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
parser.add_argument("--skewed", type=int, default=0, help="varied dataset")
parser.add_argument("--num", type=int, default=10, help="c1 get c2")
parser.add_argument("--version", type=int, default=0, help="version")
parser.add_argument("--rounds", type=int, default=0, help="local epoch repeat")
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
'''
if len(args.epochs)==1:
  epochsC1=args.epochs[0]
  epochsC2=args.epochs[0]
elif len(args.epochs)==2:
  epochsC1=args.epochs[0]
  epochsC2=args.epochs[1]

global epochs
epochs=[epochsC1,epochsC2]
'''
global epochs
epochs=args.epochs

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
              + 'SplitFed/worker_' + str(args.workers) +'/orig_sfl_version_' \
              + str(args.version) + '/' \
              + ('_rampup_' + str(args.flag) if args.rampup else '') \
              + ('_random_' + str(args.randomint) if args.random else '') \
              + ('_skewed_' if args.skewed else '') \
              + '_rounds_' + str(args.rounds) \
              + '_epochs_' + str(epochs) \
              + '_run' + str(args.run) \

print("Storing in ...", record_path)

record_file_={}
accuracy_file_={}
check_point_1=[]
check_point_2=[]
cp_={}
test_file=[]
args.checkpoint=['checkpoint_0','checkpoint_1']

for n in range(2):
  record_file_[n]=[]
  for num in range(args.workers):
    record_file_[n].append(record_path + '/training_process_Chunk{}_w{}.txt'.format(n,num))

for n in range(2):
  cp_[n]=[]
  accuracy_file_[n]=[]
  for num in range(args.workers):
    accuracy_file_[n].append(record_path + '/accuracy_epoch_Chunk{}_w{}.txt'.format(n,num))
    cp_[n].append(args.checkpoint[n] + '_w' + str(num))
    if n==0:
      check_point_1.append(os.path.join(record_path, cp_[n][num]))
    elif n==1:
      check_point_2.append(os.path.join(record_path, cp_[n][num]))

for num in range(args.workers):
  test_file.append(record_path + "/test{}.txt".format(num))

pretrain = True
    

def main():
    global best_avg_prec_
    global best_prec_
    global val_acc_

    best_avg_prec_={}
    best_prec_={}
    val_acc_={}

    for num in range(2):
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
    global train_loaders
    global val_loader
    global test_loader
    train_loaders, val_loader, test_loader = create_dataloader(train_data, test_data, 
    train_worker_indices, val_indices, test_indices,
    training_configurations[args.model]["batch_size"])
    
    global workers
    workers=[]
    
    if args.model == 'resnet':
        for num in range(args.workers):
            model1 = eval('networks.resnet_3.resnet' + str(args.layers) + '_chunk1')\
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
            model2 = eval('networks.resnet_3.resnet' + str(args.layers) + '_chunk2')\
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
            
            workers.append([model1,model2])

    else:
        raise NotImplementedError

    for num in range(args.workers):
      if not os.path.isdir(check_point_1[num]):
          mkdir_p(check_point_1[num])
      if not os.path.isdir(check_point_2[num]):
          mkdir_p(check_point_2[num])

    cudnn.benchmark = True

    global optimizers
    optimizers=[]
     
    for num in range(args.workers):
      optimizer=[]
      for n in range(2):
        optimizer.append(torch.optim.SGD(workers[num][n].parameters(),
                                  lr=training_configurations[args.model]['initial_learning_rate'],
                                  momentum=training_configurations[args.model]['momentum'],
                                  nesterov=training_configurations[args.model]['nesterov'],
                                  weight_decay=training_configurations[args.model]['weight_decay']))
      optimizers.append(optimizer)
      
      if torch.cuda.is_available():
        for c in range(2):
          workers[num][c] = torch.nn.DataParallel(workers[num][c]).cuda()
          
    '''
    start_epoch=[]
    #if args.resume==0:
    for n in range(2):
        start_epoch.append(0)

    if args.resume:
      for n in range(2):
        try:
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
            #cp_[n][num] = os.path.dirname(resumed_path_1[num])
            ckp_1.append(torch.load(resumed_path_1[num]))
            start_epoch[n] = ckp_1[num]['epoch']
            workers[num][n].load_state_dict(ckp_1[num]['state_dict'])
            optimizers[num][n].load_state_dict(ckp_1[num]['optimizer'])
            val_acc[num]=ckp_1[num]['val_acc']
            best_prec[num] += ckp_1[num]['best_acc']
            np.savetxt(accuracy_file[num], np.array(val_acc[num]))
            
        except:
          start_epoch[n] = 0
          print('nothing to restore in state {}\n'.format(n))
    '''
    start_epoch=0
    print('---start_epoch',start_epoch)

    # 0-->s1s2avgs1
    a=[]
    sup_contra_train(start_time,start_epoch,args.rounds)


    test_prec = validate(test_loader, args.workers, epochs[0], "test", start_time=start_time)
    print("After end of Training...\n")
    avg_test = 0
    for num in range(args.workers):
        print('\nTest Prec {}: '.format(num), test_prec[num]) 
        avg_test += test_prec[num]
    print("Average Test Accuracy = ", avg_test/len(test_prec))
    print("*****************************************************************************************")
    print("*****************************************************************************************")
    print("*****************************************************************************************")

def sup_contra_train(start_time, start_epoch, rounds):
    ##training learning
    for epoch in range(rounds):#training_configurations[args.model][echunk]):
      train(train_loaders, args.workers, epoch, epochs[0], start_epoch)
      print('train done**********************************')
      #supervised learning
      prec = validate(val_loader, args.workers, epoch, mode="val")
      print('val done**********************************')
      '''
      for num in range(args.workers):
        for chunknum in range(2):
          accuracy_file=accuracy_file_[chunknum]
          best_prec=best_prec_[chunknum]
          val_acc=val_acc_[chunknum]
          best_avg_prec=best_avg_prec_[chunknum]
          is_best = prec[num] > best_prec[num]
          best_prec[num] = max(prec[num], best_prec[num])

          if chunknum==0:
            save_checkpoint({
                'rounds': epoch ,
                'state_dict': workers[num][chunknum].state_dict(),
                'best_acc': best_prec[num],
                'optimizer': optimizers[num][chunknum].state_dict(),
                'val_acc': val_acc[num],
            }, is_best, checkpoint=check_point_1[num])
          elif chunknum==1:
            save_checkpoint({
                'rounds': epoch ,
                'state_dict': workers[num][chunknum].state_dict(),
                'best_acc': best_prec[num],
                'optimizer': optimizers[num][chunknum].state_dict(),
                'val_acc': val_acc[num],
            }, is_best, checkpoint=check_point_2[num])

          np.savetxt(accuracy_file[num], np.array(val_acc[num]))
          print('Best Prec {}: '.format(num), best_prec[num])
        
        if num==args.workers -1:
          avg_prec=0
          for n in range(args.workers):
            avg_prec+=best_prec[n]/args.workers
            best_avg_prec = max(best_avg_prec, avg_prec)      
          print('Best avg accuracy: ', best_avg_prec)
      '''   
      test_prec = validate(test_loader, args.workers, epoch, "test", start_time=start_time)
      avg_test = 0
      for num in range(args.workers):
        print('\nTest Prec {}: '.format(num), test_prec[num]) 
        avg_test += test_prec[num]
      print("Average Test Accuracy = ", avg_test/len(test_prec))

def train_new(train_loaders, num_workers, epoch, chunknum):
  """Train for one epoch on the training set"""
  #print('training ',train_loaders)
  batch_time = AverageMeter()
  losses=[]
  top=[]
  
  for num in range(num_workers):
    losses.append(AverageMeter())
    top.append(AverageMeter())
  
  for num in range(num_workers):
    #for ch in range(chunknum+1):
    workers[num][chunknum].train()

  train_batches_num = len(train_loaders[0])

  end = time.time()
  for i,batch in enumerate(zip(*train_loaders)):
    x=[batch[i][0] for i in range(0, len(train_loaders))]
    target=[batch[i][1] for i in range(0, len(train_loaders))]

    if torch.cuda.is_available():
      for num in range(num_workers):
        target[num] = target[num].cuda()
        x[num] = x[num].cuda()

    for num in range(num_workers):
      #for ch in range(chunknum+1):
      optimizers[num][chunknum].zero_grad()
      
    output=[]
    chunk_loss=[]
    prec=[]
    loss=[]

    for num in range(num_workers):
      
      #if chunknum==1:
      #if epoch==0:
      #  c=torch.load(check_point_1[num]+"/model_best.pth.tar")
      #  workers[num][0].load_state_dict(c["state_dict"]) #,strict=False)
      #  print("Loaded best checkpoint from epoch ", c['epoch'],'w{}'.format(num))
      #  workers[num][0].eval()
      
      output1, chunk1_loss1, hidden11= workers[num][0](img=x[num],target = target[num])
      
      if chunknum==1:
        if args.version==0:
          hidden11=hidden11.clone().detach()
        elif args.version==1:
          if epoch%args.num!=0:
            hidden11=hidden11.clone().detach()
        
        output2, chunk2_loss1 = workers[num][1](img=hidden11,
                          target = target[num])
      
      if chunknum==0:
        output.append(output1)
        chunk_loss.append(chunk1_loss1)
        loss.append(chunk_loss[num])
      elif chunknum==1:
        output.append(output2)
        chunk_loss.append(chunk2_loss1)
        loss.append(chunk_loss[num])
      
      prec.append(accuracy(output[num].data, target[num], topk=(1,))[0])
      if chunknum==0:
        losses[num].update(chunk_loss[num].data.item(), x[num].size(0))
        top[num].update(prec[num].item(), x[num].size(0))
      elif chunknum==1:
        losses[num].update(chunk_loss[num].data.item(), hidden11.size(0))
        top[num].update(prec[num].item(), hidden11.size(0))
      
   
    for num in range(num_workers):
      if num==num_workers-1:
        loss[num_workers-1].backward() 
      else:
        loss[num].backward(retain_graph = True) 
      
      #for ch in range(2):
      optimizers[num][chunknum].step()
      
    batch_time.update(time.time() - end)
    end = time.time()

    #if (i+1) % args.print_freq == 0:
    for num in range(num_workers):  
        for ch in range(2):
          fd = open(record_file_[ch][num], 'a+')
          
          string = ('W{0} C{1} Epoch: [{2}][{3}/{4}]\t'
                    'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                    'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                    'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(num+1,
                    ch+1, epoch, i+1, train_batches_num, batch_time=batch_time,
                    loss=losses[num], top1=top[num]))
          print(string)
          fd.write(string + '\n')
          fd.close()


def train(train_loaders, num_workers, rounds, epochs, start_epoch):
  """Train for one epoch on the training set"""
  for chunknum in range(2):
    for epoch in range(start_epoch, epochs):#training_configurations[args.model][echunk]):
      train_new(train_loaders, num_workers, epoch, chunknum)



def validate(val_loader, num_workers, epoch, mode="val",start_time=0):
  """Perform validation on the validation set"""
  batch_time = AverageMeter()
  losses2=[]
  top=[]
  output=[]
  prec=[]
  train_batches_num = len(val_loader)
  #if epoch>0:
  #  load_wts()

  for num in range(num_workers):
    losses2.append(AverageMeter())
    top.append(AverageMeter())
    for ch in range(2): 
      workers[num][ch].eval()

    end = time.time()
    for i, (x, target) in enumerate(val_loader):
      if torch.cuda.is_available():
        target = target.cuda()
        x = x.cuda()
      input_x = torch.autograd.Variable(x)
      target_var = torch.autograd.Variable(target)

      with torch.no_grad():
        output1, loss1, hidden11 = workers[num][0](img=input_x,
                            target=target_var)

        output1, loss2 = workers[num][1](img=hidden11,
                            target=target_var)

        prec.append(accuracy(output1.data, target, topk=(1,))[0])
        86.5234375
        
        losses2[num].update(loss2.data.item(), hidden11.size(0))
        top[num].update(prec[num].item(), hidden11.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if mode=="val":
          if i+1 == train_batches_num:
            for ch in range(2):
              fd = open(record_file_[ch][num], 'a+')
              string = ('W{0} C{1} Val: [{2}][{3}/{4}]\t'
                        'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                        'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                        'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(num+1,
                        ch+1, epoch, (i + 1), train_batches_num, batch_time=batch_time,
                        loss=losses2[num], top1=top[num]))
              print(string)
              fd.write(string + '\n')
              fd.close()

      for chunknum in range(2): 
        val_acc_[chunknum][num].append(top[num].ave)     
    
      else:      
        if i+1 == train_batches_num:
          for chunknum in range(2):
            fd = open(test_file[num], 'a+')
            
            string = ('W{0} C{1} Test: [{2}][{3}/{4}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(num+1,
                      chunknum+1, epoch, (i + 1), train_batches_num, batch_time=batch_time,
                      loss=losses2[num], top1=top[num]))
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


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
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