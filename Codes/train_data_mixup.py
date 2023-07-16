import argparse
import os
import shutil
import time
import errno
import random
import math
import numpy as np
import pandas as pd
#import pympler.asizeof as asizeof

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

import networks.resnet_mixup
from networks.losses import *
from data_utils_2 import *

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

parser.add_argument('--skewness_ratio', default=0.5, type=float,)  # Skewness between labels

parser.add_argument("--workers", type=int, help="number of workers")
parser.add_argument("--epoch", type=int, help="number of epochs")
parser.add_argument("--run", type=int, help="run count")

parser.add_argument('--batch_size', type=int, default=128, help='batch size of clients.')
parser.add_argument('--mixing_mode' , type=str, default='pooling', help = 'mixing method used on worker outputs')
parser.add_argument('--local_label_size', type=int, default=25, help='No of samples of one label in a client dataset')
parser.add_argument('--lr', type=float, default=0.1, help='learning_rate for client')
parser.add_argument('--lr_pow', type=float, help = 'Power variable in server lr function')
parser.add_argument('--comm_flag', type=float, default=0.0, help='When to stop averaged gradients')
parser.add_argument('--extra', type=str, help='Any extra file name extensions')

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
'''
# Configurations adopted for training deep networks.
'''
training_configurations = {
    'resnet': {
        'epochs': args.epoch,
        "skewness_ratio" : args.skewness_ratio, 
        'batch_size': 128 if args.dataset in ['cifar10', 'svhn', 'fmnist'] else 128,
        'initial_learning_rate': 1e-1 if args.dataset in ['cifar10', 'svhn', 'fmnist'] else 0.1,
        'changing_lr': [int(args.epoch*0.1),int(args.epoch*0.2),
        int(args.epoch*0.3),int(args.epoch*0.4),int(args.epoch*0.5),
        int(args.epoch*0.6),int(args.epoch*0.7),int(args.epoch*0.8),int(args.epoch*0.9)],
        'lr_decay_rate': 0.8,
        'momentum': 0.9,         
        'nesterov': False,
        'weight_decay': 1e-6,
    }
}
'''
training_configurations = {
    'resnet': {
        'epochs': args.epoch,
        "skewness_ratio" : args.skewness_ratio, 
        'batch_size': args.batch_size if args.dataset in ['cifar10', 'svhn', 'fmnist'] else 128,
        'initial_learning_rate': args.lr if args.dataset in ['cifar10', 'svhn', 'fmnist'] else 0.1,
        'changing_lr': [int(args.epoch*0.8)],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,         
        'nesterov': False,
        'weight_decay': 1e-6,
    }
}
record_path = './' \
              + 'MixUp_' \
              + str(args.dataset) \
              + '_' + str(args.model) + str(args.layers) \
              + '_K_' + str(args.local_module_num) \
              + '_' + str(args.name) \
              + '/multiworker/week13/worker_' + str(args.workers) + '/' \
              + '_mixing_mode_' + str(args.mixing_mode) \
              + '_local_label_size_' + str(args.local_label_size) \
              + '_batchsize_' + str(args.batch_size) \
              + '_epochs_' + str(args.epoch) \
              + '_lr_' + str(args.lr) \
              + '_lrpow_' + str(args.lr_pow) \
              + '_run_' + str(args.run) \
              + '_' + str(args.extra)

print("Storing in ...", record_path)

check_point=[]
accuracy_file_ = []
test_file_ = []
for num in range(args.workers):    
    cp = args.checkpoint + '_' + str(num)
    check_point.append(os.path.join(record_path, cp))
    accuracy_file_.append(record_path + '/accuracy_epoch{}.txt'.format(num+1))
    test_file_.append(record_path + "/test{}.txt".format(num+1))


record_file = record_path + '/training_process.txt'
accuracy_file = record_path + '/accuracy_epoch.txt'
test_file = record_path + "/test.txt"
loss_file = record_path + '/loss_epoch.txt'

central_cp = 'central_' + args.checkpoint
central_check_point = os.path.join(record_path, central_cp)


def main():
    global best_prec
    best_prec = []
    global val_acc
    val_acc = []
    global best_avg_prec
    best_avg_prec = 0
    

    for num in range(args.workers):
      val_acc.append([])
      best_prec.append(0)
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
    
    train_indices, val_indices = sort_multi_indices(train_data, args.workers*args.local_label_size, 10, 'train',args.dataset)
    test_indices, _ = sort_multi_indices(test_data, 1000, 10, "test",args.dataset)
  
    
    # PROPOSED
    train_worker_indices = divide_indices(train_indices, args.workers*args.local_label_size, 10, workers=args.workers)
    assert_distribution(train_data, train_worker_indices, None)
    #assert_distribution(train_data, None, val_indices)
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
            
            model = eval('networks.resnet_mixup.resnet' + str(args.layers) + '_chunk1')\
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

    for num in range(args.workers):
      if not os.path.isdir(check_point[num]):
        mkdir_p(check_point[num])

    cudnn.benchmark = True
    central_server_wide_list = (16, 16, 32, 64)
    # If concat architecture, with concat dims is channels
    if args.mixing_mode == 'concat_channels':
      central_server_wide_list = list(central_server_wide_list)
      central_server_wide_list[2] = central_server_wide_list[2]*args.workers
      central_server_wide_list = tuple(central_server_wide_list)
    
    central_server = eval('networks.resnet_mixup.resnet' + str(args.layers) + '_chunk2')\
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
                aux_net_feature_dim=args.aux_net_feature_dim, nworkers=args.workers, mixing = args.mixing_mode)
    
    if not os.path.isdir(central_check_point):
        mkdir_p(central_check_point)

    if torch.cuda.is_available():
      for num in range(args.workers):
        workers[num] = workers[num].cuda()
      central_server = central_server.cuda()


    optimizers=[]
    if args.mixing_mode in ['concat_batch', 'concat_batch_pool']:
      server_lr = training_configurations[args.model]['initial_learning_rate']*(args.workers**args.lr_pow)
    else:
      server_lr = training_configurations[args.model]['initial_learning_rate']

    print('Client Lr = ', training_configurations[args.model]['initial_learning_rate'])
    print('Server Lr = ', server_lr)

    central_optimizer = torch.optim.Adam(central_server.parameters(), 
    lr=server_lr)#,
    #momentum=training_configurations[args.model]['momentum'],
    #nesterov=training_configurations[args.model]['nesterov'],
    #weight_decay=training_configurations[args.model]['weight_decay'])


    for num in range(args.workers):
        optimizer = torch.optim.Adam(workers[num].parameters(),
                                    lr=training_configurations[args.model]['initial_learning_rate'])#,
                                    #momentum=training_configurations[args.model]['momentum'],
                                    #nesterov=training_configurations[args.model]['nesterov'],
                                    #weight_decay=training_configurations[args.model]['weight_decay'])
        
        optimizers.append(optimizer)

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
        adjust_learning_rate(optimizers, central_optimizer, epoch + 1)
        
        # train for one epoch
        train(train_loaders, workers, central_server, args.workers, optimizers, central_optimizer, epoch)
        
        # evaluate on validation set
        prec = validate(val_loader, workers, central_server, args.workers, epoch)
        
        # remember best prec@1 and save checkpoint
        
        # save checkpoints for workers
        for num in range(args.workers):
          is_best = prec[num] > best_prec[num]
          best_prec[num] = max(prec[num], best_prec[num])
          save_checkpoint({
              'epoch': epoch + 1,
              'state_dict': workers[num].state_dict(),
              'best_acc': best_prec[num],
              'optimizer': optimizers[num].state_dict(),
              'val_acc': val_acc[num],
          }, is_best, checkpoint=check_point[num])
          np.savetxt(accuracy_file_[num], np.array(val_acc[num]))
          print('Best Prec {}: '.format(num+1), best_prec[num])
         
        avg_prec = sum(prec)/len(prec)
        is_avg_best = best_avg_prec < avg_prec
        best_avg_prec = max(best_avg_prec, avg_prec)
        # save checkpoint for central chunk 2
        save_checkpoint({
              'epoch': epoch + 1,
              'state_dict': central_server.state_dict(),
              'best_acc': best_avg_prec,
              'optimizer': central_optimizer.state_dict(),
              'val_acc': [sum(x)/len(val_acc) for x in zip(*val_acc)],
          }, is_avg_best, checkpoint=central_check_point)

        np.savetxt(accuracy_file, np.array(val_acc))
        print('Best Avg Prec : ', best_avg_prec)
        

    test_prec = validate(val_loader, workers, central_server, args.workers, epoch, "test", start_time)
    avg_test = 0
    for num in range(args.workers):
        print('\nTest Prec {}: '.format(num+1), test_prec[num]) 
        avg_test += test_prec[num]
    print("Average Test Accuracy = ", avg_test/len(test_prec))
    print("*****************************************************************************************")


def train(train_loaders, workers, central_server, num_workers, optimizers, central_optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top = AverageMeter()
    comm = 'unicast'

    train_batches_num = len(train_loaders[0])
    central_server.train()
    
    for num in range(num_workers):
        workers[num].train()


    end = time.time()
    for i,batch in enumerate(zip(*train_loaders)):
          x=[batch[j][0] for j in range(0, len(train_loaders))]
          target=[batch[j][1] for j in range(0, len(train_loaders))]
          mixed_target = 0
          client_st = time.time()
          
          if torch.cuda.is_available():
              for num in range(num_workers):
                target[num] = target[num].cuda()
                x[num] = x[num].cuda()
          
          # Zero Grad
          central_optimizer.zero_grad()
          for num in range(num_workers):
              optimizers[num].zero_grad()
          
          outputs=[]
          backprop = []
          hiddens=[]
          prec=0
          bsz = x[0].size(0)
          central_input = 0
         
          # Weights incase of pooling mixing mode
          if args.mixing_mode == 'pooling':
            lams = []
            dist = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))

            for b in range(bsz):
              lam=[]
              for j in range(args.workers):
                lam.append(dist.sample())
              lam=np.array(lam)
              lam = lam/sum(lam)
              lams.append(lam.tolist())
            lams = np.array(lams)

          for num in range(num_workers):
              output = workers[num](img=x[num])
              outputs.append(output.detach())
              backprop.append(output)
                              
          # Mixed Target
          targets_onehot = []
          for num in range(args.workers):
            # Onehot encode targets
            target_onehot = torch.FloatTensor(target[num].size(0), 10)
            target_onehot.zero_()
            target_onehot.scatter_(1, target[num].unsqueeze(1).data.cpu(), 1)
            targets_onehot.append(target_onehot)
          client_et = time.time()

          if args.mixing_mode == 'concat_batch_pool':
            for num in range(args.workers-1):
              outputs[num] = 0.9*outputs[num] + 0.1*outputs[num+1]
              targets_onehot[num] = 0.9*targets_onehot[num] + 0.1*targets_onehot[num+1]
            outputs[args.workers-1] = 0.9*outputs[args.workers-1] + 0.1*outputs[0]
            targets_onehot[args.workers-1] = 0.9*targets_onehot[args.workers-1] + 0.1*targets_onehot[0]

          if args.mixing_mode == 'pooling':
            for num in range(args.workers):
                weight = torch.Tensor(lams[:,num])
                target_weight = torch.unsqueeze(weight, 1)
                target_weight = target_weight.expand(-1, 10)
                input_weight = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(weight, 1), 2), 3)
                input_weight = input_weight.expand(-1, outputs[num].size(1), outputs[num].size(2), outputs[num].size(3))
                if torch.cuda.is_available():
                  input_weight = input_weight.cuda()
                central_input += torch.mul((outputs[num]), input_weight)
                mixed_target += torch.mul(targets_onehot[num], target_weight)

          if args.mixing_mode == 'concat_channels':
            central_input = torch.cat(outputs, dim=1)
          if args.mixing_mode in ['concat_batch', 'concat_batch_pool']:
            central_input = torch.cat(outputs, dim=0)

          # CODE FOR BROADCASTING. Add .detach to client outputs above  
          central_input.requires_grad = True
          central_input.retain_grad()

          # for concat arch
          if args.mixing_mode in ['concat_batch', 'concat_batch_pool']:
            mixed_target = torch.cat(targets_onehot, dim=0)
          if args.mixing_mode == 'concat_channels':
            mixed_target = torch.cat(targets_onehot, dim=1)

          if torch.cuda.is_available():
            mixed_target = mixed_target.cuda()
          mixed_target = torch.clamp(mixed_target, max=1.0)
          #print(f'Upload memory head = {calc_memory_usage(output) + calc_memory_usage(target_onehot)} MB')
          logits, loss, central_output = central_server(central_input, mixed_target)
          
          #prec=accuracy(logits.data, avg_target, topk=(1,))[0]
          losses.update(loss.data.item(), x[0].size(0))
          #top.update(prec.item(), x[0].size(0))

          # Server side gradient calculation
          loss.backward()
          central_optimizer.step()

          if args.mixing_mode in ['concat_batch', 'concat_batch_pool']:    
            global_grads = central_input.grad
            s = int(central_input.grad.size(0)/args.workers)
            averaged_global_grads = 0
            for num in range(args.workers):
              averaged_global_grads += global_grads[num*s:(num+1)*s,:]
            averaged_global_grads /= args.workers
          


          '''
          comm = 'averaged'
          if args.mixing_mode == 'pooling':
            if comm == 'unicast':
              print(f'Download memory head = {calc_memory_usage(central_input) + calc_memory_usage(torch.Tensor(lams[:,0]))} MB')
            elif comm == 'broadcast':
              print(f'Download memory head = {calc_memory_usage(central_input) + calc_memory_usage(torch.Tensor(lams))} MB')
          if args.mixing_mode == 'concat_batch':
            if comm == 'unicast':
              s = int(central_input.grad.size(0)/args.workers)
              print(f'Download memory head = {calc_memory_usage(central_input.grad[:s,:])} MB')
            elif comm == 'broadcast':
              s = int(central_input.grad.size(0)/args.workers)
              print(f'Download memory head = {calc_memory_usage(central_input.grad)} MB')
            elif comm == 'averaged':
              if epoch<0.6*args.epoch:
                print(f'Download memory head = {calc_memory_usage(averaged_global_grads)} MB')
              else:
                print(f'Download memory head = {calc_memory_usage(central_input.grad)} MB')
          '''

          
          server_et = time.time()
          
          #'''
          # Client side gradient calculation using globally communicated gradient
          for num in range(args.workers):
            if args.mixing_mode == 'pooling':
              weight = torch.Tensor(lams[:,num])
              weight = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(weight, 1), 2), 3)
              weight = weight.expand(-1, outputs[num].size(1), outputs[num].size(2), outputs[num].size(3))
              if torch.cuda.is_available():
                  weight = weight.cuda()
              backprop[num].backward(torch.mul(central_input.grad, weight))
            elif args.mixing_mode == 'concat_batch':
              s = int(central_input.grad.size(0)/args.workers)
              #backprop[num].backward(central_input.grad[num*s:(num+1)*s,:])
              if epoch < args.comm_flag*args.epoch:
                #print('Using averaged client gradients.')
                backprop[num].backward(averaged_global_grads)
              else:
                #print('Using individual client gradients.')
                backprop[num].backward(central_input.grad[num*s:(num+1)*s,:])
            elif args.mixing_mode == 'concat_batch_pool':
              s = int(central_input.grad.size(0)/args.workers)
              if num == 0:
                backprop[num].backward(0.9*central_input.grad[0*s:(1)*s,:] + 0.1*central_input.grad[(args.workers-1)*s:,:])
              else:
                backprop[num].backward(0.9*central_input.grad[num*s:(num+1)*s,:] + 0.1*central_input.grad[(num-1)*s:(num)*s,:])
          #'''
          for num in range(num_workers):
            optimizers[num].step()

          #print("Total server time = ",server_et - client_et)          
          #print('Total client time = ', time.time() - server_et + client_et - client_st)
          
          batch_time.update(time.time() - end)
          end = time.time()

          if (i+1) % args.print_freq == 0:
            # print(discriminate_weights)
            fd = open(record_file, 'a+')
            string = ('Train Epoch: [{1}][{2}/{3}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'.format(num+1,
                      epoch, i+1, train_batches_num, batch_time=batch_time,
                      loss=losses))
            #'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'
            print(string)
            fd.write(string + '\n')
            fd.close()

    return 


def validate(val_loader, workers, central_server, num_workers, epoch, mode="val",start_time=0):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses=AverageMeter()
    top=[]
    for num in range(num_workers):
      top.append(AverageMeter())

    train_batches_num = len(val_loader)

    if mode != 'val':
      print("While Testing, Loading best model state for workers...")
      c=[]
      for num in range(num_workers):
        c.append(torch.load(check_point[num]+"/model_best.pth.tar"))
      
      for num in range(num_workers):
        workers[num].load_state_dict(c[num]["state_dict"])
        print("Loaded best checkpoint from epoch ", c[num]['epoch'])
      print('Loading best model state for central server...')
      cp = torch.load(central_check_point + '/model_best.pth.tar')
      central_server.load_state_dict(cp['state_dict'])
    
    
    # switch to evaluate mode
    for num in range(num_workers):
      workers[num].eval()
    central_server.eval()
    end = time.time()
    
    for i, (x, target) in enumerate(val_loader):    
        if torch.cuda.is_available():
          target = target.cuda()
          x = x.cuda()
        
        input_x = torch.autograd.Variable(x)
        target_var = torch.autograd.Variable(target)

        outputs=[]
        logits_per_worker = []
        prec = []       # For each batch
        mixed_target = 0
        targets_onehot = []
        central_input = 0
        with torch.no_grad():
          for num in range(num_workers):
            # Each worker passes batch of data
            output = workers[num](img=input_x)
            # Append the outputs of each worker
            outputs.append(output)
            
            target_onehot = torch.FloatTensor(target.size(0), 10)
            target_onehot.zero_()
            target_onehot.scatter_(1, target.unsqueeze(1).data.cpu(), 1)
            targets_onehot.append(target_onehot)
            if args.mixing_mode == 'pooling':
              mixed_target += target_onehot
            if args.mixing_mode == 'concat_channels':
              output = output.repeat(1, 2, 1, 1)
              target_onehot = target_onehot.repeat(1, 2)
            
            if torch.cuda.is_available():
              target_onehot = target_onehot.cuda()           
            
            logits, loss, central_output = central_server(output, target_onehot)
            # Store the logits of each worker+server 
            logits_per_worker.append(logits) 

            # Keep creating central input similar to train
            if args.mixing_mode == 'pooling':
              central_input += output      
              
            # measure accuracy for each worker
            if args.mixing_mode == 'concat_channels':
              prec.append(accuracy(logits[:, num*10:(num+1)*10].data, target, topk=(1,))[0])
            else:
              prec.append(accuracy(logits.data, target, topk=(1,))[0])
            top[num].update(prec[num].item(), input_x.size(0))
            # Prepare target for loss calculation

          if args.mixing_mode == 'pooling':
            central_input = central_input/args.workers
            mixed_target = mixed_target/args.workers
            mixed_target = torch.clamp(mixed_target, max=1.0)

          if args.mixing_mode in ['concat_batch', 'concat_batch_pool']:
            central_input = torch.cat(outputs, dim=0)
            mixed_target = torch.cat(targets_onehot, dim=0)
          if args.mixing_mode == 'concat_channels':
            central_input = torch.cat(outputs, dim=1)
            mixed_target = torch.cat(targets_onehot, dim=1)

          if torch.cuda.is_available():
            central_input = central_input.cuda()
            mixed_target = mixed_target.cuda()

          # Finally calculate loss based on mixup
          _, loss,_ = central_server(central_input, mixed_target)
          losses.update(loss.data.item(), input_x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    
    if mode=="val":
      for num in range(args.workers):
        fd = open(record_file, 'a+')
        string = ('W{0} Val: [{1}][{2}/{3}]\t'
                  'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                  'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                  'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(num+1,
            epoch, (i + 1), train_batches_num, batch_time=batch_time,
            loss=losses, top1=top[num]))
        print(string)
        fd.write(string + '\n')
        fd.close()
        val_acc[num].append(top[num].ave)     
    else:      
      for num in range(args.workers):
        fd = open(test_file_[num], 'a+')
        string = ('W{0} Test: [{1}][{2}/{3}]\t'
                  'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                  'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                  'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(num+1,
            epoch, (i + 1), train_batches_num, batch_time=batch_time,
            loss=losses, top1=top[num]))
        print(string)
        fd.write(string + '\n ' + str(time.time()-start_time))
        fd.close()

    return [x.ave for x in top]


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


def adjust_learning_rate(optimizers, central_optimizer, epoch):
    """Sets the learning rate"""
    if not args.cos_lr:
        if epoch in training_configurations[args.model]['changing_lr']:
            for num in range(len(optimizers)):
              for param_group in optimizers[num].param_groups:
                  param_group['lr'] *= training_configurations[args.model]['lr_decay_rate']
              print('lr:')
              for param_group in optimizers[num].param_groups:
                print(param_group['lr'])

            for param_group in central_optimizer.param_groups:
              param_group['lr'] *= training_configurations[args.model]['lr_decay_rate']
            print('lr:')
            for param_group in central_optimizer.param_groups:
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