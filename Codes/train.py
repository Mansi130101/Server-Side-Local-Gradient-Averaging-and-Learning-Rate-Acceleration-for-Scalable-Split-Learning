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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchlars import LARS

import networks.resnet
from networks.losses import Contrastive_Loss

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
parser.add_argument('--print-freq', '-p', default=10, type=int,
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

parser.add_argument('--ixx_1', default=0.0, type=float,)   # \lambda_1 for 1st local module
parser.add_argument('--ixy_1', default=0.0, type=float,)   # \lambda_2 for 1st local module

parser.add_argument('--ixx_2', default=0.0, type=float,)   # \lambda_1 for (K-1)th local module
parser.add_argument('--ixy_2', default=0.0, type=float,)   # \lambda_2 for (K-1)th local module

parser.add_argument("--branch", dest="branch", action = "store_true", help="Whether to branch Chunk 2.")  # Whether to branch Chunk 2
parser.set_defaults(branch=False)

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



# Configurations adopted for training deep networks.
training_configurations = {
    'resnet': {
        'epochs': 50,
        'batch_size': 1024 if args.dataset in ['cifar10', 'svhn'] else 128,
        'initial_learning_rate': 5.0 if args.dataset in ['cifar10', 'svhn'] else 0.1,
        'changing_lr': [80, 120],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    }
}

record_path = './' \
              + ('InfoPro*_' if args.balanced_memory else 'InfoPro_') \
              + str(args.dataset) \
              + '_' + str(args.model) + str(args.layers) \
              + '_K_' + str(args.local_module_num) \
              + '_' + str(args.name) \
              + '/' \
              + 'no_' + str(args.no) \
              + '_aux_net_config_' + str(args.aux_net_config) \
              + '_local_loss_mode_' + str(args.local_loss_mode) \
              + '_aux_net_widen_' + str(args.aux_net_widen) \
              + '_aux_net_feature_dim_' + str(args.aux_net_feature_dim) \
              + '_ixx_1_' + str(args.ixx_1) \
              + '_ixy_1_' + str(args.ixy_1) \
              + '_ixx_2_' + str(args.ixx_2) \
              + '_ixy_2_' + str(args.ixy_2) \
              + ('_cos_lr_' if args.cos_lr else '')\
              + 'branched_highlr_'

record_file1 = record_path + '/training_process1.txt'
accuracy_file1 = record_path + '/accuracy_epoch1.txt'
loss_file1 = record_path + '/loss_epoch1.txt'
check_point1 = os.path.join(record_path, args.checkpoint)

record_file2 = record_path + '/training_process2.txt'
accuracy_file2 = record_path + '/accuracy_epoch2.txt'
loss_file2 = record_path + '/loss_epoch2.txt'
check_point2 = os.path.join(record_path, args.checkpoint)

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

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': False}

    train_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('./data', download=True, transform=transform_train,
                                                **kwargs_dataset_train),
        batch_size=training_configurations[args.model]['batch_size'], shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        datasets.__dict__[args.dataset.upper()]('./data', transform=transform_test,
                                                **kwargs_dataset_test),
        batch_size=training_configurations[args.model]['batch_size'], shuffle=False, **kwargs)

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
             aux_net_feature_dim=args.aux_net_feature_dim, 
             branch = args.branch)
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
             aux_net_feature_dim=args.aux_net_feature_dim,
             branch = args.branch)
    else:
        raise NotImplementedError

    if not os.path.isdir(check_point1):
        mkdir_p(check_point1)
    if not os.path.isdir(check_point2):
        mkdir_p(check_point2)

    cudnn.benchmark = True

    optimizer1 = LARS(optimizer = torch.optim.SGD(model1.parameters(),
                                lr=training_configurations[args.model]['initial_learning_rate'],
                                momentum=training_configurations[args.model]['momentum'],
                                nesterov=training_configurations[args.model]['nesterov'],
                                weight_decay=training_configurations[args.model]['weight_decay']),
                                eps = 1e-8, trust_coef = 0.001)
    if torch.cuda.is_available():
      model1 = torch.nn.DataParallel(model1).cuda()

    optimizer2 = LARS(optimizer = torch.optim.SGD(model2.parameters(),
                                lr=training_configurations[args.model]['initial_learning_rate'],
                                momentum=training_configurations[args.model]['momentum'],
                                nesterov=training_configurations[args.model]['nesterov'],
                                weight_decay=training_configurations[args.model]['weight_decay']), 
                                eps = 1e-8, trust_coef = 0.001)
    if torch.cuda.is_available():
      model2 = torch.nn.DataParallel(model2).cuda()
      #model2=model2.cuda()

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
        chunk2_loss = train(train_loader, model1, model2, optimizer1, optimizer2, epoch)
        print("Final Chunk 2 Loss = ", chunk2_loss)
        
        # evaluate on validation set
        prec1, prec2, _, _ = validate(val_loader, model1, model2, epoch)
        
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
        print('Best Prec 1: ', best_prec1)
        print('Best Prec 2: ', best_prec2)
        print('Best avg accuracy: ', (best_prec1+best_prec2)/2)
        np.savetxt(accuracy_file2, np.array(val_acc2))


def train(train_loader, model1, model2, optimizer1,optimizer2, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()

    losses1 = AverageMeter()
    losses2 = AverageMeter()

    top1 = AverageMeter()
    top2 = AverageMeter()

    train_batches_num = len(train_loader)

    # switch to train mode
    model1.train()
    model2.train()

    end = time.time()
    for i, (orig_x, target) in enumerate(train_loader):
        x = augmented(orig_x)
        if torch.cuda.is_available():
          target = target.cuda()
          x = x.cuda()
        
        if args.branch==True:
          optimizer1.zero_grad()
          optimizer2.zero_grad()
          
          output1, loss1, hidden1, hidden12a, hidden12b = model1(img=x,
                              target=target,
                              ixx_1=args.ixx_1,
                              ixy_1=args.ixy_1,
                              ixx_2=args.ixx_2,
                              ixy_2=args.ixy_2,
                              pretrain=pretrain)

          # measure accuracy and record loss
          prec1 = accuracy(output1.data, target, topk=(1,))[0]
          losses1.update(loss1.data.item(), x.size(0))
          top1.update(prec1.item(), x.size(0))

          aug_x = augmented(orig_x)
          if torch.cuda.is_available():
            aug_x = aug_x.cuda()

          
          output2, loss2, hidden2, hidden22a, hidden22b = model2(img=aug_x,
                              target=target,
                              ixx_1=args.ixx_1,
                              ixy_1=args.ixy_1,
                              ixx_2=args.ixx_2,
                              ixy_2=args.ixy_2,
                              pretrain=pretrain)

          if pretrain:
            # Optimise Chunk 1
            contra_loss1 = Contrastive_Loss(hidden1, hidden2, 5.0)
            #print("Chunk 1 Loss = ", contra_loss1.item())
            contra_loss1.backward()

            #Optimise Chunk 2
            
            #with torch.autograd.set_detect_anomaly(True):
            contra_loss2 = Contrastive_Loss((hidden12a + hidden12b)/2, (hidden22a + hidden22b)/2, 5.0) # Average of vectors then loss
            #print("Chunk 2 Loss (Avg Vector) = ", contra_loss2.item())
            contra_loss2.backward()

            '''
            contra_loss2a = Contrastive_Loss(hidden12a, hidden12b, 5.0)
            contra_loss2b = Contrastive_Loss(hidden12a, hidden22a, 5.0)
            contra_loss2c = Contrastive_Loss(hidden12a, hidden22b, 5.0)
            contra_loss2d = Contrastive_Loss(hidden12b, hidden22a, 5.0)
            contra_loss2e = Contrastive_Loss(hidden12b, hidden22b, 5.0)
            contra_loss2f = Contrastive_Loss(hidden22a, hidden22b, 5.0)

            contra_loss2 = contra_loss2a + contra_loss2b + contra_loss2c + contra_loss2d + contra_loss2e + contra_loss2f
            #print(f"Combined Chunk 2 Loss = {contra_loss2.item()} Avg = {contra_loss2.item()/6}")
            contra_loss2.backward()
            '''
          optimizer1.step()
          optimizer2.step()

          # measure accuracy and record loss
          prec2 = accuracy(output2.data, target, topk=(1,))[0]
          losses2.update(loss2.data.item(), aug_x.size(0))
          top2.update(prec2.item(), aug_x.size(0))
          
          batch_time.update(time.time() - end)
          end = time.time()

        else:
          optimizer1.zero_grad()
          optimizer2.zero_grad()
          output1, loss1, hidden1, hidden12 = model1(img=x,
                              target=target,
                              ixx_1=args.ixx_1,
                              ixy_1=args.ixy_1,
                              ixx_2=args.ixx_2,
                              ixy_2=args.ixy_2)

          # measure accuracy and record loss
          prec1 = accuracy(output1.data, target, topk=(1,))[0]
          losses1.update(loss1.data.item(), x.size(0))
          top1.update(prec1.item(), x.size(0))

          aug_x = augmented(orig_x)
          if torch.cuda.is_available():
            aug_x = aug_x.cuda()

          output2, loss2, hidden2, hidden22 = model2(img=aug_x,
                              target=target,
                              ixx_1=args.ixx_1,
                              ixy_1=args.ixy_1,
                              ixx_2=args.ixx_2,
                              ixy_2=args.ixy_2)


          #print("Hidden 1 computerd. Size = ", hidden1.size())
          #print("Hidden 2 computed. Size = ", hidden2.size())

          contra_loss1 = Contrastive_Loss(hidden1, hidden2, 5.0)
          contra_loss2 = Contrastive_Loss(hidden12, hidden22, 5.0)
          #print("Chunk 1 Loss = ", contra_loss.item())
          contra_loss1.backward()
          contra_loss2.backward()
          optimizer1.step()
          optimizer2.step()

          # measure accuracy and record loss
          prec2 = accuracy(output2.data, target, topk=(1,))[0]
          losses2.update(loss2.data.item(), aug_x.size(0))
          top2.update(prec2.item(), aug_x.size(0))

          batch_time.update(time.time() - end)
          end = time.time()

        if (i+1) % args.print_freq == 0:
            # print(discriminate_weights)
            fd = open(record_file1, 'a+')
            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                       epoch, i+1, train_batches_num, batch_time=batch_time,
                       loss=losses1, top1=top1))
            print(string)
            fd.write(string + '\n')
            fd.close()

            fd = open(record_file2, "a+")
            string = ('Augmented** Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
                       epoch, i+1, train_batches_num, batch_time=batch_time,
                       loss=losses2, top1=top2))
            print(string)
            fd.write(string + '\n')
            fd.close()
    return contra_loss2.item() 


def validate(val_loader, model1, model2, epoch):
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
    for i, (input, target) in enumerate(val_loader):
        #aug_input = augmented(input)
        aug_input=input
        if torch.cuda.is_available():
          target = target.cuda()
          input = input.cuda()
          aug_input.cuda()
        input_var = torch.autograd.Variable(input)
        aug_input_var = torch.autograd.Variable(aug_input)
        target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            output1, loss1, hidden1 = model1(img=input_var,
                                 target=target_var,)
            output2, loss2, hidden2 = model2(img=aug_input_var,
                                 target=target_var,)

        # measure accuracy and record loss
        prec1 = accuracy(output1.data, target, topk=(1,))[0]
        losses1.update(loss1.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        prec2 = accuracy(output2.data, target, topk=(1,))[0]
        losses2.update(loss2.data.item(), aug_input.size(0))
        top2.update(prec2.item(), aug_input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    fd = open(record_file1, 'a+')
    string = ('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
              'Loss {loss.value:.4f} ({loss.ave:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.ave:.3f})\t'.format(
        epoch, (i + 1), train_batches_num, batch_time=batch_time,
        loss=losses2, top1=top1))
    print(string)
    fd.write(string + '\n')
    fd.close()

    fd = open(record_file2, 'a+')
    string = ('Test: [{0}][{1}/{2}]\t'
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

    return top1.ave, top2.ave, hidden1, hidden2


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