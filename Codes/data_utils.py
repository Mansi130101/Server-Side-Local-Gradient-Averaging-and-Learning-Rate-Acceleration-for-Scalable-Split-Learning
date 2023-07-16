import numpy as np
import random
import torch

def sort_indices(data, nclass = 2, size="l"):
  full_labels = np.array(data.targets)
  labels = [i for i in range(10)]
  indices = []
  for label in labels:
    indice = (full_labels == label).nonzero()[0]
    indices.append(indice)

  total_indices=np.array([], dtype=np.int16)
  for i in range(nclass):
    #print(indices[i])
    total_indices = np.append(total_indices, indices[i]) 
  unused_indices = np.array([])
  used_indices = total_indices
  if size == "xxxs":                                       # 100 total samples
    used_indices = total_indices[4950:5050]
    unused_indices = np.append(total_indices[:4950], total_indices[5050:])
  elif size =="xxs":                                       # 128 total samples
    used_indices = total_indices[4936:5064]
    unused_indices = np.append(total_indices[:4936], total_indices[5064:])
  elif size == "xs":                                       # 300 total samples
    used_indices = total_indices[4850:5150]
    unused_indices = np.append(total_indices[:4850], total_indices[5150:])
  elif size == "s":                                        # 500 total samples
    used_indices = total_indices[4750:5250]
    unused_indices = np.append(total_indices[:4750], total_indices[5250:])
  elif size == "m":
    used_indices = total_indices[4500:5500]                # 1000 total samples
    unused_indices = np.append(total_indices[:4500], total_indices[5500:])
  elif size == "l":
    used_indices = total_indices[4000:6000]                # 2000 total samples
    unused_indices = np.append(total_indices[:4000], total_indices[6000:])
  
  print(f"No of samples used {used_indices.shape} and unused {unused_indices.shape}")
  return used_indices, unused_indices


def sort_multi_indices(data, size=100, nclass = 10, split="train",dataset='cifar10'):
  label_size=1000
  if split == "train":
    assert size<=4000, "10000 samples for Validation set necessary"
    if dataset=='cifar10':
      label_size = 5000
    elif dataset=='fmnist':
      label_size = 6000

  full_labels = np.array(data.targets)
  labels = [i for i in range(nclass)]
  indices = []
  for label in labels:
    indice = (full_labels == label).nonzero()[0]
    indices.append(indice)
  
  total_indices=np.array([], dtype=np.int16)
  for i in range(nclass):
    #print(indices[i])
    total_indices = np.append(total_indices, indices[i]) 
  
  unused_indices = np.array([], dtype=np.int16)
  used_indices = np.array([], dtype=np.int16)
  val_indices = np.array([], dtype=np.int16)
  
  for i in range(nclass):
    used_indices = np.append(total_indices[i*label_size:i*label_size+size], used_indices)
    unused_indices = np.append(total_indices[i*label_size+size:(i+1)*label_size], unused_indices)

  print(f"No of samples used {used_indices.shape} and unused {unused_indices.shape}")
  if len(unused_indices)!=0:
    for i in range(nclass):
      val_indices = np.append(val_indices, unused_indices[i*(label_size-size):i*(label_size-size)+1000])
    unused_indices = val_indices
  
  return used_indices, unused_indices


def divide_indices(indices, size, no_of_classes=10, workers=2,skewed=0): #added
  train_indices = [[] for i in range(workers)]
  label_per_worker = int(size/workers)
  #added
  distribution=[[] for i in range(workers)]

  for i in range(workers):
    labelnum=label_per_worker*no_of_classes
    for j in range(no_of_classes):
      if skewed==0:
        distribution[i].append(label_per_worker)
      else:
        if j==no_of_classes-1:
          distribution[i].append(labelnum)
        else:
          l=int((labelnum/(no_of_classes-i))*0.98 )
          u=int((labelnum/(no_of_classes-i))*2 )
          a=int(random.randint(l,u))
          distribution[i].append(a)
          labelnum-=a
          #print('l',l,'u',u,'a',a,'j',j)
          #print('labelnum',labelnum)
    print('worker',i)
    print(distribution[i])
  #added

  for i in range(no_of_classes):   #shuffle within a label
    random.shuffle(indices[i*size:(i+1)*size])

  for i in range(no_of_classes):
    for j in range(workers):
      labelssize=distribution[j][i]
      #print(size)
      #print("Worker train indices \n", train_indices[j])
      #print("Indices slice \n", indices[i*size + j*label_per_worker : i*size + (j+1)*label_per_worker])
      #train_indices[j].extend(indices[i*size*no_of_classes + j*label_per_worker : i*size*no_of_classes + (j+1)*label_per_worker])
      train_indices[j].extend(indices[i*size*no_of_classes + j*labelssize : i*size*no_of_classes + (j+1)*labelssize])
  
  #for j in range(workers):  
  #  print('num of samples in w',j,len(train_indices[j]))
  return train_indices


def assert_distribution(train_data, train_indices, val_indices, nclass=10):
  if train_indices is not None:
    c=0
    for w in range(len(train_indices)):
      print(f"Worker {w+1} Set:-")
      for i in range(nclass):
        c=0
        for ix in train_indices[w]:
          if train_data.targets[ix] == i:
            c += 1 
        print(f"Label {i+1} samples = {c}")
  else:
    c=0
    print(f"Val/Test Set:-")
    for i in range(nclass):
      c=0
      for ix in val_indices:
        if train_data.targets[ix] == i:
          c += 1 
      print(f"Label {i+1} samples = {c}")
  return


def my_func1():
  return 0.4
def my_func2():
  return 0.7

def create_dataloader(train_data, test_data, train_indices, val_indices, test_indices, batch_size):
  for i in range(len(train_indices)):
    random.shuffle(train_indices[i])

  train_dataloaders=[]
  for i in range(len(train_indices)):
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = False, sampler = train_indices[i])
    train_dataloaders.append(train_dataloader)
  
  val_dataloader = torch.utils.data.DataLoader(train_data, batch_size = 1024, shuffle = False, sampler = val_indices)
  test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = 1024, shuffle = False, sampler = test_indices)
  return train_dataloaders, val_dataloader, test_dataloader
