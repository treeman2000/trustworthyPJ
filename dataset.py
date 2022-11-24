import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler

class To2val():
    def __init__(self,num_outchannel=1):
        self.grayscale=transforms.Grayscale(num_output_channels=num_outchannel)

    def __call__(self,x):
        return torch.round(self.grayscale(x))

class To3chan2val():
    def __init__(self):
        return

    def __call__(self,x):
        return torch.round(x)

def cifar10_loader(batch_size,test_batch_size,root='../data',extra_trans=None,ddp=False,pin_mem=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if extra_trans is not None:
        transform_train = transforms.Compose([
            transform_train,
            extra_trans,
        ])
        transform_test = transforms.Compose([
            transform_test,
            extra_trans,
        ])
        

    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    if ddp:
        train_sampler=DistributedSampler(train_set,shuffle=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=1,pin_memory=pin_mem,sampler=train_sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1,pin_memory=pin_mem)

    # validate set 就是trainset
    val_loader = DataLoader(train_set, batch_size=test_batch_size, shuffle=False, num_workers=1,pin_memory=pin_mem)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=1,pin_memory=pin_mem)

    return train_loader,val_loader,test_loader

def cifar10_testset(root='../data',extra_trans=None):
    """for auto attack"""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if extra_trans is not None:
        transform_test = transforms.Compose([
            transform_test,
            extra_trans,
        ])
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    return testset
