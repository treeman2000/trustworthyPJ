import os
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from torchnet.meter import AverageValueMeter

from torch.utils.tensorboard import SummaryWriter

from models.wideresnet import *
from models.resnet import *
from trades import trades_loss,sat_loss

from dataset import cifar10_loader,To2val,To3chan2val

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=76, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', type=float, default=2e-4,
                     metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon',type=float, default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps',type=int, default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size',type=float, default=0.007,
                    help='perturb step size')
parser.add_argument('--beta',type=float, default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=5, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--start_from', default=None, type=str,
                    help='start from a checkpoint')
parser.add_argument('--start_epoch', default=1, type=int,
                    help='start from a epoch')
parser.add_argument('--title', default=None, type=str,
                    help='title for logs')
parser.add_argument('--submitit', action='store_true', default=False,
                    help='use submitit')
parser.add_argument('--loss', default=None, type=str,
                    help='loss function: sat, st, trades')
parser.add_argument('--model', default=None, type=str,
                    help='model: wr(wideresnet),new')
parser.add_argument('--chan3val2',action='store_true')

# dp
parser.add_argument('--dp', action='store_true', default=False)

args = parser.parse_args()
assert args.loss is not None

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
torch.manual_seed(args.seed)

if args.chan3val2:
    train_loader,val_loader,test_loader=cifar10_loader(args.batch_size,args.test_batch_size,extra_trans=To3chan2val())
else:
    train_loader,val_loader,test_loader=cifar10_loader(args.batch_size,args.test_batch_size)



def train_one_epoch(args, model, device, train_loader, optimizer, epoch):
    model.train()
    avg_meter=AverageValueMeter()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        if args.loss == 'trades':
            loss = trades_loss(model=model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            step_size=args.step_size,
                            epsilon=args.epsilon,
                            perturb_steps=args.num_steps,
                            beta=args.beta,
                            avg_meter=avg_meter)
        elif args.loss == 'sat':
            loss = sat_loss(model=model,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            step_size=args.step_size,
                            epsilon=args.epsilon,
                            perturb_steps=args.num_steps,
                            avg_meter=avg_meter)
        elif args.loss == 'st':
            loss=F.cross_entropy(model(data),target)
        else:
            raise Exception('false loss')

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    print('purturb suceess rate of generated adv example:',avg_meter.value()[0])


def eval_train(model, device, train_loader,w,epoch):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    w.add_scalar('training_average_loss',train_loss,epoch)
    training_accuracy = correct / len(train_loader.dataset)
    w.add_scalar('training_acc',training_accuracy,epoch)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader,w,epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    w.add_scalar('test_average_loss',test_loss,epoch)
    w.add_scalar('test_acc',test_accuracy,epoch)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    print(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # init model, ResNet18() can be also used here for training
    model = WideResNet().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    if args.start_from is not None:
        model.load_state_dict(torch.load(args.start_from))
    
    with SummaryWriter(f'log_{args.title}') as w:
        for epoch in range(args.start_epoch, args.epochs + 1):
            # adjust learning rate for SGD
            adjust_learning_rate(optimizer, epoch)

            # adversarial training
            train_one_epoch(args, model, device, train_loader, optimizer, epoch)

            # evaluation on natural examples
            print('================================================================')
            eval_train(model, device, val_loader,w,epoch)
            eval_test(model, device, test_loader,w,epoch)
            print('current time: ',datetime.now())
            print('================================================================')

            # save checkpoint
            if epoch % args.save_freq == 0 or epoch==args.epochs:
                torch.save(model.state_dict(),
                        os.path.join(model_dir, f'{args.title}_epoch{epoch}.pt'))
                # torch.save(optimizer.state_dict(),
                #            os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))


log_folder = f'./logs_of_submitit/{args.title}_%j'
def main_submitit():
    import submitit
    executor = submitit.AutoExecutor(folder=log_folder)

    if args.dp:
        print('dp training')
        num_gpu=2
        ntasks_per_node=1
        qos='high'
    else:
        print('single gpu')
        num_gpu=1
        ntasks_per_node=1
        qos='low'

    # 必须填timeout_min，最多2天
    executor.update_parameters(
        # gpus_per_node=num_gpus,
        slurm_gres=f'gpu:3090:{num_gpu}',
        cpus_per_task=2,
        nodes=1,
        timeout_min=2*60*24, 
        slurm_ntasks_per_node=ntasks_per_node,
        slurm_partition="fvl", 
        slurm_qos=qos,
        # slurm_partition="scavenger", 
        # slurm_qos='scavenger',
        slurm_mem=ntasks_per_node*64*1024,
        )
    # The submission interface is identical to concurrent.futures.Executor
    job = executor.submit(main)
    print(job.job_id)  # ID of your job

    # job.result()  # waits for the submitted function to complete and returns its output
    # print('ok')

if __name__ == '__main__':
    if args.submitit:
        main_submitit()
    else:
        main()