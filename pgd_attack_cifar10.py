from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *
from torch.utils.tensorboard import SummaryWriter
from torchnet.meter import AverageValueMeter

parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon',type=float, default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps',type=int, default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', type=float, default=0.003,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--ckpt',
                    default='./model-cifar-wideResNet/trades_pretrained.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')

parser.add_argument('--val2', action='store_true',
                    help='input as 2val')
parser.add_argument('--val2v2', action='store_true',
                    help='use BPDA to attack')
parser.add_argument('--chan3val2', action='store_true',
                    help='input as 2val')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
from dataset import To2val,To3chan2val
transform_test = transforms.Compose([transforms.ToTensor()])
if args.val2:
    transform_test=transforms.Compose([transform_test,To2val()])
elif args.chan3val2:
    transform_test=transforms.Compose([transform_test,To3chan2val()])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

X_pgd_total=None
# avg_meter=[AverageValueMeter(),AverageValueMeter(),AverageValueMeter()]
def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    
    global X_pgd_total

    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    # 统计扰动在各个channel的分布是否不同
    delta=(X_pgd-X)
    # torch.set_printoptions(profile="full")
    # print(delta)
    # torch.set_printoptions(profile="default")
    X_pgd_statistic=delta.abs().sum(dim=0).detach().cpu().numpy()
    X_pgd_statistic=X_pgd_statistic.reshape((X_pgd_statistic.shape[0],-1))
    # shape[3,H*W]
    if X_pgd_total is None:
        X_pgd_total=X_pgd_statistic
    else:
        X_pgd_total+=X_pgd_statistic

    out_adv=model(X_pgd)
    err_pgd = (out_adv.data.max(1)[1] != y.data).float().sum()
    print('err clean: ', err)
    print('err pgd (white-box): ', err_pgd)
    # diff是对抗样本的预测和原来样本预测的不同。衡量stability
    diff = (out_adv.data.max(1)[1] != out.data.max(1)[1]).float().sum()
    print('sucess purturbed:',diff)
    # 原本就错err个，扰动了diff个，那么最多错err+diff个
    assert err+diff>=err_pgd
    return err, err_pgd, diff

to2val=To2val()
def _pgd_whitebox_BPDA(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    
    global X_pgd_total

    out = model(to2val(X))
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        X_pgd_def=Variable(to2val(X_pgd).data,requires_grad=True)
        # print(X_pgd_def)
        # exit()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd_def), y)
        loss.backward()
        eta = step_size * X_pgd_def.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        delta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + delta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    # 统计扰动在各个channel的分布是否不同
    delta=(X_pgd-X)
    # torch.set_printoptions(profile="full")
    # print(delta)
    # torch.set_printoptions(profile="default")
    X_pgd_statistic=delta.abs().sum(dim=0).detach().cpu().numpy()
    X_pgd_statistic=X_pgd_statistic.reshape((X_pgd_statistic.shape[0],-1))
    # shape[3,H*W]
    if X_pgd_total is None:
        X_pgd_total=X_pgd_statistic
    else:
        X_pgd_total+=X_pgd_statistic

    out_adv=model(to2val(X_pgd))
    err_pgd = (out_adv.data.max(1)[1] != y.data).float().sum()
    print('err clean: ', err)
    print('err pgd (white-box): ', err_pgd)
    # diff是对抗样本的预测和原来样本预测的不同。衡量stability
    diff = (out_adv.data.max(1)[1] != out.data.max(1)[1]).float().sum()
    print('sucess purturbed:',diff)
    # 原本就错err个，扰动了diff个，那么最多错err+diff个
    assert err+diff>=err_pgd
    return err, err_pgd, diff


def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    err_pgd = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err pgd black-box: ', err_pgd)
    return err, err_pgd


def eval_adv_test_whitebox(model, device, test_loader):
    """
    evaluate model by white-box attack
    """

    global X_pgd_total

    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    diff_total=0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        if args.val2v2:
            err_natural, err_robust,diff = _pgd_whitebox_BPDA(model, X, y)
        else:
            err_natural, err_robust,diff = _pgd_whitebox(model, X, y)
        robust_err_total += err_robust.item()
        natural_err_total += err_natural.item()
        diff_total+=diff.item()
        # break
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)
    total=len(test_loader.dataset)
    print('diff_total: ', diff_total)
    print('purturb sucess rate: ', diff_total/total)
    print('clean acc',(total-natural_err_total)/total)
    print('pgd20 acc',(total-robust_err_total)/total)



def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)


def main():

    if args.white_box_attack:
        # white-box attack
        print('pgd white-box attack')
        if args.val2 or args.val2v2:
            model = WideResNet(num_in_channel=1).to(device)
        else:
            model = WideResNet().to(device)

        model.load_state_dict(torch.load(args.ckpt))

        eval_adv_test_whitebox(model, device, test_loader)
    else:
        # black-box attack
        print('pgd black-box attack')
        model_target = WideResNet().to(device)
        model_target.load_state_dict(torch.load(args.target_model_path))
        model_source = WideResNet().to(device)
        model_source.load_state_dict(torch.load(args.source_model_path))

        eval_adv_test_blackbox(model_target, model_source, device, test_loader)


if __name__ == '__main__':
    main()
