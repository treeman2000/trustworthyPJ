# train a 2val model
python train_cifar10.py --loss=st --chan3val2 --epsilon=0.2 --step-size=0.02

loss 可以是st，sat，trades。分别是standard training, standard adversarial training, trades

默认是在前台运行。可以加--submitit作为批任务提交。

# test with pgd attack
python pgd_attack_cifar10.py --chan3val2 --ckpt=model-cifar-wideResNet\sat_3chan2val_epoch76.pt --epsilon=0.2 --step-size=0.02

# result
3channel, 2val per channel
| 训练方法 | clean acc | pgd acc |
| -- | -- | -- |
| st | 81 | 0.1 |
| sat | 64.7 | 64.7 |

# todo
目前做的是先把数据集转换成01二值的数据集，然后再训练、对抗。打算试试对原数据集先加对抗扰动，再转成01。