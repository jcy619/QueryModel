from utils.mix import cutmix_data, mixup_data, mixup_criterion
import numpy as np
import random
import logging as log
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from colorama import Fore, Style
from torchsummary import summary
from utils.losses import LabelSmoothingCrossEntropy
import os
from utils.sampler import RASampler
from utils.logger_dict import Logger_dict
from utils.print_progress import progress_bar
from utils.training_functions import accuracy
import argparse
from utils.scheduler import build_scheduler
from utils.dataloader import datainfo, dataload
from models.build_model import create_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
from query_head import QueryHead
from utils import utils_ssl as utils
import torch.nn.functional as F
import torchvision

warnings.filterwarnings("ignore", category=Warning)

best_acc1 = 0
MODELS = ['vit', 'swin' , 'cait']


def init_parser():
    parser = argparse.ArgumentParser(description='Vit small datasets quick training script')

    # Data args
    parser.add_argument('--datapath', default='./data', type=str, help='dataset path')
    
    parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR10', 'CIFAR100', 'Tiny-Imagenet', 'SVHN','CINIC'], type=str, help='small dataset path')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=1, type=int, metavar='N', help='log frequency (by iteration)')

    # Optimization hyperparams
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    
    parser.add_argument('--warmup', default=10, type=int, metavar='N', help='number of warmup epochs')
    
    parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N', help='mini-batch size (default: 128)', dest='batch_size')
    
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    
    parser.add_argument('--weight-decay', default=5e-2, type=float, help='weight decay (default: 1e-4)')

    parser.add_argument('--arch', type=str, default='cait', choices=MODELS)

    parser.add_argument('--disable-cos', action='store_true', help='disable cosine lr schedule') #禁用余弦学习率

    parser.add_argument('--enable_aug', action='store_true', help='disable augmentation policies for training')#禁用数据增强

    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--no_cuda', action='store_true', help='disable cuda')

    parser.add_argument('--ls', action='store_false', help='label smoothing')

    parser.add_argument('--channel', type=int, help='disable cuda')

    parser.add_argument('--heads', type=int, help='disable cuda')

    parser.add_argument('--depth', type=int, help='disable cuda')

    parser.add_argument('--tag', type=str, help='tag', default='')

    parser.add_argument('--seed', type=int, default=0, help='seed')
    
    parser.add_argument('--sd', default=0.1, type=float, help='rate of stochastic depth')
    
    parser.add_argument('--resume', default=False, help='Version')
       
    parser.add_argument('--aa', action='store_false', help='Auto augmentation used'),
    
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    
    parser.add_argument('--cm',action='store_false' , help='Use Cutmix')
    
    parser.add_argument('--beta', default=1.0, type=float,
                        help='hyperparameter beta (default: 1)')
    
    parser.add_argument('--mu',action='store_false' , help='Use Mixup')
    
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='mixup interpolation coefficient (default: 1)')
    
    parser.add_argument('--mix_prob', default=0.5, type=float,
                        help='mixup probability')
    
    parser.add_argument('--ra', type=int, default=3, help='repeated augmentation')
    
    parser.add_argument('--re', default=0.25, type=float, help='Random Erasing probability')
    
    parser.add_argument('--re_sh', default=0.4, type=float, help='max erasing area')
    
    parser.add_argument('--re_r1', default=0.3, type=float, help='aspect of erasing area')
    
    parser.add_argument('--is_LSA', action='store_true', help='Locality Self-Attention')
    
    parser.add_argument('--is_SPT', action='store_true', help='Shifted Patch Tokenization')

    parser.add_argument('--pretrained_weights', default='./checkpoint-3.pth', type=str, help="Path to pretrained weights to evaluate.")

    parser.add_argument("--checkpoint_key", default="model", type=str, help='Key to use in the checkpoint (example: "teacher")')

    parser.add_argument('--patch_size', default=4, type=int, help='patch size for ViT')

    parser.add_argument('--vit_mlp_ratio', default=2, type=int, help='MLP layers in the transformer encoder')

    return parser


def main(args):
    global best_acc1    
    
    #torch.cuda.set_device(args.gpu)

    data_info = datainfo(logger, args) #把图片转换成tensor
    transform = DataAugmentation(
        args
    )


    num_queries = 100
    dec_layers = 1
    num_patches = int((data_info['img_size'] /args.patch_size) ** 2 + 1)

    num_heads = 12
    embed_dim = 192
    CaiTModel = create_model(data_info['img_size'], 0, args)
    model = utils.MultiCropWrapper(CaiTModel,
                                   QueryHead(num_classes=data_info['n_classes'],
                                             num_queries=num_queries,
                                             embed_dim=embed_dim,
                                             num_patches=num_patches,
                                             dec_layers=dec_layers,
                                             num_heads=num_heads,
                                             mlp_ratio=args.vit_mlp_ratio, ))
   
    #model.cuda(args.gpu)
        
    print(Fore.GREEN+'*'*80)
    logger.debug(f"Creating model: {model_name}")    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(f'Number of params: {format(n_parameters, ",")}')
    logger.debug(f'Initial learning rate: {args.lr:.6f}')
    logger.debug(f"Start training for {args.epochs} epochs")
    print('*'*80+Style.RESET_ALL)
    
    if os.path.isfile(args.pretrained_weights):
        model_dict = model.state_dict()
        print("loading pretrained weights . . .")
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        state_dict={k:v if v.size()==model_dict[k].size()  else  model_dict[k] for k,v in zip(model_dict.keys(), state_dict.values())}
        model.load_state_dict(state_dict, strict=False)
        #print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))



    '''
        Data Augmentation
    '''

    normalize = [transforms.Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])]
    train_dataset = torchvision.datasets.CIFAR100(
        root=args.datapath, train=True, download=False, transform=transform)
    val_dataset = torchvision.datasets.CIFAR100(
        root=args.datapath, train=False, download=False, transform=transforms.Compose([
            transforms.Resize(data_info['img_size']),
            transforms.ToTensor(),
            *normalize]))


    train_loader = torch.utils.data.DataLoader(
        train_dataset,  num_workers=args.workers, pin_memory=True,
        batch_sampler=RASampler(len(train_dataset), args.batch_size, 1, args.ra, shuffle=True, drop_last=True))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    '''
        Training
    '''
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args, optimizer, len(train_loader))
    
    #summary(model, (3, data_info['img_size'], data_info['img_size']))
    
    print()
    print("Beginning training")
    print()
    
    lr = optimizer.param_groups[0]["lr"]
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        final_epoch = args.epochs
        args.epochs = final_epoch - (checkpoint['epoch'] + 1)

    from matcher import build_matcher

    num_classes = 100
    matcher = build_matcher()
    eos_coef = 0.01
    criterion = SetCriterion(num_classes, matcher=matcher, eos_coef=eos_coef)

    for epoch in tqdm(range(args.epochs)):
        lr = train(train_loader, model, criterion, optimizer, epoch, scheduler, args)
        acc1 = validate(val_loader, model, criterion, lr, args, epoch=epoch)
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 
            }, 
            os.path.join(save_path, 'checkpoint.pth'))
        
        logger_dict.print()
        
        if acc1 > best_acc1:
            print('* Best model upate *')
            best_acc1 = acc1
            
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, os.path.join(save_path, 'best.pth'))         
        
        print(f'Best acc1 {best_acc1:.2f}')
        print('*'*80)
        print(Style.RESET_ALL)        
        
        writer.add_scalar("Learning Rate", lr, epoch)
        
        
    print(Fore.RED+'*'*80)
    logger.debug(f'best top-1: {best_acc1:.2f}, final top-1: {acc1:.2f}')
    print('*'*80+Style.RESET_ALL)
    torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint.pth'))


def train(train_loader, model, criterion, optimizer, epoch, scheduler,  args):
    model.train()
    criterion.train()
    loss_val, acc1_val = 0, 0
    n = 0
        
    
    for i, (images, target) in enumerate(train_loader):
        if (not args.no_cuda) and torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)


        output_pre = model(images) #(bs,nq,nc+1)
        loss,output = criterion(output_pre,target) #output(bs,nc+1)
        acc = accuracy(output, target, (1,))
        acc1 = acc[0]
        n += images.size(0)
        loss_val += float(loss.item() * images.size(0))
        acc1_val += float(acc1[0] * images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if args.print_freq >= 0 and i % args.print_freq == 0:
            avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
            progress_bar(i, len(train_loader),f'[Epoch {epoch+1}/{args.epochs}][T][{i}]   Loss: {avg_loss:.4e}   Top-1: {avg_acc1:6.2f}   LR: {lr:.7f}'+' '*10)

    logger_dict.update(keys[0], avg_loss)
    logger_dict.update(keys[1], avg_acc1)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Acc/train", avg_acc1, epoch)
    
    return lr


def validate(val_loader, model, criterion, lr, args, epoch=None):
    model.eval()
    criterion.eval()
    loss_val, acc1_val = 0, 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if (not args.no_cuda) and torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            
            output_pre = model(images)
            loss,output = criterion(output_pre, target) #label smoothing cross entrophy
            acc = accuracy(output, target, (1, 5))
            acc1 = acc[0]
            n += images.size(0)
            loss_val += float(loss.item() * images.size(0))
            acc1_val += float(acc1[0] * images.size(0))

            if args.print_freq >= 0 and i % args.print_freq == 0:
                avg_loss, avg_acc1 = (loss_val / n), (acc1_val / n)
                progress_bar(i, len(val_loader), f'[Epoch {epoch+1}][V][{i}]   Loss: {avg_loss:.4e}   Top-1: {avg_acc1:6.2f}   LR: {lr:.6f}')
    print()        

    print(Fore.BLUE)
    print('*'*80)
    
    logger_dict.update(keys[2], avg_loss)
    logger_dict.update(keys[3], avg_acc1)
    
    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Acc/val", avg_acc1, epoch)

    
    return avg_acc1

class SetCriterion(nn.Module):
    def __init__(self,num_classes,matcher,eos_coef):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef

        empty_weight = torch.ones(self.num_classes+1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight',empty_weight)
        self.losses = ['labels']

    def _get_src_permutation_idx(self, indices):

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self,outputs,targets,indices):

        src_logits = outputs  # (bs,nq,nc+1)
        idx = self._get_src_permutation_idx(indices)
        targets_classes_o = targets
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = targets_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight=self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses


    def get_loss(self,loss,outputs,targets,indices):
        loss_map = {
            'labels':self.loss_labels,
        }
        return loss_map[loss](outputs,targets,indices)

    def forward(self,outputs,targets):

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)  # 哪个query学到了label，学到了几个，0个表示1个
        idx = self._get_src_permutation_idx(indices)
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))
        return losses['loss_ce'],outputs[idx]#(bs,cl+1)

class DataAugmentation(object):
    def __init__(self, args):
        if args.dataset=="CIFAR10":
            mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        elif args.dataset == "CIFAR100":
            mean, std = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        elif args.dataset == "SVHN":
            mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
        elif args.dataset == "CINIC":
            mean, std = (0.47889522, 0.47227842, 0.43047404),(0.24205776, 0.23828046, 0.25874835)
        elif args.dataset == "Tiny-Imagenet":
            mean, std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        #数据增强操作序列

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5), #50%对输入图像进行水平反转
            transforms.RandomApply( #随机应用操作
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8  #80%的概率对输入图像进行颜色增强，亮度±0.4，对比度，饱和度，色调
            ),
            transforms.RandomGrayscale(p=0.2), #20%的概率将输入图像转换为灰度图像
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
	    transforms.Normalize(mean, std),
        ])


        from utils.autoaug import CIFAR10Policy
        from utils.random_erasing import RandomErasing

        image_size=32
        self.global_transfo1 = transforms.Compose([ #对输入图像进行随机剪裁，剪裁后的大小为size，剪裁是的缩放范围，采用的插值算法为双三次插值。
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(image_size, padding=4),
            flip_and_color_jitter,#数据增强
            utils.GaussianBlur(1.0), #对图像进行高斯模糊
            CIFAR10Policy(),
            normalize,
            RandomErasing(probability=0.25, sh=0.4, r1=0.3, mean=(0.5070, 0.4865, 0.4409))
        ])

    def __call__(self, image): #一张图片输入，处理成10张图片，2张global view,8张local view

        return self.global_transfo1(image)


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    global save_path
    global writer
    
    # random seed

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    model_name = args.arch

    if not args.is_SPT:
        model_name += "-Base"
    else:
        print("spt present")
        model_name += "-SPT"
 
    if args.is_LSA:
        print("lsa present")
        model_name += "-LSA"
        
    model_name += f"-{args.tag}-{args.dataset}-LR[{args.lr}]-Seed{args.seed}"
    save_path = os.path.join(os.getcwd(), 'save_finetuned', model_name)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
    writer = SummaryWriter(os.path.join(os.getcwd(), 'tensorboard', model_name))
    
    # logger

    log_dir = os.path.join(save_path, 'history.csv')
    logger = log.getLogger(__name__)
    formatter = log.Formatter('%(message)s')
    streamHandler = log.StreamHandler()
    fileHandler = log.FileHandler(log_dir, 'a')
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(level=log.DEBUG)

    
    global logger_dict
    global keys
    
    logger_dict = Logger_dict(logger, save_path)
    keys = ['T Loss', 'T Top-1', 'V Loss', 'V Top-1']
    
    main(args)
