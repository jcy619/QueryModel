# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Self-supervised training script for low resolution small-scale datasets"""


import argparse
from concurrent.futures import process
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from utils import utils_ssl as utils
from projection_head import MLPHead
from functools import partial
from models.vit import VisionTransformer
from models.swin import SwinTransformer
from models.cait import cait_models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from matcher import build_matcher
from query_head import QueryHead


torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('SSL for low resolution dataset', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit', type=str,
        choices=['vit', 'swin', 'cait'] \
                + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs, we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=4, type=int, help="""Size in pixels of input square patches - default 4 (for 4x4 patches) """)
    parser.add_argument('--out_dim', default=1024, type=int, help="""Dimensionality of the SSL MLP head output. For complex and large datasets large values (like 65k) work well.""")

    parser.add_argument('--norm_last_layer', default=False, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the MLP head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    parser.add_argument('--image_size', default=32, type=int, help=""" Size of input image. """)
    parser.add_argument('--in_channels',default=3, type=int, help=""" input image channels. """)
    parser.add_argument('--embed_dim',default=192, type=int, help=""" dimensions of vit """)
    parser.add_argument('--num_layers',default=9, type=int, help=""" No. of layers of ViT """)
    parser.add_argument('--num_heads',default=12, type=int, help=""" No. of heads in attention layer
                                                                                 in ViT """)
    parser.add_argument('--vit_mlp_ratio',default=2, type=int, help=""" MLP hidden dim """)
    parser.add_argument('--qkv_bias',default=True, type=bool, help=""" Bias in Q K and V values """)
    parser.add_argument('--drop_rate',default=0., type=float, help=""" dropout """)
    parser.add_argument('--vit_init_value',default=0.1, type=float, help=""" initialisation values of ViT """)
    parser.add_argument('--use_ape',default=False, type=bool, help=""" Absolute position embeddings """)
    parser.add_argument('--use_rpb',default=False, type=bool, help=""" Relative position embeddings """)
    parser.add_argument('--use_shared_rpb',default=False, type=bool, help=""" Shared Relative position embeddings """)
    parser.add_argument('--use_mean_pooling',default=False, type=bool, help=""" Shared Relative position embeddings """)

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.07, type=float, help="""Final value (after linear warmup)#用来控制教师模型的输出分布的平滑程度，
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend#较高的教师温度可以使学生模型更容易学习，
        starting with the default value of 0.04 and increase this slightly if needed.""")#但也可能会降低模型的泛化能力。
    parser.add_argument('--warmup_teacher_temp_epochs', default=10, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 10).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=False, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter #控制梯度裁剪的阈值，表示梯度的最大范数。
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can  #在使用梯度剪裁时，如果梯度的范数超过了该阈值
        help optimization for larger ViT architectures. 0 for disabling.""") #则会对梯度进行裁剪，将其限制在该阈值范围内。
    parser.add_argument('--batch_size_per_gpu', default=8, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs  #冻结最后一层的权重，允许前面的层进行参数更新
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0001, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""") #结束时的学习率
    parser.add_argument("--warmup_epochs", default=30, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.5, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. """) #剪裁的范围从0.5~1
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. """) #local view
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.2, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--dataset', default='CIFAR100', type=str,
        choices=['Tiny-Imagenet', 'CIFAR10', 'CIFAR100','CINIC','SVHN'],
        help='Please specify path to the training data.')
    parser.add_argument('--datapath', default='./data', type=str,
        help='Please specify path to the training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--mlp_head_in", default=192, type=int, help="input dimension going inside MLP projection head")
    return parser


def train(args):
    #utils.init_distributed_mode(args)#初始化分布式训练环境
    utils.fix_random_seeds(args.seed) #设置随机种子，每次运行代码时生成的随机数是可重现的
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))) #把args打印出来
    #cudnn.benchmark = True #优化使用GPU的运行性能
    cudnn.benchmark = False

    # ============ preparing data ... ============
    # ============ preparing data ... ============
    transform = DataAugmentation(
        args
    )  # 把一张图像处理成10张，数据增强，归一化。。。
    if args.dataset == 'Tiny-Imagenet':
        dataset = datasets.ImageFolder(
            root=args.datapath, transform=transform)
    elif args.dataset == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(root=args.datapath, train=True,
                                               download=True, transform=transform)
    elif args.dataset == "CIFAR100":
        dataset = torchvision.datasets.CIFAR100(root=args.datapath, train=True,
                                                download=True, transform=transform)  # transformer已经在这里应用
    elif args.dataset == "CINIC":
        dataset = datasets.ImageFolder(root=args.datapath, transform=transform)

    elif args.dataset == "SVHN":
        dataset = torchvision.datasets.SVHN(root=args.datapath, split='train',
                                            download=True, transform=transform)

    # sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        # sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============

    # Custom models
    if args.arch == 'vit':

        VitModel = VisionTransformer(img_size=[args.image_size],
            patch_size=args.patch_size,
            in_chans=args.in_channels,
            num_classes=0,#没有分类头
            embed_dim=192,
            depth=9,
            num_heads=12,
            mlp_ratio=2,
            qkv_bias=args.qkv_bias,#True
            drop_rate=args.drop_rate,#0.0
            drop_path_rate=args.drop_path_rate, #0.1
            norm_layer=partial(nn.LayerNorm, eps=1e-6))


    elif args.arch =='swin':

        mlp_ratio = args.vit_mlp_ratio
        window_size = 4
        patch_size = 2 if args.image_size == 32 else 4

        student = SwinTransformer(img_size=args.image_size,num_classes=0,
        window_size=window_size, patch_size=patch_size, embed_dim=96, depths=[2, 6, 4], num_heads=[3, 6, 12],
        mlp_ratio=mlp_ratio, qkv_bias=True, drop_path_rate=args.drop_path_rate)

        teacher = SwinTransformer(img_size=args.image_size,num_classes=0,
        window_size=window_size, patch_size=patch_size, embed_dim=96, depths=[2, 6, 4], num_heads=[3, 6, 12],
        mlp_ratio=mlp_ratio, qkv_bias=True, drop_path_rate=args.drop_path_rate)


    elif args.arch == 'cait':
        patch_size =  4 if args.image_size == 32 else 8
        student = cait_models(
        img_size= args.image_size,patch_size=patch_size, embed_dim=192, depth=24, num_heads=4, mlp_ratio=2, qkv_bias=True,num_classes=0,
                drop_path_rate=args.drop_path_rate,norm_layer=partial(nn.LayerNorm, eps=1e-6),init_scale=1e-5,depth_token_only=2)
        teacher = cait_models(
        img_size= args.image_size,patch_size=patch_size, embed_dim=192, depth=24, num_heads=4, mlp_ratio=2, qkv_bias=True,num_classes=0,
                drop_path_rate=args.drop_path_rate,norm_layer=partial(nn.LayerNorm, eps=1e-6),init_scale=1e-5,depth_token_only=2)

    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions



    num_classes = 100
    num_queries = 100
    dec_layers = 6
    num_patches = int((args.image_size/args.patch_size)**2+1)
    model = utils.MultiCropWrapper(VitModel,
                                   QueryHead(num_classes = num_classes,
                                             num_queries = num_queries,
                                             embed_dim = args.embed_dim,
                                             num_patches = num_patches,
                                             dec_layers = dec_layers,
                                             num_heads = args.num_heads,
                                             mlp_ratio = args.vit_mlp_ratio,))
                                            

    # move networks to gpu
    #student, teacher = student.cuda(), teacher.cuda()
   
    # synchronize batch norms (if any)
    if utils.has_batchnorms(model):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #     teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
    #
    #     # we need DDP wrapper to have synchro batch norms working...
    #     teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
    #     teacher_without_ddp = teacher.module
    # else:
    #     # teacher_without_ddp and teacher are the same thing
    #     teacher_without_ddp = teacher
    #student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    model = nn.DataParallel(model)


    # teacher and student start with the same weights
    # filtered_student_state_dict = {k: v for k, v in student.module.state_dict().items() if k in teacher_without_ddp.state_dict().keys()}
    # teacher_without_ddp.load_state_dict(filtered_student_state_dict)
    #
    # # there is no backpropagation through the teacher, so no need for gradients
    # for p in teacher.parameters():
    #     p.requires_grad = False

    #print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    # view_pred_loss = ViewPredLoss(
    #     args.out_dim,
    #     args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
    #     args.warmup_teacher_temp,
    #     args.teacher_temp,
    #     args.warmup_teacher_temp_epochs,
    #     args.epochs,
    # ) #.cuda()
    num_classes = 100
    matcher = build_matcher()
    eos_coef=0.01
    criterion = SetCriterion(num_classes,matcher = matcher,eos_coef = eos_coef)

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(model)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256. ,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )

    wd_schedule = utils.cosine_scheduler(
        args.weight_decay, #0.04
        args.weight_decay_end, #0.4
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    # momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, #0.996~1
    #                                            args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")
    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        model = model,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        view_pred_loss=criterion,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting SSL training !")
    for epoch in range(start_epoch, args.epochs):

        
        #data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch ... ============
        train_stats = train_one_epoch(model, criterion,data_loader, optimizer, lr_schedule, wd_schedule,
            epoch,fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            #'student': student.state_dict(), #所有的参数
            #'teacher': teacher.state_dict(),
            'model':model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'view_pred_loss': criterion.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model, criterion, data_loader,
                    optimizer, lr_schedule, wd_schedule,epoch,fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images,target) in enumerate(metric_logger.log_every(data_loader, 10, header)):  #_target  #add mask here
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration全局迭代参数
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        #images = [im.cuda(non_blocking=True) for im in images]

        # teacher and student forward passes + compute loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            #teacher_output  = teacher(images[:2])  #（16,1024） # only the 2 global views pass through the teacher
            output  = model(images) #(bs,nq,nc+1)
            total_loss = criterion(output,target)
            loss = total_loss['loss_ce']
            #loss_view = total_loss.pop('ce_loss')
            loss_view = loss


        if not math.isfinite(loss.item()):
            print("Loss is {}, View Pred loss is {}, stopping training".format(loss.item(),loss_view.item()
                                                                                           ), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(model, args.clip_grad) #对student进行梯度剪裁
            utils.cancel_gradients_last_layer(epoch, model,
                                              args.freeze_last_layer) #冻结最后一层的权重，允许前面的层进行参数更新
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(model, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, model,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()


        # logging
        #torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(view_pred_loss=loss_view.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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
        temp = outputs[idx]
        return losses


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

        # first global crop

        # self.augmentation = []
        # self.augmentation += [torchvision.transforms.RansomHorizontalFlip(),
        #                       torchvision.transforms.RandomCrop(args.image_size,padding=4)]

        from utils.autoaug import CIFAR10Policy
        from utils.random_erasing import RandomErasing
        # self.augmentation += [CIFAR10Policy()]

        self.global_transfo1 = transforms.Compose([ #对输入图像进行随机剪裁，剪裁后的大小为size，剪裁是的缩放范围，采用的插值算法为双三次插值。
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(args.image_size, padding=4),
            flip_and_color_jitter,#数据增强
            utils.GaussianBlur(1.0), #对图像进行高斯模糊
            CIFAR10Policy(),
            normalize,
            RandomErasing(probability=0.25, sh=0.4, r1=0.3, mean=(0.5070, 0.4865, 0.4409))
        ])

    def __call__(self, image): #一张图片输入，处理成10张图片，2张global view,8张local view

        return self.global_transfo1(image)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('SSL for low resolution dataset', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)