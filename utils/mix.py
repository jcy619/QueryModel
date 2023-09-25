import numpy as np
import torch

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, args):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if args.alpha > 0: #1
        lam = np.random.beta(args.alpha, args.alpha) #从beta分布中生成随机数
    else:
        lam = 1

    batch_size = x.size()[0]
    
    #index = torch.randperm(batch_size).cuda()
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :] #根据lam对两个特征向量x和x[index:]进行特征加权求和得到mixed_x
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, args):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if args.beta > 0:
        lam = np.random.beta(args.beta, args.beta)
    else:
        lam = 1

    batch_size = x.size()[0]
    
    #index = torch.randperm(batch_size).cuda()
    index = torch.randperm(batch_size)

    y_a, y_b = y, y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x_sliced = x[index, :, bbx1:bbx2, bby1:bby2] #（8，3，18，18）
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return [bbx1, bby1, bbx2, bby2 ], y_a, y_b, lam, x_sliced

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)