import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim

from utils import clip_gradient, AvgMeter
from torch.autograd import Variable
from datetime import datetime
import torch.nn.functional as F

from model.mms_net import MMSNet

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, img_paths, mask_paths, aug=True, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = cv2.resize(image, (352, 352))
            mask = cv2.resize(mask, (352, 352)) 

        image = image.astype('float32') / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:,:,np.newaxis]
        mask = mask.astype('float32') / 255
        mask = mask.transpose((2, 0, 1))

        return np.asarray(image), np.asarray(mask)
    
epsilon = 1e-7

def recall_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall

def precision_m(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clip(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision

def dice_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+epsilon))

def iou_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return recall*precision/(recall+precision-recall*precision + epsilon)


# class FocalLossV1(nn.Module):
    
#     def __init__(self,
#                 alpha=0.25,
#                 gamma=2,
#                 reduction='mean',):
#         super(FocalLossV1, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#         self.crit = nn.BCEWithLogitsLoss(reduction='none')

#     def forward(self, logits, label):
#         # compute loss
#         logits = logits.float() # use fp32 if logits is fp16
#         with torch.no_grad():
#             alpha = torch.empty_like(logits).fill_(1 - self.alpha)
#             alpha[label == 1] = self.alpha

#         probs = torch.sigmoid(logits)
#         pt = torch.where(label == 1, probs, 1 - probs)
#         ce_loss = self.crit(logits, label.float())
#         loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
#         if self.reduction == 'mean':
#             loss = loss.mean()
#         if self.reduction == 'sum':
#             loss = loss.sum()
#         return loss

# def structure_loss(pred, mask):
#     weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wfocal = FocalLossV1()(pred, mask)
#     wfocal = (wfocal*weit).sum(dim=(2,3)) / weit.sum(dim=(2, 3))

#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask)*weit).sum(dim=(2, 3))
#     union = ((pred + mask)*weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + 1)/(union - inter+1)
#     return (wfocal + wiou).mean()

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super().__init__()
#         self.smooth = smooth

#     def forward(self, logits, target):
#         # logits: N × 1 × H × W
#         # target: N × 1 × H × W (0 or 1)

#         pred = torch.sigmoid(logits)
#         target = target.float()

#         intersection = (pred * target).sum(dim=(2, 3))
#         union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

#         dice = (2. * intersection + self.smooth) / (union + self.smooth)
#         loss = (1 - dice)

#         return loss.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, target):
        # logits: N x 1 x H x W
        # target: N x 1 x H x W (0 or 1)

        pred = torch.sigmoid(logits)
        target = target.float()

        # tp = (pred * target).sum(dim=(2, 3))
        # fp = (pred * (1 - target)).sum(dim=(2, 3))
        # fn = ((1 - pred) * target).sum(dim=(2, 3))
        tp = (pred * target).sum(dim=(1, 2, 3))
        fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
        fn = ((1 - pred) * target).sum(dim=(1, 2, 3))

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        loss = 1 - tversky

        return loss.mean()

def train(train_loader, model, optimizer, epoch, lr_scheduler, args):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    dice, iou = AvgMeter(), AvgMeter()
    precision_record = AvgMeter()
    recall_record = AvgMeter()
    loss_function = TverskyLoss(alpha=0.7, beta=0.3)
    with torch.autograd.set_detect_anomaly(True):
        for i, pack in enumerate(tqdm(train_loader, total=total_step), start=1):
            if epoch <= 1:
                    optimizer.param_groups[0]["lr"] = (epoch * i) / (1.0 * total_step) * args.init_lr
            else:
                lr_scheduler.step()

            for rate in size_rates: 
                optimizer.zero_grad()
                # ---- data prepare ----
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- rescale ----
                trainsize = int(round(args.init_trainsize*rate/32)*32)
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=False)
                gts = F.interpolate(
                    gts,
                    size=(trainsize, trainsize),
                    mode='nearest'
                )
                # ---- forward ----
                map1, map2, map3 = model(images)
                map1 = F.interpolate(map1, size=(trainsize, trainsize), mode='bilinear', align_corners=False)
                map2 = F.interpolate(map2, size=(trainsize, trainsize), mode='bilinear', align_corners=False)
                map3 = F.interpolate(map3, size=(trainsize, trainsize), mode='bilinear', align_corners=False)
                loss = loss_function(map1, gts) + loss_function(map2, gts) + loss_function(map3, gts) 
            
                # ---- metrics ----
                with torch.no_grad():
                    pred = torch.sigmoid(map1)
                    pred_mask = (pred > 0.7).float()

                    dice_score = dice_m(pred, gts)
                    iou_score  = iou_m(pred, gts)

                    tp = (pred_mask * gts).sum()
                    fp = (pred_mask * (1 - gts)).sum()
                    fn = ((1 - pred_mask) * gts).sum()

                    precision = tp / (tp + fp + 1e-6)
                    recall    = tp / (tp + fn + 1e-6)

                # ---- backward ----
                loss.backward()
                # clip_gradient(optimizer, args.clip)
                optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_record.update(loss.data, args.batchsize)
                    dice.update(dice_score.data, args.batchsize)
                    iou.update(iou_score.data, args.batchsize)
                    precision_record.update(precision.item(), args.batchsize)
                    recall_record.update(recall.item(), args.batchsize)

            # ---- train visualization ----
            if i == total_step:
                print(
                    '{} Training Epoch [{:03d}/{:03d}], '
                    '[loss: {:.4f}, dice: {:.4f}, iou: {:.4f}, '
                    'precision: {:.4f}, recall: {:.4f}]'.format(
                        datetime.now(),
                        epoch,
                        args.num_epochs,
                        loss_record.show(),
                        dice.show(),
                        iou.show(),
                        precision_record.show(),
                        recall_record.show()
                    )
                )

    ckpt_path = save_path + 'last.pth'
    print('[Saving Checkpoint:]', ckpt_path)
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict()
    }
    torch.save(checkpoint, ckpt_path)
    
    return loss_record.show(), dice.show(), iou.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int,
                        default=20, help='epoch number')
    parser.add_argument('--init_lr', type=float,
                        default=1e-3, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--init_trainsize', type=int,
                        default=352, help='training dataset size')
    # parser.add_argument('--clip', type=float,
    #                     default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./data/train', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='MMS+RA')
    parser.add_argument('--resume_path', type=str, help='path to checkpoint for resume training',
                        default='')
    args = parser.parse_args()

    save_path = 'snapshots/{}/'.format(args.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    else:
        print("Save path existed")

    train_img_paths = []
    train_mask_paths = []
    train_img_paths = glob('{}/images/*'.format(args.train_path))
    train_mask_paths = glob('{}/masks/*'.format(args.train_path))
    train_img_paths.sort()
    train_mask_paths.sort()

    train_dataset = Dataset(train_img_paths, train_mask_paths)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    total_step = len(train_loader)

    model = MMSNet().cuda()

    # ---- flops and params ----
    params = model.parameters()
    optimizer = torch.optim.Adam(params, args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                        T_max=len(train_loader)*args.num_epochs,
                                        eta_min=args.init_lr/1000)

    start_epoch = 1
    if args.resume_path != '':
        checkpoint = torch.load(args.resume_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # ---- Training Loop ----
    print("#" * 20, "Start Training", "#" * 20)

    best_dice = 0.0

    for epoch in range(start_epoch, args.num_epochs + 1):

        train_loss, train_dice, train_iou = train(
            train_loader,
            model,
            optimizer,
            epoch,
            lr_scheduler,
            args
        )

        # ---- Save best model ----
        if train_dice > best_dice:
            best_dice = train_dice
            best_epoch = epoch

            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'best_loss': best_dice
            }, os.path.join(save_path, 'best.pth'))

            print(f"✅ Saved best model at epoch {epoch} (loss={best_dice:.4f})")
