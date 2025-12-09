import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils import clip_gradient, AvgMeter
from torch.autograd import Variable
from datetime import datetime
import torch.nn.functional as F

from model.mms_base import MMSNet


class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = cv2.resize(image, (352, 352))
            mask = cv2.resize(mask, (352, 352), interpolation=cv2.INTER_NEAREST)

        # image
        image = image.astype("float32") / 255.0
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)

        # mask (class indices: 0 or 1)
        mask = (mask > 127).astype(np.int64)   # binarize
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


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

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: N × C × H × W (logits)
        # target: N × H × W (class indices)

        pred = torch.softmax(pred, dim=1)

        # convert to one-hot: target_onehot: N × C × H × W
        target_onehot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1])
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        intersection = (pred * target_onehot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def train(train_loader, model, optimizer, epoch, lr_scheduler, args):
    model.train()

    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    dice_meter, iou_meter = AvgMeter(), AvgMeter()

    with torch.autograd.set_detect_anomaly(True):
        for i, pack in enumerate(tqdm(train_loader, total=total_step), start=1):

            # Warmup LR
            if epoch <= 1:
                optimizer.param_groups[0]["lr"] = (epoch * i) / (1.0 * total_step) * args.init_lr
            else:
                lr_scheduler.step()

            images, gts = pack
            images = images.cuda()
            gts = gts.cuda()              # (B,H,W)

            if gts.dim() == 4:
                gts = gts.squeeze(1)      # (B,1,H,W) → (B,H,W)
            gts = gts.long()              # class indices

            for rate in size_rates:

                optimizer.zero_grad()

                trainsize = int(np.ceil(args.init_trainsize * rate / 32) * 32)

                images_resized = F.interpolate(
                    images,
                    size=(trainsize, trainsize),
                    mode="bilinear",
                    align_corners=True
                )

                gts_resized = F.interpolate(
                    gts.unsqueeze(1).float(),
                    size=(trainsize, trainsize),
                    mode="nearest"
                ).squeeze(1).long()       # BACK to int labels

                # ---- forward ----

                pred = model(images_resized)       # (B,2,H,W)

                # ensure pred matches target size
                if pred.shape[2:] != gts_resized.shape[1:]:
                    pred = F.interpolate(pred, size=gts_resized.shape[1:], mode='bilinear', align_corners=False)

                loss_fn = DiceLoss()
                loss = loss_fn(pred, gts_resized)

                # ---- metrics ----
                prob = torch.softmax(pred, dim=1)[:, 1]     # (B,H,W)
                dice_score = dice_m(prob, gts_resized.float())
                iou_score = iou_m(prob, gts_resized.float())

                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, args.clip)
                optimizer.step()

                # ---- logging ----
                if rate == 1:
                    loss_record.update(loss.item(), args.batchsize)
                    dice_meter.update(dice_score.item(), args.batchsize)
                    iou_meter.update(iou_score.item(), args.batchsize)

            if i == total_step:
                print('{} Training Epoch [{:03d}/{:03d}], '
                      '[loss: {:.4f}, dice: {:.4f}, iou: {:.4f}]'.format(
                          datetime.now(), epoch, args.num_epochs,
                          loss_record.show(), dice_meter.show(), iou_meter.show()
                ))


    # ---- save checkpoint ----
    ckpt_path = save_path + 'last.pth'
    print('[Saving Checkpoint:]', ckpt_path)

    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict()
    }
    torch.save(checkpoint, ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=70, help='epoch number')
    parser.add_argument('--init_lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--init_trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str, default='./data/train', help='path to train dataset')
    parser.add_argument('--train_save', type=str, default='MMS-Net+')
    parser.add_argument('--resume_path', type=str, default='', help='path to checkpoint for resume training')
    args = parser.parse_args()

    save_path = f'snapshots/{args.train_save}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    else:
        print("Save path existed")

    # ---- Load training images ----
    train_img_paths = sorted(glob(f'{args.train_path}/images/*'))
    train_mask_paths = sorted(glob(f'{args.train_path}/masks/*'))

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

    optimizer = torch.optim.Adam(model.parameters(), args.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_step * args.num_epochs,
        eta_min=args.init_lr/1000
    )

    # ---- Resume Training ----
    start_epoch = 1
    if args.resume_path != '':
        print(f"Loading checkpoint: {args.resume_path}")
        checkpoint = torch.load(args.resume_path)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])

        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")

    # ---- Training Loop ----
    print("#" * 20, "Start Training", "#" * 20)
    for epoch in range(start_epoch, args.num_epochs + 1):
        train(train_loader, model, optimizer, epoch, lr_scheduler, args)
