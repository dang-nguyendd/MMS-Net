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
import torch

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

def recall_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall

def precision_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision

def dice_np(y_true, y_pred):
    precision = precision_np(y_true, y_pred)
    recall = recall_np(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+epsilon))

def iou_np(y_true, y_pred):
    intersection = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    union = np.sum(y_true)+np.sum(y_pred)-intersection
    return intersection/(union+epsilon)

def get_scores(gts, prs):
    mean_precision = 0
    mean_recall = 0
    mean_iou = 0
    mean_dice = 0
    for gt, pr in zip(gts, prs):
        mean_precision += precision_np(gt, pr)
        mean_recall += recall_np(gt, pr)
        mean_iou += iou_np(gt, pr)
        mean_dice += dice_np(gt, pr)

    mean_precision /= len(gts)
    mean_recall /= len(gts)
    mean_iou /= len(gts)
    mean_dice /= len(gts)        
    
    print("scores: dice={}, miou={}, precision={}, recall={}".format(mean_dice, mean_iou, mean_precision, mean_recall))

    return (mean_iou, mean_dice, mean_precision, mean_recall)



def inference(model, args):
    print("#"*20)
    model.eval()
    
    X_test = glob('{}/images/*'.format(args.test_path))
    X_test.sort()
    y_test = glob('{}/masks/*'.format(args.test_path))
    y_test.sort()

    test_dataset = Dataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    gts = []
    prs = []
    for i, pack in enumerate(test_loader, start=1):
        image, gt = pack
        gt = gt[0][0]
        gt = np.asarray(gt, np.float32)
        image = image.cuda()

        res, res2, res3, res4 = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pr = res.round()
        gts.append(gt)
        prs.append(pr)
    get_scores(gts, prs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str,
                        default='b3')
    parser.add_argument('--weight', type=str,
                        default='')
    parser.add_argument('--test_path', type=str,
                        default='./data/TestDataset', help='path to dataset')
    args = parser.parse_args()

    model = MMSNet().cuda()

    if args.weight != '':
        ckpt_path = ""
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['state_dict'])

    inference(model, args)

