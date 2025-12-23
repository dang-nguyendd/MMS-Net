import argparse
import os
import numpy as np
import cv2
from glob import glob
import torch
import torch.nn.functional as F

from model.mms_base import MMSNet


# ---------------- Dataset ----------------
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

        image = cv2.resize(image, (352, 352))
        mask = cv2.resize(mask, (352, 352), interpolation=cv2.INTER_NEAREST)

        image = image.astype("float32") / 255.0
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)

        mask = (mask > 127).astype(np.int64)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


# --------------- Metrics ------------------
epsilon = 1e-7

def recall_np(gt, pr):
    tp = np.sum(gt * pr)
    pp = np.sum(gt)
    return tp / (pp + epsilon)

def precision_np(gt, pr):
    tp = np.sum(gt * pr)
    pp = np.sum(pr)
    return tp / (pp + epsilon)

def dice_np(gt, pr):
    p = precision_np(gt, pr)
    r = recall_np(gt, pr)
    return 2 * (p * r) / (p + r + epsilon)

def iou_np(gt, pr):
    inter = np.sum(gt * pr)
    union = np.sum(gt) + np.sum(pr) - inter
    return inter / (union + epsilon)


def get_scores(gts, prs):
    dices = []
    ious = []
    precs = []
    recs = []

    for gt, pr in zip(gts, prs):
        dices.append(dice_np(gt, pr))
        ious.append(iou_np(gt, pr))
        precs.append(precision_np(gt, pr))
        recs.append(recall_np(gt, pr))

    print("Dice:", np.mean(dices))
    print("IoU:", np.mean(ious))
    print("Precision:", np.mean(precs))
    print("Recall:", np.mean(recs))


# ---------------- Inference ----------------
def inference(model, args):
    print("#" * 20)
    model.eval()

    X_test = sorted(glob(f"{args.test_path}/images/*"))
    y_test = sorted(glob(f"{args.test_path}/masks/*"))

    dataset = Dataset(X_test, y_test)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, pin_memory=True)

    gts = []
    prs = []

    for image, gt in loader:
        gt = gt.squeeze().numpy().astype(np.float32)  # fix shape

        image = image.cuda()

        pred = model(image)
        pred = F.interpolate(pred, size=gt.shape, mode='bilinear', align_corners=False)

        # --- Case 1: model output is 2 channels (softmax) ---
        if pred.shape[1] == 2:
            pred = torch.softmax(pred, dim=1)[:, 1]

        # --- Case 2: model output is 1 channel (sigmoid) ---
        else:
            pred = torch.sigmoid(pred)

        pred = pred.detach().cpu().numpy().squeeze()
        pr = (pred > 0.5).astype(np.float32)

        gts.append(gt)
        prs.append(pr)

    get_scores(gts, prs)


# ---------------- Main ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default="./snapshots/MMS-Net+/last.pth")
    parser.add_argument("--test_path", type=str,
                        default="./data/test")
    args = parser.parse_args()

    model = MMSNet().cuda()

    if args.weight != "":
        checkpoint = torch.load(args.weight)
        model.load_state_dict(checkpoint["state_dict"])
        # checkpoint = torch.load(args.weight, weights_only=True)
        # model.load_state_dict(checkpoint)
        print("Loaded weights:", args.weight)

    inference(model, args)
