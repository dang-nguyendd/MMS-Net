import argparse
import os
import numpy as np
import cv2
from glob import glob
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from model.mms_net_less_branch import MMSNet


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
def multi_inference(model, args):
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
        gt = gt.squeeze(0)   # (352, 352)
        gt = np.asarray(gt, np.float32)
        image = image.cuda()

        res, res2, res3 = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pr = res.round()
        gts.append(gt)
        prs.append(pr)
    get_scores(gts, prs)

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
        pr = (pred > 0.3).astype(np.float32)

        gts.append(gt)
        prs.append(pr)

    get_scores(gts, prs)

def visualize_random_samples(model, args, num_samples=10):
    print("#" * 20)
    model.eval()

    X_test = sorted(glob(f"{args.test_path}/images/*"))
    y_test = sorted(glob(f"{args.test_path}/masks/*"))

    dataset = Dataset(X_test, y_test)

    # ---- Select random indices ----
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        image, gt = dataset[idx]

        # Get filename
        filename = os.path.basename(X_test[idx])

        # Prepare GT
        gt_np = gt.numpy().astype(np.float32)

        # Prepare image for model
        image_input = image.unsqueeze(0).cuda()

        with torch.no_grad():
            outputs = model(image_input)

            # If model returns multiple outputs (like MMSNet)
            if isinstance(outputs, tuple):
                pred = outputs[0]
            else:
                pred = outputs

            pred = F.interpolate(
                pred,
                size=gt_np.shape,
                mode='bilinear',
                align_corners=False
            )

            # If 2-channel output (softmax)
            if pred.shape[1] == 2:
                pred = torch.softmax(pred, dim=1)[:, 1]
            else:
                pred = torch.sigmoid(pred)

            pred = pred.squeeze().cpu().numpy()
            pr_np = (pred > 0.5).astype(np.float32)

        # Convert image back to HWC for plotting
        img_np = image.numpy().transpose(1, 2, 0)

        # ---- Plot ----
        plt.figure(figsize=(12, 4))
        plt.suptitle(filename, fontsize=14, fontweight="bold")
        print(filename)

        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_np, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pr_np, cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave room for suptitle
        plt.show()

# ---------------- Main ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default="./snapshots/MMS+RA/best.pth")
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

    multi_inference(model, args)
    visualize_random_samples(model, args, num_samples=30)

