import random
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
import cv2
from glob import glob
import torch
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

        image = cv2.resize(image, (352, 352))
        mask = cv2.resize(mask, (352, 352), interpolation=cv2.INTER_NEAREST)

        image = image.astype("float32") / 255.0
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)

        mask = (mask > 127).astype(np.int64)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
    
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

        # Prepare GT
        gt_np = gt.numpy().astype(np.float32)

        # Prepare image for model
        image_input = image.unsqueeze(0).cuda()

        with torch.no_grad():
            pred = model(image_input)

            pred = F.interpolate(
                pred, size=gt_np.shape,
                mode='bilinear', align_corners=False
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

        plt.tight_layout()
        plt.show()
