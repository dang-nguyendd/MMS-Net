import numpy as np
import torch
import torch.nn as nn
import os
import shutil
import random

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        # accept floats or tensors
        if torch.is_tensor(val):
            v = val.detach()
        else:
            v = torch.tensor(val, dtype=torch.float32)

        self.val = v
        self.sum += v.item() * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(v)

    def show(self):
        # get last N values
        idx = max(len(self.losses) - self.num, 0)
        window = self.losses[idx:]

        # stack as tensors and return python float
        return torch.mean(torch.stack(window)).item()
    
    
def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
            

def split_train_test(base_dir = "./data"):
    # Paths
    img_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")

    train_img_dir = os.path.join(base_dir, "train/images")
    train_mask_dir = os.path.join(base_dir, "train/masks")
    test_img_dir = os.path.join(base_dir, "test/images")
    test_mask_dir = os.path.join(base_dir, "test/masks")

    # Create target folders
    for path in [train_img_dir, train_mask_dir, test_img_dir, test_mask_dir]:
        os.makedirs(path, exist_ok=True)

    # List all images
    images = sorted(os.listdir(img_dir))

    # Shuffle for randomness
    random.shuffle(images)

    # 80/20 split
    split_idx = int(0.8 * len(images))
    train_files = images[:split_idx]
    test_files = images[split_idx:]

    # Move files
    for fname in train_files:
        shutil.copy(os.path.join(img_dir, fname), train_img_dir)
        shutil.copy(os.path.join(mask_dir, fname), train_mask_dir)

    for fname in test_files:
        shutil.copy(os.path.join(img_dir, fname), test_img_dir)
        shutil.copy(os.path.join(mask_dir, fname), test_mask_dir)

    print(f"Done! {len(train_files)} train files, {len(test_files)} test files.")

