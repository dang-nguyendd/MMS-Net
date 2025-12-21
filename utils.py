import numpy as np
import torch
import os
import shutil
import random
import numpy as np
import os
import cv2
from glob import glob


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

def histogram_equalise(
        input_dir="./data/test/images",
        output_dir="./data/test_hist/images",
        clip_limit=2.0,
        tile_grid_size=(8, 8)
    ):
        os.makedirs(output_dir, exist_ok=True)

        image_paths = glob(os.path.join(input_dir, "*"))

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Convert BGR → LAB
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE to L channel
            l_eq = clahe.apply(l)

            # Merge channels back
            lab_eq = cv2.merge((l_eq, a, b))
            img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

            # Save result
            filename = os.path.basename(img_path)
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, img_eq)
        

        os.makedirs(output_dir, exist_ok=True)
        image_paths = glob(os.path.join(input_dir, "*.*"))

        count = 0
        for img_path in image_paths:
            if os.path.isfile(img_path):
                filename = os.path.basename(img_path)
                dst_path = os.path.join(output_dir, filename)
                shutil.copy2(img_path, dst_path)
                count += 1

        print(f"CLAHE applied to {len(image_paths)} images.")
        print(f"Saved results to: {output_dir}")

        # change images -> masks
        input_dir = input_dir.replace("images", "masks")
        output_dir = output_dir.replace("images", "masks")

        os.makedirs(output_dir, exist_ok=True)

        image_paths = glob(os.path.join(input_dir, "*.*"))

        count = 0
        for img_path in image_paths:
            if os.path.isfile(img_path):
                filename = os.path.basename(img_path)
                dst_path = os.path.join(output_dir, filename)

                shutil.copy2(img_path, dst_path)
                count += 1

        print(f"Copied {count} files.")
        print(f"From: {input_dir}")
        print(f"To:   {output_dir}")


histogram_equalise()