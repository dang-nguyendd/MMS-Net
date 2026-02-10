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
            

# def split_train_test(base_dir = "./data/ETIS"):
#     # Paths
#     img_dir = os.path.join(base_dir, "images")
#     mask_dir = os.path.join(base_dir, "masks")

#     train_img_dir = os.path.join(base_dir, "train/images")
#     train_mask_dir = os.path.join(base_dir, "train/masks")
#     test_img_dir = os.path.join(base_dir, "test/images")
#     test_mask_dir = os.path.join(base_dir, "test/masks")

#     # Create target folders
#     for path in [train_img_dir, train_mask_dir, test_img_dir, test_mask_dir]:
#         os.makedirs(path, exist_ok=True)

#     # List all images
#     images = sorted(os.listdir(img_dir))

#     # Shuffle for randomness
#     random.shuffle(images)

#     # 80/20 split
#     split_idx = int(0.9 * len(images))
#     train_files = images[:split_idx]
#     test_files = images[split_idx:]

#     # Move files
#     for fname in train_files:
#         shutil.copy(os.path.join(img_dir, fname), train_img_dir)
#         shutil.copy(os.path.join(mask_dir, fname), train_mask_dir)

#     for fname in test_files:
#         shutil.copy(os.path.join(img_dir, fname), test_img_dir)
#         shutil.copy(os.path.join(mask_dir, fname), test_mask_dir)

#     print(f"Done! {len(train_files)} train files, {len(test_files)} test files.")

import os
import shutil
import random
import hashlib

def hash_name(name: str) -> str:
    """Deterministic hash (keeps extension)."""
    stem, ext = os.path.splitext(name)
    h = hashlib.sha256(stem.encode()).hexdigest()[:16]
    return f"{h}{ext}"

def split_train_test(base_dir="./data/ETIS", split_ratio=0.9, seed=42):
    random.seed(seed)

    img_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")

    train_img_dir = os.path.join(base_dir, "train/images")
    train_mask_dir = os.path.join(base_dir, "train/masks")
    test_img_dir = os.path.join(base_dir, "test/images")
    test_mask_dir = os.path.join(base_dir, "test/masks")

    for path in [train_img_dir, train_mask_dir, test_img_dir, test_mask_dir]:
        os.makedirs(path, exist_ok=True)

    # Build (image, mask) pairs safely
    images = sorted(os.listdir(img_dir))
    masks = set(os.listdir(mask_dir))

    pairs = []
    for img in images:
        if img in masks:
            pairs.append(img)
        else:
            print(f"⚠️ No mask found for image: {img}")

    random.shuffle(pairs)

    split_idx = int(split_ratio * len(pairs))
    train_files = pairs[:split_idx]
    test_files = pairs[split_idx:]

    def copy_pairs(files, img_out, mask_out):
        for fname in files:
            new_name = hash_name(fname)

            shutil.copy(
                os.path.join(img_dir, fname),
                os.path.join(img_out, new_name)
            )
            shutil.copy(
                os.path.join(mask_dir, fname),
                os.path.join(mask_out, new_name)
            )

    copy_pairs(train_files, train_img_dir, train_mask_dir)
    copy_pairs(test_files, test_img_dir, test_mask_dir)

    print(f"Done! {len(train_files)} train pairs, {len(test_files)} test pairs.")

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



def convert_multiclass_to_binary_clean(
    input_dir,
    output_dir,
    min_area=100,        # remove tiny blobs (tune this)
    kernel_size=3        # morphology strength
):
    os.makedirs(output_dir, exist_ok=True)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    converted = 0

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
            ):
                continue

            mask_path = os.path.join(root, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

            if mask is None:
                continue

            # RGB → grayscale if needed
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # Step 1: binarize (non-black = foreground)
            binary = np.zeros_like(mask, dtype=np.uint8)
            binary[mask != 0] = 255

            # Step 2: morphological cleanup
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Step 3: remove small connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )

            clean = np.zeros_like(binary)
            for i in range(1, num_labels):  # skip background
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    clean[labels == i] = 255

            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, clean)
            converted += 1

    print(f"✅ Converted and cleaned {converted} masks.")

# import os
# from PIL import Image
# import numpy as np

# input_dir = "data/bkai-igh-neopolyp/train_gt/train_gt"
# output_dir = "data/bkai-igh-neopolyp/train_gt/train_gt_bin"
# os.makedirs(output_dir, exist_ok=True)

# for filename in os.listdir(input_dir):
#     if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
#         img_path = os.path.join(input_dir, filename)

#         img = Image.open(img_path).convert("RGB")
#         arr = np.array(img)

#         # mask for exact black pixels
#         black_mask = np.all(arr == [0, 0, 0], axis=-1)

#         # create output: start all white
#         out = np.ones_like(arr) * 255

#         # keep black pixels
#         out[black_mask] = [0, 0, 0]

#         Image.fromarray(out).save(os.path.join(output_dir, filename))


# input_dir = "data/bkai-igh-neopolyp/train_gt/train_gt"
# output_dir = "data/bkai-igh-neopolyp/train_gt/train_gt_bin"

# convert_multiclass_to_binary_clean(input_dir, output_dir)


