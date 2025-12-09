import torch

ckpt = torch.load("snapshots/MMS-Net+/last.pth")

print(ckpt["epoch"])

# def print_nested(name, obj, indent=0):
#     pad = "  " * indent
#     if isinstance(obj, dict):
#         print(f"{pad}{name}: dict with {len(obj)} keys")
#         for k, v in obj.items():
#             print_nested(k, v, indent+1)
#     elif torch.is_tensor(obj):
#         print(f"{pad}{name}: Tensor {tuple(obj.shape)} dtype={obj.dtype}")
#     elif isinstance(obj, (list, tuple)):
#         print(f"{pad}{name}: {type(obj).__name__} length={len(obj)}")
#     else:
#         print(f"{pad}{name}: {obj}")

# print("=== CHECKPOINT CONTENT ===")
# for key, value in ckpt.items():
#     print_nested(key, value)
