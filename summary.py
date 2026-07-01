from model.mms_net_2 import MMSNet
import torch
import torch.nn as nn
from torchinfo import summary

model = MMSNet()
inp = torch.randn(1, 3, 128, 128) 
out1, out2, out3, out_bgm = model(inp)

print("Output 1 shape:", out1.shape)
print("Output 2 shape:", out2.shape)
print("Output 3 shape:", out3.shape)
print("Output bgm shape:", out_bgm.shape)


summary(model, input_size=(1, 3, 128, 128))

print(torch.cuda.is_available())
print(torch.cuda.get_device_name())


trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
buffers = sum(b.numel() for b in model.buffers())

print("Trainable params:", trainable)
print("Non-trainable params:", non_trainable)
print("Non-trainable buffers:", buffers)
print("Total params including buffers:", trainable + non_trainable + buffers)
