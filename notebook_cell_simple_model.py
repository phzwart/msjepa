# Single cell: 160×160 input, 8×8 patches — copy into a notebook and run
import torch
from msjepa import MSJEPA, MSJEPAConfig

config = MSJEPAConfig(image_size=(160, 160), patch_size=(8, 8), stride=(8, 8))
model = MSJEPA(config).eval()
x = torch.randn(2, config.in_channels, 160, 160)
with torch.no_grad():
    out = model(x)
print("student dense feature map:", out.student.dense_feature_map.shape)
print("teacher dense feature map:", out.teacher.dense_feature_map.shape)
