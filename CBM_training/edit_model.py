import torch

# ✅ 기존 모델 로드
model = torch.load("/data/dataset/videocbm/backbones/haa500-subset_backbone.pt")
print(model)