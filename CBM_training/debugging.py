from collections import OrderedDict
from typing import Tuple, Union
from einops import rearrange
from sparselinear import SparseLinear
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import cbm
import torch
from tqdm import tqdm

from timm.utils import accuracy
import os
import random
import cbm_utils
import data_utils
import similarity
import argparse
import datetime
import torch.optim as optim
import json
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset
import video_utils
import torch.distributed as dist
import torch
import numpy as np
from PIL import Image
from IPython.display import display, Image as IPImage
# Dataloader로부터 얻은 tensor (C, T, H, W)

def visualize_gif(image,label,path,index,img_ind,boring):
    tensor = image
    video_name = path.split('/')[-1].split('.')[0]
    gif_path = f'./gif/{img_ind}_{video_name}_{index}.gif'
    # if not os.path.exists(gif_path):
# 텐서를 (T, H, W, C)로 변환
    tensor = tensor.permute(1, 2, 3, 0)  # (T, H, W, C)

    # 텐서를 numpy 배열로 변환
    tensor_np = tensor.numpy()

    # 이미지 리스트 생성
    images = []
    for i in range(tensor_np.shape[0]):
        # 각 프레임을 (H, W, C) 형태로 변환 후 0~255 범위로 스케일링
        frame = ((tensor_np[i] - tensor_np[i].min()) / (tensor_np[i].max() - tensor_np[i].min()) * 255).astype(np.uint8)
        image = Image.fromarray(frame)
        images.append(image)

    # GIF로 저장 (duration은 각 프레임 사이의 시간, 100ms = 0.1초)
    

    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)
    # else:
        # pass
    # Jupyter에서 GIF 표시
    display(IPImage(filename=gif_path))
# def dual_encoder(args,save_name)

def debug(args,save_name):
    d_val = args.data_set + "_test"
    val_data_t = data_utils.get_data(d_val,args=args)
    val_data_t.end_point = 2
    device = torch.device(args.device)

    model,_ = cbm.load_cbm_triple(save_name, device,args)
    num_object,num_action,num_scene=model.s_proj_layer.weight.shape[0],model.t_proj_layer.weight.shape[0],model.p_proj_layer.weight.shape[0]
    total_concepts = num_object + num_action + num_scene
    k=3
    print("?***? Start test")
    accuracy,concept_counts = cbm_utils.get_accuracy_and_concept_distribution_cbm(model, k, val_data_t, device,64,10,save_name)
    counted_object,counted_action,counted_scene = concept_counts
    # 전체 테스트 샘플 개수 (total_samples)를 직접 넣어주세요.
    total_samples = len(val_data_t)
    print("=== Basic Information ===")
    print(f"Total Samples: {total_samples}")
    print(f"Number of Object Concepts: {num_object}")
    print(f"Number of Action Concepts: {num_action}")
    print(f"Number of Scene Concepts: {num_scene}")
    print(f"Total Number of Concepts: {total_concepts}")
    print("=========================\n\n")
    # 각 속성에 대해 샘플당 평균 사용된 concept 수 계산
# 각 속성의 평균 사용 비율 계산
    avg_usage_object = counted_object / (total_samples*k)
    avg_usage_action = counted_action / (total_samples*k)
    avg_usage_scene = counted_scene / (total_samples*k)

    # 각 속성의 사용 비율을 속성별 concept 개수로 정규화
    normalized_usage_object = avg_usage_object / num_object
    normalized_usage_action = avg_usage_action / num_action
    normalized_usage_scene = avg_usage_scene / num_scene

    # 합이 1이 되도록 각 사용 비율을 조정
    total_normalized_usage = normalized_usage_object + normalized_usage_action + normalized_usage_scene
    ratio_object = normalized_usage_object / total_normalized_usage
    ratio_action = normalized_usage_action / total_normalized_usage
    ratio_scene = normalized_usage_scene / total_normalized_usage

    
    
    
    print(f"Average Usage of Object Concepts per Sample: {avg_usage_object:.3f}")
    print(f"Average Usage of Action Concepts per Sample: {avg_usage_action:.3f}")
    print(f"Average Usage of Scene Concepts per Sample: {avg_usage_scene:.3f}")

    print("\n=== Proportion of Concept Usage ===")
    print(f"Proportion of Object Concepts Usage: {ratio_object:.3f}")
    print(f"Proportion of Action Concepts Usage: {ratio_action:.3f}")
    print(f"Proportion of Scene Concepts Usage: {ratio_scene:.3f}")
    # print("Sum of Ratios (should be 1):", ratio_object + ratio_action + ratio_scene)
    # 결과 출력
    

    # print("Normalized Usage Ratio for Object Concepts:", normalized_usage_object)
    # print("Normalized Usage Ratio for Action Concepts:", normalized_usage_action)
    # print("Normalized Usage Ratio for Scene Concepts:", normalized_usage_scene)
    print("?****? Accuracy: {:.2f}%".format(accuracy*100))