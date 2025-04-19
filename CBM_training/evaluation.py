import argparse
import sys
from datetime import datetime
import os
import json
import torch
import os
import random
import cbm
import plots
import random
import numpy as np
from video_dataloader import video_utils
import train_video_cbm
from video_dataloader import datasets
import cbm_utils

def load_all_concepts_by_type(load_dir, train_mode):
    concept_dict = {}
    for concept_type in train_mode:
        concept_path = os.path.join(load_dir, concept_type, "concepts.txt")
        if not os.path.exists(concept_path):
            print(f"⚠️ Warning: concepts.txt not found for {concept_type}")
            continue
        with open(concept_path, 'r') as f:
            concepts = f.read().splitlines()
        concept_dict[concept_type] = concepts
    return concept_dict


parser = argparse.ArgumentParser(description='Evaluatin ours')
parser.add_argument('--load_dir',required=True,type=str, default='', help='Trained model path')
parser.add_argument('--batch_size',type=int, default=32, help='Inference batch size')

def main(train_args, test_args):
    # video_utils.init_distributed_mode(test_args)
    train_video_cbm.setup_seed(train_args.seed)
    # save_dir = test_args.load_dir #! 같은 폴더에 저장
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    load_dir=test_args.load_dir
    model = cbm.load_cbm_dynamic(load_dir, device,train_args)
    
    
    cls_file = os.path.join(train_args.video_anno_path, 'class_list.txt')
    with open(cls_file, "r") as f:
        class_names = f.read().split("\n")

    # 사용 예시
    concept_dict = load_all_concepts_by_type(load_dir, train_args.train_mode)
    with open(os.path.join(load_dir, "aggregated", "concepts.txt"), 'r') as f:
        aggregated_concepts = f.read().splitlines()
    assert len(aggregated_concepts) ==model.final.weight.shape[1], f"miss"
    print(f"Concept number: {len(aggregated_concepts)}")

    val_video_dataset,_ =   datasets.build_dataset(False, False, train_args)
    # accuracy = cbm_utils.get_accuracy_cbm(model, val_video_dataset, device,test_args.batch_size,8)
    report, cm = cbm_utils.get_detailed_metrics_cbm(model, val_video_dataset, device, batch_size=test_args.batch_size, class_names=class_names)
    # Save classification report to text file
    report_path = os.path.join(load_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"📄 Saved classification report to {report_path}")
    


if __name__=='__main__':
    test_args = parser.parse_args()
    start_time = datetime.now()
    # 저장된 경로
    args_dir = os.path.join(test_args.load_dir, "args.txt")
    # json 파일 불러오기
    with open(args_dir, 'r') as f:
        args_dict = json.load(f)

    # dict → argparse.Namespace로 변환
    train_args = argparse.Namespace(**args_dict)
    
    main(train_args,test_args)
    end_time = datetime.now()
    print(f"🚀 Run time: {(end_time-start_time).total_seconds():.2f} seconds")  # 초 단위로 변환
