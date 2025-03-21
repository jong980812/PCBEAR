import json
import numpy as np
import glob
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import util

    

def process_keypoints(json_file, scaler, confidence_threshold=0.1):
    """JSON 파일에서 keypoints를 추출하고 정규화."""
    with open(json_file, "r") as f:
        keypoints_data = json.load(f).get("keypoints", [])
    
    frames_data = []
    for frame in keypoints_data:
        if "0" in frame:
            keypoints = np.array(frame["0"])  # (17, 3)
            mean_confidence = np.mean(keypoints[:, 2])
            if mean_confidence < confidence_threshold:
                continue
            pose = keypoints[:, :2]
            mean_pose = np.mean(pose, axis=0)
            pose_centered = pose - mean_pose
            pose_normalized = scaler.fit_transform(pose_centered)
            frames_data.append(pose_normalized)
    return frames_data


def subsampling(args, json_files, class_list):
    """샘플링 및 데이터 저장."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    class_data = []
    class_metadata = []
    
    L, T = args.num_sample, args.len_frame
    
    for json_file in tqdm(json_files, desc="Processing JSON files", leave=True):
        class_name = os.path.basename(os.path.dirname(json_file))
        video_id = os.path.basename(json_file).replace("_result.json", "")
        frames_data = process_keypoints(json_file, scaler)
        
        num_frames = len(frames_data)

        if num_frames == 0:
            print(f"Skipping {json_file} (No valid frames)")
            continue
    
        # 부족한 프레임을 확장
        if num_frames < L * T:
            expanded_frames = util.expand_array(frames_data, L * T)
            if expanded_frames is not None:
                frames_data = expanded_frames
                num_frames = len(frames_data)
            else:
                print(f"Skipping {json_file} (Unable to expand frames)")
                continue

        
        # 평균 구간 길이 계산
        segment_length = num_frames // L
        all_clips = []
        
        for i in range(L):
            segment_start = i * segment_length
            segment_end = (i + 1) * segment_length
            
            # T개의 작은 구간으로 나누기
            sub_segment_length = max(segment_length // T, 1)
            sampled_indices = [segment_start + j * sub_segment_length + np.random.randint(0, sub_segment_length) 
                            for j in range(T)]
            if len(sampled_indices) != T:
                print(len(sampled_indices))
            
            # 샘플링한 인덱스로 클립 생성
            frames_data = np.array(frames_data) 
            clip = frames_data[sampled_indices]
            if len(clip) == T:
                all_clips.append(clip)
                class_metadata.append(video_id)
        
        class_data.extend(all_clips)
    return class_data, class_metadata
        

def Keypointset(args):
    processed_keypoints_path = os.path.join(args.output_path, "processed_keypoints.npy")
    if os.path.exists(processed_keypoints_path):
        print(f"✅ {processed_keypoints_path} 파일이 존재하므로 Keypointset()을 건너뜁니다.")
        return  
    class_list = util.load_class_list(args.anno_path)
    json_files = util.load_json_files(args.json_path, class_list, args.dataset)
    class_data, class_metadata = subsampling(args, json_files, class_list)
    util.save_data(args.output_path, class_data, class_metadata)