import json
import numpy as np
import glob
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def load_class_list(anno_path):
    """클래스 리스트를 파일에서 불러옴."""
    file_path = os.path.join(anno_path, "class_list.txt")
    with open(file_path, "r") as file:
        return [line.strip() for line in file]

def load_json_files(base_path, class_list, dataset):
    """주어진 클래스 리스트를 기반으로 JSON 파일 리스트를 가져옴."""
    if dataset == "Penn_action":
        return glob.glob(os.path.join(base_path, "*_result.json"))
    elif dataset == "HAA49":
        json_files = []
        for class_name in class_list:
            class_folder = os.path.join(base_path, class_name)
            if os.path.isdir(class_folder):
                json_files.extend(glob.glob(os.path.join(class_folder, "*_result.json")))
        return json_files
    elif dataset == "KTH":
        return glob.glob(os.path.join(base_path, "*_result.json"))
    

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

def expand_array(frames_data, F):
    frames_data = np.array(frames_data) 
    if frames_data.shape[0] == 0:
        return None  # F개 프레임을 0으로 채움
    if frames_data.shape[0] == 1:
        return np.tile(frames_data, (F, 1, 1))  # 단일 프레임을 F번 복제

    indices = np.linspace(0, frames_data.shape[0] - 1, F, dtype=int)
    expanded_array = frames_data[indices]
    return expanded_array

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
            expanded_frames = expand_array(frames_data, L * T)
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
    
def save_data(output_path, class_data, class_metadata):
    os.makedirs(output_path, exist_ok=True)
    processed_keypoints_path = os.path.join(output_path, "processed_keypoints.npy")
    sample_metadata_path = os.path.join(output_path, "sample_metadata.json")
    
    np.save(processed_keypoints_path, np.array(class_data))
    with open(sample_metadata_path, "w") as f:
        json.dump(class_metadata, f, indent=4)
    

def Keypointset(args):
    class_list = load_class_list(args.anno_path)
    json_files = load_json_files(args.json_path, class_list, args.dataset)
    class_data, class_metadata = subsampling(args, json_files, class_list)
    save_data(args.output_path, class_data, class_metadata)