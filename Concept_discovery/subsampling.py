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


def subsampling_ver1(args, json_files):
    """샘플을 L등분하고, 각 등분에서 T등분하여 
    랜덤으로 길이 T인 subsequence L개 만들기"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    class_data = []
    class_metadata = []
    
    L, T = args.num_subsequence, args.len_subsequence
    
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

def subsampling_ver2(args, json_files):
    """기존 방식. (sliding window 방식) 
    -> 한 칸(stride를 통해 변경 가능)씩 이동하며 샘플링"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    class_data = []
    class_metadata = []
    
    L, T, stride = args.num_subsequence, args.len_subsequence, 1
    
    for json_file in tqdm(json_files, desc="Processing JSON files", leave=True):
        class_name = os.path.basename(os.path.dirname(json_file))
        video_id = os.path.basename(json_file).replace("_result.json", "")
        frames_data = process_keypoints(json_file, scaler)
        
        num_frames = len(frames_data)

        if num_frames == 0:
            print(f"Skipping {json_file} (No valid frames)")
            continue
    
        # 부족한 프레임을 확장
        if num_frames < T:
            expanded_frames = util.expand_array(frames_data, T)
            if expanded_frames is not None:
                frames_data = expanded_frames
                num_frames = len(frames_data)
            else:
                print(f"Skipping {json_file} (Unable to expand frames)")
                continue
        
        for i in range(0, num_frames - T + 1, stride):  # stride만큼 이동동
            clip = np.array(frames_data[i:i+T])  # (T, 17, 2)
            class_data.append(clip)
            class_metadata.append(video_id)

    return class_data, class_metadata    

def subsampling_ver3(args, json_files):
    """Keyframe을 중심으로 일정 구간을 샘플링"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    class_data = []
    class_metadata = []

    L, T = args.num_subsequence, args.len_subsequence
    
    for json_file in tqdm(json_files, desc="Processing JSON files", leave=True):
        video_id = os.path.basename(json_file).replace("_result.json", "")
        frames_data = process_keypoints(json_file, scaler)
        
        num_frames = len(frames_data)
        all_clips = []

        if num_frames == 0:
            print(f"Skipping {json_file} (No valid frames)")
            continue
        
        keyframe_path = os.path.join(args.keyframe_path,video_id, "csvFile", f"{video_id}.txt")
        with open(keyframe_path, 'r') as f:
            keyframes = [int(line.strip()) for line in f.readlines()]
        for keyframe in keyframes:
            # keyframe을 중심으로 앞뒤로 T//2개씩 샘플링
            start = max(keyframe - T//2, 0)  # T개 샘플을 위해 앞뒤로 T//2개씩 선택
            end = min(keyframe + T//2 + 1, num_frames)  # 마지막 프레임을 넘지 않도록

            if end - start < T:
                if start == 0:
                    end = min(start+T, num_frames)
                else :
                    start = max(end-T,0)
            
            sampled_indices = np.arange(start, end)
            
            # 샘플링한 인덱스로 클립 생성
            frames_data = np.array(frames_data)
            clip = frames_data[sampled_indices]
            
            if len(clip) == T:
                all_clips.append(clip)
                class_metadata.append(video_id)
        
        class_data.extend(all_clips)
    
    return class_data, class_metadata

def subsampling_ver4(args, json_files):
    """Keyframe 내에서 uniform sampling"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    class_data = []
    class_metadata = []

    L, T = args.num_subsequence, args.len_subsequence
    
    for json_file in tqdm(json_files, desc="Processing JSON files", leave=True):
        video_id = os.path.basename(json_file).replace("_result.json", "")
        frames_data = process_keypoints(json_file, scaler)
        
        num_frames = len(frames_data)
        all_clips = []

        if num_frames == 0:
            print(f"Skipping {json_file} (No valid frames)")
            continue
        
        keyframe_path = os.path.join(args.keyframe_path,video_id, "csvFile", f"{video_id}.txt")
        with open(keyframe_path, 'r') as f:
            keyframes = [int(line.strip()) for line in f.readlines()]
            
        if len(keyframes) == 0 or keyframes[-1] > num_frames:
            print(f"Skipping {keyframe_path} (No valid frames)")
            continue

        if len(keyframes) < L * T:
            expanded_frames = util.expand_array(keyframes, L * T)
            if expanded_frames is not None:
                keyframes_data = expanded_frames
                num_keyframes = len(keyframes_data)
            else:
                print(f"Skipping {json_file} (Unable to expand frames)")
                continue

        
        # 평균 구간 길이 계산
        segment_length = num_keyframes // L
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
            if sampled_indices[-1] > num_frames:
                continue
            else:
                clip = frames_data[sampled_indices]

            # if np.array(clip).shape != (T,17,2):
            #     print(np.array(clip).shape)

            if len(clip) == T:
                all_clips.append(clip)
                class_metadata.append(video_id)
        
        class_data.extend(all_clips)
    return class_data, class_metadata

def subsampling_final(args, json_files):
    """Keyframe을 중심으로 일정 구간을 샘플링
    keyframe 개수에 따라 샘플링 개수 다르게
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    class_data = []
    class_metadata = []

    L, T = args.num_subsequence, args.len_subsequence
    
    for json_file in tqdm(json_files, desc="Processing JSON files", leave=True):
        video_id = os.path.basename(json_file).replace("_result.json", "")
        frames_data = process_keypoints(json_file, scaler)
        
        num_frames = len(frames_data)
        all_clips = []

        if num_frames == 0:
            print(f"Skipping {json_file} (No valid frames)")
            continue
        elif num_frames < T:
            frames_data = util.repeat_to_clip(num_frames, T)
            print(f"Original frame len : {num_frames}")
            print(f"Expanding frames to {len(frames_data)}")
        
        keyframe_path = os.path.join(args.keyframe_path, video_id, "csvFile", f"{video_id}.txt")
        with open(keyframe_path, 'r') as f:
            keyframes = [int(line.strip()) for line in f.readlines()]
            
        if len(keyframes) == 0:
            print(f"Skipping {keyframe_path} (No valid keyframes)")
            continue
        
        # L, keyframe 개수 관계에 따른 처리
        selected_keyframes = []
        
        if L > len(keyframes):
            # L > keyframe: keyframe 개수만큼만
            selected_keyframes = keyframes
        elif L < len(keyframes):
            indices = np.linspace(0, len(keyframes)-1, L, dtype=int)
            selected_keyframes = [keyframes[i] for i in indices]
        else:  # L == len(keyframes)
            selected_keyframes = keyframes
        
        frames_data = np.array(frames_data)
        
        for keyframe in selected_keyframes:
            # keyframe을 중심으로 앞뒤로 T//2개씩 샘플링
            start = max(keyframe - T//2, 0)  # T개 샘플을 위해 앞뒤로 T//2개씩 선택
            end = min(keyframe + T//2 + 1, num_frames)  # 마지막 프레임을 넘지 않도록

            if end - start < T:
                if start == 0:
                    end = min(start+T, num_frames)
                else:
                    start = max(end-T, 0)
            
            sampled_indices = np.arange(start, end)
            
            # 샘플링한 인덱스로 클립 생성
            clip = frames_data[sampled_indices]
            
            if len(clip) == T:
                all_clips.append(clip)
                class_metadata.append(video_id)
        
        class_data.extend(all_clips)
    
    return class_data, class_metadata


def Keypointset(args, save_path):
    processed_keypoints_path = os.path.join(save_path, "processed_keypoints.npy")
    if os.path.exists(processed_keypoints_path):
        print(f"✅ {processed_keypoints_path} 파일이 존재하므로 Keypointset()을 건너뜁니다.")
        return  
    class_list = util.load_class_list(args.anno_path)
    json_files = util.load_json_files(args.json_path, class_list, args.dataset)
    if args.subsampling_mode == "ver1":
        class_data, class_metadata = subsampling_ver1(args, json_files)
    elif args.subsampling_mode == "ver2":
        class_data, class_metadata = subsampling_ver2(args, json_files)
    elif args.subsampling_mode == "ver3":
        class_data, class_metadata = subsampling_ver3(args, json_files)
    elif args.subsampling_mode == "ver4":
        class_data, class_metadata = subsampling_ver4(args, json_files)
    elif args.subsampling_mode == "ver5":
        class_data, class_metadata = subsampling_final(args, json_files)
    print(np.array(class_data).shape)
    print(np.array(class_metadata).shape)
    util.save_data(save_path, class_data, class_metadata)

# if __name__ == "__main__":
#     import argparse

#     # Argument parser 설정
#     parser = argparse.ArgumentParser(description='Settings for creating conceptset')
#     parser.add_argument('--anno_path', default='')
#     parser.add_argument('--json_path', default='')
#     parser.add_argument('--output_path', default='')
#     parser.add_argument('--save_path', default='')
#     parser.add_argument('--keyframe_path', default='')
#     parser.add_argument('--num_subsequence', type=int, default=10)
#     parser.add_argument('--len_subsequence', type=int, default=16)
#     parser.add_argument('--dataset', default='Penn_action', 
#                         choices=['Penn_action','KTH','HAA100'],type=str)
#     # parser.add_argument('--req_cluster',  type=int, default=500)
#     parser.add_argument('--subsampling_mode', type=str, default="ver1", choices=["ver1","ver2","ver3","ver4"])

#     args = parser.parse_args()
#     output_path = os.path.join(args.output_path,args.subsampling_mode)
#     Keypointset(args,output_path)