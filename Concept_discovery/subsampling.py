import json
import numpy as np
import glob
import os
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys

def haa500_subsampling(args):

    # 데이터 경로 설정
    base_path = args.json_path  # JSON 파일 폴더
    video_base_path = args.video_path  # 원본 비디오 폴더
    output_path = args.output_path # 저장할 폴더
    anno_path = args.anno_path
    

    # 1️⃣ 클래스 리스트 정의
    file_path = os.path.join(anno_path, "class_list.txt")

# 파일 읽어서 리스트에 저장
    with open(file_path, "r") as file:
        class_list = [line.strip() for line in file]

    class_to_id = {cls: i for i, cls in enumerate(class_list)}  # 클래스 이름 → ID 매핑
    subsampling(args, class_list)



def subsampling(args, class_list):
    confidence_threshold = 0.1  # 최소 신뢰도 설정
    scaler = MinMaxScaler(feature_range=(0, 1))
    processed_keypoints_path = os.path.join(args.output_path, "processed_keypoints.npy")
    sample_metadata_path = os.path.join(args.output_path, "sample_metadata.json")
    processed_keypoints_normalized_path = os.path.join(args.output_path, "processed_keypoints_normalized.npy")
    
    L = args.num_sample
    T = args.len_frame
    
    if os.path.exists(processed_keypoints_path):
        print("✅ 기존 keypoints 파일이 존재합니다.")
        processed_keypoints = np.load(processed_keypoints_normalized_path)
        print(f"✅ processed_keypoints shape: {processed_keypoints.shape}")  
        sys.exit()
    else:
        print("🚨 keypoints 파일이 없습니다. 새로 생성합니다.")
    
    # 데이터 불러오기
    json_files = []
    for class_name in class_list:
        class_folder = os.path.join(args.base_path, class_name)
        if os.path.isdir(class_folder):
            json_files.extend(glob.glob(os.path.join(class_folder, "*_result.json")))

    class_data = {cls: [] for cls in class_list}
    class_metadata = {cls: [] for cls in class_list}

    for json_file in json_files:
        class_name = os.path.basename(os.path.dirname(json_file))
        video_id = os.path.basename(json_file).replace("_result.json", "")
        
        with open(json_file, "r") as f:
            keypoints_data = json.load(f)["keypoints"]

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
        
        num_frames = len(frames_data)
        if num_frames >= L * T:
            average_duration = num_frames // L
            sampled_indices = [i * average_duration + random.randint(0, average_duration - 1) for i in range(L)]
            for idx in sampled_indices:
                clip = np.array(frames_data[idx:idx + T])  # (T, 17, 2)
                if len(clip) == T:
                    class_data[class_name].append(clip)
                    class_metadata[class_name].append(video_id)
        else:
            print(f"🚨 {video_id}: {num_frames} 프레임이 부족하여 샘플링할 수 없습니다.")

    np.save(processed_keypoints_path, np.array([clip for v in class_data.values() for clip in v]))
    with open(sample_metadata_path, "w") as f:
        json.dump([meta for v in class_metadata.values() for meta in v], f, indent=4)
