import os
import glob
import numpy as np
import json
import random
import pandas as pd

def load_class_list(anno_path):
    """클래스 리스트를 파일에서 불러옴."""
    file_path = os.path.join(anno_path, "class_list.txt")
    with open(file_path, "r") as file:
        return [line.strip() for line in file]
    
def load_json_files(base_path, class_list, dataset):
    """주어진 클래스 리스트를 기반으로 JSON 파일 리스트를 가져옴."""
    if dataset == "Penn_action" or dataset == "KTH":
        return glob.glob(os.path.join(base_path, "*_result.json"))
    elif dataset == "HAA100":
        json_files = []
        for class_name in class_list:
            class_folder = os.path.join(base_path, class_name)
            if os.path.isdir(class_folder):
                json_files.extend(glob.glob(os.path.join(class_folder, "*_result.json")))
        return json_files

def load_data(base_path):
    data = np.load(os.path.join(base_path, 'processed_keypoints.npy'))
    with open(os.path.join(base_path, "sample_metadata.json"), "r") as f:
        json_data = json.load(f)
    return data, json_data

def expand_array(frames_data, F):
    frames_data = np.array(frames_data) 
    if frames_data.shape[0] == 0:
        return None  # F개 프레임을 0으로 채움
    if frames_data.shape[0] == 1:
        return np.tile(frames_data, (F, 1, 1))  # 단일 프레임을 F번 복제

    indices = np.linspace(0, frames_data.shape[0] - 1, F, dtype=int)
    expanded_array = frames_data[indices]
    return expanded_array

def save_data(output_path, class_data, class_metadata):
    os.makedirs(output_path, exist_ok=True)
    processed_keypoints_path = os.path.join(output_path, "processed_keypoints.npy")
    sample_metadata_path = os.path.join(output_path, "sample_metadata.json")
    
    np.save(processed_keypoints_path, np.array(class_data))
    with open(sample_metadata_path, "w") as f:
        json.dump(class_metadata, f, indent=4)

def find_closest_to_centroid(features, cluster_labels):
    unique_clusters = np.unique(cluster_labels)  # 클러스터 ID 찾기
    closest_indices = {}  # 각 클러스터의 대표 샘플 인덱스 저장

    for cluster in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster)[0]  # 해당 클러스터 샘플의 인덱스
        cluster_points = features[cluster_indices]  # 클러스터에 속한 데이터들

        centroid = np.mean(cluster_points, axis=0)  # 클러스터 평균 벡터 계산
        distances = np.linalg.norm(cluster_points - centroid, axis=1)  # 평균과 각 샘플 간 거리 계산
        closest_idx = cluster_indices[np.argmin(distances)]  # 가장 가까운 샘플의 원본 인덱스 저장

        closest_indices[cluster] = closest_idx  # 결과 저장
    
    return closest_indices

def remove_missing_videos(csv_path, missing_videos, output_csv_path):
    """누락된 비디오를 CSV에서 제거하고 새로운 CSV 파일로 저장"""
    df = pd.read_csv(csv_path, header=None, names=["video_name", "class_label"], sep=",")
    df_filtered = df[~df["video_name"].isin(missing_videos)]  # ✅ 누락된 비디오 제외
    df_filtered.to_csv(output_csv_path, header=False, index=False, sep=",")
    print("--------Removed---------")

def set_seed(seed=42):
    """랜덤 시드를 고정하여 재현성을 보장."""
    np.random.seed(seed)
    random.seed(seed)

def class_mapping(anno_path):
    class_list = load_class_list(anno_path)

    return {name : idx for idx,name in enumerate(class_list)}

def video_class_mapping(args):
    class_list = load_class_list(args.anno_path)
    train_csv = os.path.join(args.anno_path,"train.csv")
    val_csv = os.path.join(args.anno_path,"val.csv")
    train_df = pd.read_csv(train_csv, header=None, names=["video_name", "class_id"], sep=",")
    val_df = pd.read_csv(val_csv, header=None, names=["video_name", "class_id"], sep=",")
    df = pd.concat([train_df, val_df], ignore_index=True)
    df["video_id"] = df["video_name"].str.replace(".mp4", "", regex=False)

    df["class_name"] = df["class_id"].apply(lambda x: class_list[int(x)])
    return dict(zip(df["video_id"], df["class_name"]))