import numpy as np
import os
import clustering
import json
from collections import defaultdict
import pandas as pd
import pickle
import util


def make_attribute(args, result_gt,save_path,output_path,num_concept):
    with open(os.path.join(output_path, "sample_metadata.json"), "r") as f :
        json_data = json.load(f)
    labels = result_gt
    video_frames = defaultdict(list)
    for idx, item in enumerate(json_data):
        video_id = item.split('[')[0]
        video_frames[video_id].append(labels[idx])
    video_attributes = []
    for video_id, labels_ in video_frames.items():
        one_hot_vector = np.zeros(num_concept, dtype=int)
        for label in labels_:
            one_hot_vector[label] = 1  # 등장한 클러스터에 1 할당
        if args.dataset == "Penn_action" or args.dataset == "KTH":
            video_name = f"{video_id}.mp4"
        elif args.dataset == "HAA100" :
            video_name = f"{video_id.rsplit('_', 1)[0]}/{video_id}.mp4"
        elif args.dataset == "UCF101":
    # video_id 예시: v_ApplyEyeMakeup_g10_c03
            if video_id.startswith("v_"):
                # v_ 제거 → ApplyEyeMakeup_g10_c03
                no_prefix = video_id[2:]
                # 클래스 이름만 추출: ApplyEyeMakeup
                class_name = no_prefix.split('_')[0]
                if class_name == "HandStandPushups":
                    class_name = "HandstandPushups"
                video_name = f"{class_name}/{video_id}.avi"
            else:
                video_name = f"{video_id}.avi"  # fallback
        video_entry = {
            "video_name": video_name,
            "attribute_label": one_hot_vector.tolist()
        }
        
        video_attributes.append(video_entry)

    # JSON 파일로 저장
    output_annotation_path = os.path.join(save_path,"video_attributes.json")
    with open(output_annotation_path, "w") as f:
        json.dump(video_attributes, f, indent=4)
    return video_attributes


def hard_label(video_attributes, args, save_path, num_concept, mode):
    output_json_path = os.path.join(save_path, f"hard_label_{mode}.json")
    output_pkl_path = os.path.join(save_path, f"hard_label_{mode}.pkl")
    csv_path = os.path.join(args.anno_path, f"{mode}.csv")
    new_csv_path = os.path.join(save_path, f"{mode}.csv") 
    cnt = 0

    df = pd.read_csv(csv_path, header=None, names=["video_name", "class_label"], sep=",")
    video_list = df["video_name"].tolist()

    annotation_dict = {
        item["video_name"]: item["attribute_label"]
        for item in video_attributes
    }
    sorted_annotations = []
    #missing_videos = []

    for video_name in video_list:
        if video_name in annotation_dict:
            sorted_annotations.append({
                "video_name": video_name,
                "attribute_label" : annotation_dict[video_name]
            })
        else :
            # missing_videos.append(video_name)
            print(f"Missing video: {video_name}")
            cnt+=1
            sorted_annotations.append({
            "video_name": video_name,
            "attribute_label": [-1] * num_concept
        })
    print(f"Number of missing video : {cnt}")
    with open(output_json_path, "w") as f:
        json.dump(sorted_annotations, f, indent=4)

    # ✅ Pickle 파일로 저장
    with open(output_pkl_path, "wb") as f:
        pickle.dump(sorted_annotations, f)
    # util.remove_missing_videos(csv_path, missing_videos, new_csv_path)

    return sorted_annotations

def labeling(args, data, result_gt,save_path,output_path,num_concept):
    # data, result_gt = clustering.clustering(args)
    closest_sample_indices = util.find_closest_to_centroid(data, result_gt)
    with open(os.path.join(save_path,'concept_index.txt'), "w", encoding="utf-8") as f:
        for key, value in closest_sample_indices.items():
            f.write(f"{key}: {value}\n") 

    video_attributes = make_attribute(args, result_gt,save_path,output_path,num_concept)
    train_anno = hard_label(video_attributes, args, save_path, num_concept,mode = "train")
    if args.dataset == "UCF101":
        val_anno = hard_label(video_attributes, args, save_path,num_concept, mode = "test")
    else :
        val_anno = hard_label(video_attributes, args, save_path,num_concept, mode = "val")
    return closest_sample_indices

