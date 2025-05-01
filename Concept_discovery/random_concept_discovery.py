from sklearn.metrics import normalized_mutual_info_score as nmi
from collections import defaultdict
import numpy as np
import os
import json
import pickle
import pandas as pd
    
def labeling_from_random_sampling(args, data, class_metadata, save_path, output_path):
    class_names = [meta.split('/')[0] for meta in class_metadata]
    unique_classes = sorted(set(class_names))
    class_to_label = {cls: i for i, cls in enumerate(unique_classes)}
    num_concept = len(unique_classes)
    result_gt = np.array([class_to_label[cls] for cls in class_names])

    closest_sample_indices = defaultdict(int)
    for concept_id in range(num_concept):
        indices = np.where(result_gt == concept_id)[0]
        if len(indices) > 0:
            closest_sample_indices[concept_id] = int(indices[0])  # 첫 번째 sample 선택

    with open(os.path.join(save_path, 'concept_index.txt'), "w", encoding="utf-8") as f:
        for key, value in closest_sample_indices.items():
            f.write(f"{key}: {value}\n")

    # 3. sample_metadata.json 저장 (class_metadata 기반)
    with open(os.path.join(output_path, "sample_metadata.json"), "w") as f:
        json.dump(class_metadata, f, indent=4)

    # 4. make_attribute() 방식 구현
    video_frames = defaultdict(list)
    for idx, item in enumerate(class_metadata):
        video_id = item.split('[')[0]
        video_frames[video_id].append(result_gt[idx])

    video_attributes = []
    for video_id, labels_ in video_frames.items():
        one_hot_vector = np.zeros(num_concept, dtype=int)
        for label in labels_:
            one_hot_vector[label] = 1
        if args.dataset in ["Penn_action", "KTH"]:
            video_name = f"{video_id}.mp4"
        elif args.dataset == "HAA100":
            video_name = f"{video_id.rsplit('_', 1)[0]}/{video_id}.mp4"
        elif args.dataset == "UCF101":
            if video_id.startswith("v_"):
                no_prefix = video_id[2:]
                class_name = no_prefix.split('_')[0]
                if class_name == "HandStandPushups":
                    class_name = "HandstandPushups"
                video_name = f"{class_name}/{video_id}.avi"
            else:
                video_name = f"{video_id}.avi"
        else:
            video_name = f"{video_id}.mp4"

        video_attributes.append({
            "video_name": video_name,
            "attribute_label": one_hot_vector.tolist()
        })

    with open(os.path.join(save_path, "video_attributes.json"), "w") as f:
        json.dump(video_attributes, f, indent=4)

    # 5. hard_label 생성 함수
    def hard_label(video_attributes, mode):
        output_json_path = os.path.join(save_path, f"hard_label_{mode}.json")
        output_pkl_path = os.path.join(save_path, f"hard_label_{mode}.pkl")
        csv_path = os.path.join(args.anno_path, f"{mode}.csv")
        df = pd.read_csv(csv_path, header=None, names=["video_name", "class_label"], sep=",")
        video_list = df["video_name"].tolist()

        annotation_dict = {
            item["video_name"]: item["attribute_label"]
            for item in video_attributes
        }
        sorted_annotations = []
        cnt = 0
        for video_name in video_list:
            if video_name in annotation_dict:
                sorted_annotations.append({
                    "video_name": video_name,
                    "attribute_label": annotation_dict[video_name]
                })
            else:
                cnt += 1
                print(f"Missing video: {video_name}")
                sorted_annotations.append({
                    "video_name": video_name,
                    "attribute_label": [-1] * num_concept
                })
        print(f"Missing videos in {mode}: {cnt}")

        with open(output_json_path, "w") as f:
            json.dump(sorted_annotations, f, indent=4)
        with open(output_pkl_path, "wb") as f:
            pickle.dump(sorted_annotations, f)

        return sorted_annotations

    train_anno = hard_label(video_attributes, "train")
    val_anno = hard_label(video_attributes, "test" if args.dataset == "UCF101" else "val")

    return closest_sample_indices