import numpy as np
import h5py
from finch import FINCH
import random
import os
import json
from sklearn.metrics import normalized_mutual_info_score as nmi
import subsampling
import re
import pandas as pd
from tqdm import tqdm
import util


def extract_label(args,json_data, class_map):
    video_ids = [item.split('[')[0] for item in json_data]
    if args.dataset == "Penn_action":
        # class_list = subsampling.load_class_list(args.anno_path)
        # df_list = []
        # for csv_file in ["train.csv", "val.csv"]:
        #     csv_path = os.path.join(args.anno_path, csv_file)
        #     df_list.append(pd.read_csv(csv_path, header=None, names=["filename", "class_id"], sep="\s+"))

        # df = pd.concat(df_list, ignore_index=True)
        # df["video_id"] = df["filename"].str.replace(".mp4", "", regex=False).astype(str)
        # df["class_id"] = pd.to_numeric(df["class_id"], errors="coerce")

        # df = df.dropna(subset=["class_id"])
        # df["class_id"] = df["class_id"].astype(int)
        
        # df["class_name"] = df["class_id"].map(lambda x: class_list[int(x)])
        # label_dict = dict(zip(df["video_id"], df["class_name"]))
        label_dict = util.video_class_mapping(args)
        def get_label(video_id):
            class_name = label_dict.get(video_id, "unknown")  # 매칭되지 않으면 "unknown"
            return class_map.get(class_name, -1)

        labels = np.array([get_label(video_id) for video_id in video_ids])
    
    elif args.dataset == "HAA100" :
        def extract_class_name(label):
            return re.sub(r"_\d+$", "", label)
        labels = np.array([class_map.get(extract_class_name(item), -1) for item in video_ids])
    
    elif args.dataset == "UCF101":
        def extract_class_name(label):
            match = re.match(r"v_(.+?)_g\d+_c\d+", label)
            if match:
                return match.group(1)
            else:
                return "unknown"
        labels = np.array([class_map.get(extract_class_name(item), -1) for item in video_ids])

    elif args.dataset == "KTH":
        def extract_class_name(item):
            parts = item.split('_')
            for part in parts:
                if part in class_map:
                    return class_map[part]
            return -1 
        labels = np.array([extract_class_name(item) for item in video_ids])
    return labels

def run_finch_clustering(data, labels, args, output_path):
    """FINCH 클러스터링 후 원하는 partition의 결과로 NMI 계산."""
    use_partition_num = args.use_partition_num
    c, num_clust, req_c = FINCH(
        data,
        req_clust=None,
        use_ann_above_samples=(data.shape[0]-1000),
        verbose=True,
        seed=655
    )
    

    for i in range(c.shape[1]):
        score = nmi(labels, c[:, i])
        print('NMI Score {}: {:.2f}'.format(i, score * 100))

    result_gt = c[:, use_partition_num]
    num_concept = num_clust[use_partition_num]
    selected_score = nmi(labels, result_gt)
    output_txt_path = os.path.join(output_path, f"{num_concept}_output.txt")

    with open(output_txt_path, 'w') as f:
        text = 'Selected Partition: {}, num_concept :{} NMI Score: {:.2f}'.format(use_partition_num, num_concept, selected_score * 100)
        f.write(text + '\n')

    
    
    return result_gt, num_concept
def clustering(args,output_path):
    """메인 실행 함수."""
    util.set_seed(42)  # 랜덤 시드 설정
    data, json_data = util.load_data(output_path)
    data = data.reshape(data.shape[0], -1)
    print(data.shape)
    print(len(json_data))
    class_map = util.class_mapping(args.anno_path)
    labels = extract_label(args, json_data, class_map)
    result_gt, num_concept = run_finch_clustering(data, labels,args,output_path)
    return data,result_gt,num_concept
