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

        labels = np.array([get_label(video_id) for video_id in json_data])
    
    elif args.dataset == "HAA49":
        def extract_class_name(label):
            return re.sub(r"_\d+$", "", label)
        labels = np.array([class_map.get(extract_class_name(item), -1) for item in json_data])

    elif args.dataset == "KTH":
        def extract_class_name(item):
            parts = item.split('_')
            for part in parts:
                if part in class_map:
                    return class_map[part]
            return -1 
        labels = np.array([extract_class_name(item) for item in json_data])
    return labels

def run_finch_clustering(data, labels,args):
    """FINCH 클러스터링 실행 및 NMI 점수 계산."""
    # output_txt_path = os.path.join(args.output_path, "nmi_score.txt")
    c, num_clust, req_c = FINCH(data, req_clust=args.req_cluster, use_ann_above_samples=(data.shape[0]-1000), verbose=True, seed=655)
    
    req_score = nmi(labels, req_c)
    req_score_text = f'NMI Score for req_cluster : {req_score*100:2f}'
        # for i in range(c.shape[1]):
        #     score = nmi(labels, c[:, i])
        #     score_text = f'NMI Score {i} : {score*100:2f}'
        #     f.write(score_text + "\n")
    print(req_score_text)
    return req_c
def clustering(args,output_path):
    """메인 실행 함수."""
    util.set_seed(42)  # 랜덤 시드 설정
    data, json_data = util.load_data(output_path)
    data = data.reshape(data.shape[0], -1)
    class_map = util.class_mapping(args.anno_path)
    labels = extract_label(args, json_data, class_map)
    result_gt = run_finch_clustering(data, labels,args)
    return data,result_gt
