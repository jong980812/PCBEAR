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
from sklearn.cluster import KMeans
from minisom import MiniSom
from sklearn.preprocessing import LabelEncoder


def extract_label(args,json_data, class_map):
    video_ids = [item.split('[')[0] for item in json_data]
    if args.dataset == "Penn_action":
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
    if args.clustering_mode == "partition":
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
        output_txt_path = os.path.join(output_path, f"L{args.len_subsequence}N{args.num_subsequence}_{num_concept}_output.txt")
        with open(output_txt_path, 'w') as f:
            text = 'Selected Partition: {}, num_concept :{} NMI Score: {:.2f}'.format(use_partition_num, num_concept, selected_score * 100)
            f.write(text + '\n')

        return result_gt, num_concept
    elif args.clustering_mode == "req":
        c, num_clust, req_c = FINCH(data, req_clust=args.req_cluster, use_ann_above_samples=(data.shape[0]-1000), verbose=True, seed=655)
        req_score = nmi(labels, req_c)
        output_txt_path = os.path.join(output_path, f"L{args.len_subsequence}N{args.num_subsequence}_{args.req_cluster}_output.txt")
        with open(output_txt_path, 'w') as f:
            text = 'num_concept :{} NMI Score: {:.2f}'.format(args.req_cluster, req_score * 100)
            f.write(text + '\n')
        
        return req_c, args.req_cluster


def run_kmeans_clustering(data, labels, args, output_path):
    kmeans = KMeans(n_clusters=args.req_cluster, random_state=655, n_init='auto')  
    cluster_labels = kmeans.fit_predict(data)

    # NMI 계산
    score = nmi(labels, cluster_labels)
    
    # 결과 저장
    output_txt_path = os.path.join(
        output_path, f"L{args.len_subsequence}N{args.num_subsequence}_{args.req_cluster}_output.txt"
    )
    with open(output_txt_path, 'w') as f:
        text = f"num_concept :{args.req_cluster} NMI Score: {score * 100:.2f}"
        f.write(text + '\n')
    
    return cluster_labels, args.req_cluster


def run_som_clustering(data, labels, args, output_path):
    """SOM 기반 클러스터링 후 NMI 계산 및 저장"""
    
    n_clusters = args.req_cluster

    # SOM 설정 (가로 x 세로 셀 수 = 원하는 클러스터 수)
    som_side = int(np.ceil(np.sqrt(n_clusters)))
    som = MiniSom(x=som_side, y=som_side, input_len=data.shape[1], sigma=1.0, learning_rate=0.5, random_seed=655)
    
    som.random_weights_init(data)
    
    som.train_random(data, num_iteration=1000)
    
    bmu_list = np.array([som.winner(x) for x in data])  
    cluster_labels = np.ravel_multi_index((bmu_list[:,0], bmu_list[:,1]), dims=(som_side, som_side))  # 2D → 1D
    cluster_labels = LabelEncoder().fit_transform(cluster_labels)
    

    num_concept = len(np.unique(cluster_labels))

    # NMI 계산
    score = nmi(labels, cluster_labels)

    # 결과 저장
    output_txt_path = os.path.join(output_path, f"L{args.len_subsequence}N{args.num_subsequence}_{num_concept}_output.txt")
    with open(output_txt_path, 'w') as f:
        text = f"num_concept :{num_concept} NMI Score: {score * 100:.2f}"
        f.write(text + '\n')
    
    return cluster_labels, num_concept
    
    
    
def clustering(args,output_path):
    """메인 실행 함수."""
    util.set_seed(42)  # 랜덤 시드 설정
    data, json_data = util.load_data(output_path)
    data = data.reshape(data.shape[0], -1)
    print(data.shape)
    print(len(json_data))
    class_map = util.class_mapping(args.anno_path)
    labels = extract_label(args, json_data, class_map)
    if args.clustering_mode == "req" or args.clustering_mode == "partition":
        result_gt, num_concept = run_finch_clustering(data, labels,args,output_path)
    elif args.clustering_mode == "k-means":
        result_gt, num_concept = run_kmeans_clustering(data, labels,args,output_path)
    elif args.clustering_mode == "som":
        result_gt, num_concept = run_som_clustering(data, labels,args,output_path)
    return data,result_gt,num_concept
