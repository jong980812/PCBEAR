import argparse
import json
import numpy as np
import os
import glob
import pandas as pd
import sys
import subsampling
import clustering
import labeling
import visualize
import make_pose
from datetime import datetime

parser = argparse.ArgumentParser(description='Settings for creating conceptset')
parser.add_argument('--anno_path', default='')
parser.add_argument('--json_path', default='')
parser.add_argument('--output_path', default='')
#parser.add_argument('--save_path', default='')
parser.add_argument('--keyframe_path', default='')
parser.add_argument('--num_subsequence', type=int, default=10)
parser.add_argument('--len_subsequence', type=int, default=16)
parser.add_argument('--dataset', default='Penn_action', 
                    choices=['Penn_action','KTH','HAA100','UCF101'],type=str)
parser.add_argument('--use_partition_num',  type=int, default=1)
parser.add_argument('--subsampling_mode', type=str, default="ver1", choices=["ver1","ver2","ver3","ver4","ver5","sim+conf","wo_cos_sim"])
parser.add_argument('--confidence', type=float, default="0.5")
parser.add_argument('--save_fps', type=int, default=20)
parser.add_argument('--clustering_mode', type=str, default="partition", choices=["req","partition"])
parser.add_argument('--req_cluster', type=int, default = 0)

def concept_decovery(args):
    output_path = os.path.join(args.output_path,args.subsampling_mode,f"L{args.len_subsequence}N{args.num_subsequence}")
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    subsampling.Keypointset(args,output_path)
    data, result_gt, num_concept= clustering.clustering(args,output_path)
    save_path = os.path.join(output_path, f"{num_concept}concepts_fps:{args.save_fps}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)
    closest_sample_indices = labeling.labeling(args,data,result_gt,save_path,output_path,num_concept)
    make_pose.concept_visualize_video(args,data,result_gt,closest_sample_indices,save_path,output_path)
if __name__=='__main__':
    args = parser.parse_args()
    concept_decovery(args)