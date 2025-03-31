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

parser = argparse.ArgumentParser(description='Settings for creating conceptset')
parser.add_argument('--anno_path', default='')
parser.add_argument('--json_path', default='')
parser.add_argument('--output_path', default='')
# parser.add_argument('--save_path', default='')
parser.add_argument('--keyframe_path', default='')
parser.add_argument('--num_subsequence', type=int, default=10)
parser.add_argument('--len_subsequence', type=int, default=16)
parser.add_argument('--dataset', default='Penn_action', 
                    choices=['Penn_action','KTH','HAA100'],type=str)
parser.add_argument('--req_cluster',  type=int, default=500)
parser.add_argument('--subsampling_mode', type=str, default="ver1", choices=["ver1","ver2","ver3","ver4","ver5"])
# group_ver2 = parser.add_argument_group('subsampling_mode=ver2 settings')
# group_ver2.add_argument('--stride', type=int, default=1, 
#                         help='Confidence threshold for keypoints filtering (only for ver2)')

def concept_decovery(args):
    last_folder = os.path.basename(args.keyframe_path)
    output_path = os.path.join(args.output_path,args.subsampling_mode,last_folder)
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, "result")
    os.makedirs(save_path, exist_ok=True)
    subsampling.Keypointset(args,output_path)
    data, result_gt = clustering.clustering(args,output_path)
    closest_sample_indices = labeling.labeling(args,data,result_gt,output_path,save_path)
    visualize.concept_visualize(args,data,result_gt,closest_sample_indices,output_path,save_path)
if __name__=='__main__':
    args = parser.parse_args()
    concept_decovery(args)