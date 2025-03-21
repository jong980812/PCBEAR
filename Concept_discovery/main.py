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
parser.add_argument('--save_path', default='')
parser.add_argument('--num_sample', type=int, default=10)
parser.add_argument('--len_frame', type=int, default=16)
parser.add_argument('--dataset', default='Penn_action', 
                    choices=['Penn_action','KTH','HAA49'],type=str)
parser.add_argument('--req_cluster',  type=int, default=500)

def concept_decovery(args):
    subsampling.Keypointset(args)
    data, result_gt = clustering.clustering(args)
    closest_sample_indices = labeling.labeling(args,data,result_gt)
    visualize.concept_visualize(args,data,result_gt,closest_sample_indices)
if __name__=='__main__':
    args = parser.parse_args()
    concept_decovery(args)