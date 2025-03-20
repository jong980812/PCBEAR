#!/bin/bash
#SBATCH --job-name Penn_extract_skeleton
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=50G
#SBATCH --time 4-00:00:0
#SBATCH --partition batch
#SBATCH -w vll3
#SBATCH -o /data/jongseo/project/easy_ViTPose/%A-%x.out
#SBATCH -e /data/jongseo/project/easy_ViTPose/%A-%x.err

export OPENCV_FFMPEG_CAPTURE_OPTIONS="loglevel;quiet"
python /data/jongseo/project/easy_ViTPose/inference_folder.py \
    --input "/data/dataset/kth/video/person09_running_d1_0.mp4" \
    --output-path "./" \
    --model "./vitpose-b-coco.pth" \
    --yolo "yolov8l.pt" \
    --dataset "coco" \
    --det-class "human" \
    --model-name "b" \
    --yolo-size "320" \
    --conf-threshold "0.3" \
    --rotate "0" \
    --yolo-step "1" \
    --save-img \
    --save-json \
    --single-pose \


