#!/bin/bash
#SBATCH --job-name Penn_extract_skeleton
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=50G
#SBATCH --time 4-00:00:0
#SBATCH --partition batch
#SBATCH -w augi3
#SBATCH -o penn_action_test.out
#SBATCH -e penn_action_test.err

export OPENCV_FFMPEG_CAPTURE_OPTIONS="loglevel;quiet"
python /data/jongseo/project/PCBEAR/Pose_extraction/inference_folder.py \
    --output-path "./" \
    --model "/data/jongseo/project/PCBEAR/Pose_extraction/vitpose-b-coco.pth" \
    --yolo "/data/jongseo/project/PCBEAR/Pose_extraction/yolov8l.pt" \
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


