#!/bin/bash
#SBATCH --job-name kth_extract_skeleton
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=50G
#SBATCH --time 4-00:00:0
#SBATCH --partition batch
#SBATCH -w vll3
#SBATCH -o biggest_box_video.out
#SBATCH -e biggest_box_video.err

export OPENCV_FFMPEG_CAPTURE_OPTIONS="loglevel;quiet"
python /data/jongseo/project/PCBEAR/Pose_extraction/inference_folder.py \
    --output-path "./" \
    --src_dir /local_datasets/haa500/video \
    --skeleton_video_dir /local_datasets/haa500/video/biggest_box_video \
    --skeleton_json_dir /local_datasets/haa500/video/biggest_box_json \
    --model "/data/jongseo/project/PCBEAR/Pose_extraction/vitpose-l-coco.pth" \
    --yolo "/data/jongseo/project/PCBEAR/Pose_extraction/yolov8l.pt" \
    --dataset "coco" \
    --det-class "human" \
    --model-name "l" \
    --yolo-size "320" \
    --conf-threshold "0.3" \
    --rotate "0" \
    --yolo-step "1" \
    --save-img \
    --save-json \
    --single-pose \
    --class_folders




