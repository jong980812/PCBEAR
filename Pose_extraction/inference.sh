#!/bin/bash
#SBATCH --job-name ucf101_extract_skeleton
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=30G
#SBATCH --time 4-00:00:0
#SBATCH --partition batch
#SBATCH -w vll3
#SBATCH -o /data/jongseo/project/PCBEAR/Pose_extraction/results/ucf101/ucf101_5.out
#SBATCH -e /data/jongseo/project/PCBEAR/Pose_extraction/results/ucf101/ucf101_5.err

export OPENCV_FFMPEG_CAPTURE_OPTIONS="loglevel;quiet"
python /data/jongseo/project/PCBEAR/Pose_extraction/inference_folder.py \
    --output-path "./" \
    --src_dir /local_datasets/ucf101/videos \
    --skeleton_video_dir /data/dataset/ucf101_skeleton/biggest_box_video \
    --skeleton_json_dir /data/dataset/ucf101_skeleton/biggest_box_json \
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
    --class_folders \
    --allowed_classes /data/jongseo/project/PCBEAR/dataset/UCF-101/anno_for_pose_extraction/ucf_class_list_5.txt




