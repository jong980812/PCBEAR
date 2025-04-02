import os
import json
import numpy as np
import matplotlib.pyplot as plt
import labeling
import subsampling
import pandas as pd
import util

# Keypoint Skeleton Structure
skeleton = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
    [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [0, 5], [0, 6]
]

# COCO Keypoint Mapping
coco_keypoints = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
}

# Face Keypoints (Only Nose)
face_keypoints = [0]


def plot_pose(pose, ax=None, title="Pose Visualization", original_index=None, video_name=None, invert_y=True):
    pose = pose.reshape(17, 2)
    pose = pose[:, [1, 0]]  # Swap (y, x) to (x, y)
    if invert_y:
        pose[:, 1] = -pose[:, 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    # Draw Skeleton
    for joint1, joint2 in skeleton:
        p1, p2 = pose[joint1], pose[joint2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'bo-', linewidth=2, alpha=0.6)

    # Draw Keypoints
    body_keypoints = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    ax.scatter(pose[body_keypoints, 0], pose[body_keypoints, 1], c="red", marker="o", s=50, label="Body Joints")
    ax.scatter(pose[face_keypoints, 0], pose[face_keypoints, 1], c="yellow", edgecolors="black", marker="o", s=300, label="Nose")

    ax.set_xticks([])
    ax.set_yticks([])

def visualize_pose_by_index(args, original_index, processed_keypoints, sample_metadata, save_path, concept=-1):
    if original_index >= len(processed_keypoints):
        print(f"❌ ERROR: Index {original_index} out of range!")
        return
    
    input_pose = processed_keypoints[original_index]
    video_name = sample_metadata[original_index]
    video_id =  video_name.replace(".mp4", "")


    if args.dataset == "Penn_action":
        video_to_class = util.video_class_mapping(args)
        class_name = video_to_class.get(video_id, "Unknown Class")
        video_name = f"{video_name}_{class_name}"
    else :
        video_name = video_name
    T = input_pose.shape[0]
    frame_step = T//5
    selected_frames = list(range(0, T, frame_step))
    fig, axes_frames = plt.subplots(1, len(selected_frames), figsize=(len(selected_frames) * 3, 4))

    if len(selected_frames) == 1:
        axes_frames = [axes_frames]
    
    for i, t in enumerate(selected_frames):
        frame_pose = input_pose[t].flatten()
        plot_pose(frame_pose, ax=axes_frames[i], title=f"Frame {t}", video_name=video_name)
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, f'{concept}_{video_name}.jpg')
    plt.savefig(save_file)
    print(f"✅ Saved: {save_file}")
    plt.close(fig)

def concept_visualize(args,data,result_gt,closest_sample_indices, save_path):
    with open(os.path.join(save_path, "sample_metadata.json"), "r") as f:
        json_data = json.load(f)

    processed_keypoints = np.load(os.path.join(save_path, 'processed_keypoints.npy'))

    save_path = os.path.join(save_path, "concept")
    for concept, index in closest_sample_indices.items():
        print(f'Concept {concept}')
        visualize_pose_by_index(args, index, processed_keypoints, json_data,save_path , concept=concept)
