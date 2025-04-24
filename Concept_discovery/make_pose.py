import json
import numpy as np
import matplotlib.pyplot as plt
# import imageio
import imageio.v2 as imageio
import os
import cv2
from matplotlib.cm import tab20
custom_colors = [
    "#BEAED4", "#BEAED4", "#7FC97F", "#7FC97F", "#CCCCCC",  # 팔, 다리, 몸통
    "#7FC97F", "#7FC97F", "#FDC086", "#FDC086", "#386CB0",  # 몸통 & 연결
    "#386CB0", "#386CB0", "#F0027F", "#F0027F"              # 얼굴 연결
]
# def draw_pose(ax, keypoints, bone_pairs, colors):
#     for i, (j1, j2) in enumerate(bone_pairs):
#         x1, y1 = keypoints[j1]
#         x2, y2 = keypoints[j2]
#         if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:
#             ax.plot([x1, x2], [y1, y2], color=colors[i % len(colors)], linewidth=2)

#     # 코 표시
#     nose_x, nose_y = keypoints[0]
#     if nose_x > 0 and nose_y > 0:
#         ax.scatter(nose_x, nose_y, c='red', s=20, zorder=5)
from matplotlib.patches import FancyArrowPatch

def draw_pose_2(ax, keypoints, bone_pairs, colors):
    # 팔다리: 두꺼운 선 (화살표처럼)
    for i, (j1, j2) in enumerate(bone_pairs):
        x1, y1 = keypoints[j1]
        x2, y2 = keypoints[j2]
        if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:
            arrow = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='-',
                linewidth=3,  # 부피감 표현
                color=colors[i % len(colors)],
                alpha=0.7,
                zorder=2
            )
            ax.add_patch(arrow)

    # 관절 점은 생략하고 코만 강조
    nose_x, nose_y = keypoints[0]
    if nose_x > 0 and nose_y > 0:
        ax.scatter(nose_x, nose_y, c='red', s=50, zorder=5)
# def draw_pose(ax, keypoints, bone_pairs, colors):
#     # 연결선 그리기
#     for i, (j1, j2) in enumerate(bone_pairs):
#         x1, y1 = keypoints[j1]
#         x2, y2 = keypoints[j2]
#         if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:
#             ax.plot([x1, x2], [y1, y2], color=colors[i % len(colors)], linewidth=2)

#     # 관절마다 점 찍기
#     for x, y in keypoints:
#         if x > 0 and y > 0:
#             ax.scatter(x, y, c='black', s=60, zorder=4)

#     # 코 (중앙 포인트) 강조
#     nose_x, nose_y = keypoints[0]
#     if nose_x > 0 and nose_y > 0:
#         ax.scatter(nose_x, nose_y, c='red', s=60, zorder=5)

def make_pose_video_cropped(json_path, start_frame, end_frame, save_dir='pose_output.mp4',scailing=[1.0,1.0]):
    with open(json_path, 'r') as f:
        data = json.load(f)
    save_name = os.path.basename(json_path).replace('json','mp4')
    save_path= os.path.join(save_dir,save_name)
    bone_pairs = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
        [7, 9], [8, 10], [0, 5], [0, 6]
    ]
    colors = custom_colors#tab20.colors
    keypoints_list = []

    # for frame in data['keypoints'][start_frame:end_frame]:
    #     if '0' in frame:
    #         keypoints = np.array(frame['0'])[:, :2]  # (17, 2)
    #         keypoints_list.append(keypoints[:,[1,0]])
    if len(data['keypoints'])<1:
        return
    for frame in data['keypoints'][start_frame:end_frame]:
        if '0' in frame:
            keypoints_full = np.array(frame['0'])  # (17, 3)
            conf = keypoints_full[:, 2]
            # mask = conf > 0.1

            keypoints = keypoints_full[:, :2]
            # keypoints[~mask] = -1  # 무시되게 처리
            keypoints_list.append(keypoints[:, [1, 0]])  # (y, x)로 변환
    scale_x,scale_y = scailing  # y축만 90%로 줄이기
    # 전체 keypoints로 bounding box 계산
    all_kpts = np.concatenate(keypoints_list, axis=0)
    x_min, y_min = np.min(all_kpts, axis=0) - 20
    x_max, y_max = np.max(all_kpts, axis=0) + 20
    width = int(x_max - x_min)
    height = int(y_max - y_min)

    for i in range(len(keypoints_list)):
        keypoints_list[i][:, 0] = (keypoints_list[i][:, 0] - x_min) * scale_x + x_min
        keypoints_list[i][:, 1] = (keypoints_list[i][:, 1] - y_min) * scale_y + y_min
    # for i in range(len(keypoints_list)):
    # 영상 저장 준비 (2fps = 0.5초 per frame)
    # writer = imageio.get_writer(save_path, fps=2, codec='libx264', quality=8)
    writer = imageio.get_writer(
    save_path,
    # format='FFMPEG',
    codec='libx264',
    fps=20  # <-- 이 줄이 핵심
)

    for keypoints in keypoints_list:
        fig, ax = plt.subplots(figsize=(width / 50, height / 50), dpi=500)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # y축 반전
        ax.axis('off')
        draw_pose_2(ax, keypoints, bone_pairs, colors)

        # 이미지 버퍼에서 읽어와서 frame 저장
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(frame)
        plt.close()

    writer.close()
    print(f"MP4 영상 저장 완료: {save_path}")

path = '/data/jongseo/project/PCBEAR/Concept_discovery/ucf_json/biggest_box_json/GolfSwing'
class_name = 'GolfSwing'
# 사용 예시
for video_path in os.listdir(path):
    make_pose_video_cropped(
        json_path=os.path.join(path,video_path),#'/data/jongseo/project/PCBEAR/Frame_confidence/ucf_json/biggest_box_json/TennisSwing/v_TennisSwing_g09_c01_result.json',
        start_frame=0,
        end_frame=100,
        save_dir=f'/data/jongseo/project/PCBEAR/Concept_discovery/pose_gif/{class_name}',
        scailing=[1.0,1.0]
    )