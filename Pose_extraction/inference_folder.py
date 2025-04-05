from scipy.io import loadmat
import argparse
import json
import os
import time

from PIL import Image
import cv2
import numpy as np
import torch
import tqdm

from easy_ViTPose.vit_utils.inference import NumpyEncoder, VideoReader
from easy_ViTPose.inference import VitInference
from easy_ViTPose.vit_utils.visualization import joints_dict

try:
    import onnxruntime  # noqa: F401
    has_onnx = True
except ModuleNotFoundError:
    has_onnx = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, default='',
                        help='output path, if the path provided is a directory '
                        'output files are "input_name +_result{extension}".')
    parser.add_argument('--skeleton_json_dir', type=str, default='',
                        help='output path, if the path provided is a directory '
                        'output files are "input_name +_result{extension}".')
    parser.add_argument('--skeleton_video_dir', type=str, default='',
                        help='output path, if the path provided is a directory '
                        'output files are "input_name +_result{extension}".')
    parser.add_argument('--src_dir', type=str, default='',
                        help='output path, if the path provided is a directory '
                        'output files are "input_name +_result{extension}".')
    parser.add_argument('--model', type=str,
                        help='checkpoint path of the model')
    parser.add_argument('--yolo', type=str, required=False, default=None,
                        help='checkpoint path of the yolo model')
    parser.add_argument('--dataset', type=str, required=False, default=None,
                        help='Name of the dataset. If None it"s extracted from the file name. \
                              ["coco", "coco_25", "wholebody", "mpii", "ap10k", "apt36k", "aic"]')
    parser.add_argument('--det-class', type=str, required=False, default=None,
                        help='["human", "cat", "dog", "horse", "sheep", \
                               "cow", "elephant", "bear", "zebra", "giraffe", "animals"]')
    parser.add_argument('--model-name', type=str, required=False, choices=['s', 'b', 'l', 'h'],
                        help='[s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]')
    parser.add_argument('--yolo-size', type=int, required=False, default=320,
                        help='YOLOv8 image size during inference')
    parser.add_argument('--conf-threshold', type=float, required=False, default=0.5,
                        help='Minimum confidence for keypoints to be drawn. [0, 1] range')
    parser.add_argument('--rotate', type=int, choices=[0, 90, 180, 270],
                        required=False, default=0,
                        help='Rotate the image of [90, 180, 270] degress counterclockwise')
    parser.add_argument('--yolo-step', type=int,
                        required=False, default=1,
                        help='The tracker can be used to predict the bboxes instead of yolo for performance, '
                             'this flag specifies how often yolo is applied (e.g. 1 applies yolo every frame). '
                             'This does not have any effect when is_video is False')
    parser.add_argument('--single-pose', default=False, action='store_true',
                        help='Do not use SORT tracker because single pose is expected in the video')
    parser.add_argument('--show', default=False, action='store_true',
                        help='preview result during inference')
    parser.add_argument('--show-yolo', default=False, action='store_true',
                        help='draw yolo results')
    parser.add_argument('--show-raw-yolo', default=False, action='store_true',
                        help='draw yolo result before that SORT is applied for tracking'
                        ' (only valid during video inference)')
    parser.add_argument('--save-img', default=False, action='store_true',
                        help='save image results')
    parser.add_argument('--save-json', default=False, action='store_true',
                        help='save json results')
    parser.add_argument('--class_folders', default=False, action='store_true',
                        help='video 폴더안에 class_folder들이 따로 있는 경우')
    parser.add_argument('--allowed_classes', type=str, required=False, default=None,
                        help='checkpoint path of the yolo model')
    args = parser.parse_args()

    use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()


    # Load Yolo
    yolo = args.yolo
    if yolo is None:
        yolo = 'easy_ViTPose/' + ('yolov8s' + ('.onnx' if has_onnx and not (use_mps or use_cuda) else '.pt'))
    # input_path = args.input


    is_video = True
    ext = '.mp4' if is_video else '.png'
    assert not (args.save_img or args.save_json) or args.skeleton_json_dir, \
        'Specify an output path if using save-img or save-json flags'
    # output_path = args.output_path #! log 파일
    
    
    src_dir = args.src_dir#'/local_datasets/haa500/video'#! 비디오 폴더 
    # src_dir = '/data/jongseo/project/PCBEAR/Pose_extraction/test_video/370'
    output_path = args.skeleton_video_dir#'/data/dataset/haa500/large_skeleton_video'#! skeleton 합쳐진 영상 저장되는경로
    json_path = args.skeleton_json_dir#'/data/dataset/haa500/large_skeleton_json' #! skeleton 값들 json으로
    # output_path = '/data/jongseo/project/PCBEAR/Pose_extraction/test_video/370/skeleton_video'#! skeleton 합쳐진 영상 저장되는경로
    # json_path = '/data/jongseo/project/PCBEAR/Pose_extraction/test_video/370/skeleton_json' #! skeleton 값들 json으로
    model = VitInference(args.model, yolo, args.model_name,
                        args.det_class, args.dataset,
                        args.yolo_size, is_video=is_video,
                        single_pose=args.single_pose,
                        yolo_step=args.yolo_step)  # type: ignore
    print(f">>> Model loaded: {args.model}")
    
    if args.class_folders:
        if args.allowed_classes is not None:
            with open(args.allowed_classes, 'r') as f:
                class_names = set(line.strip() for line in f if line.strip())
        else:
            class_names = os.listdir(src_dir)
                # Initialize model

        for class_name in class_names:
            # if class_name not in allowed_classes:
            #     continue
            print(f"****************Class {class_name} ****************")
            class_folder = os.path.join(src_dir,class_name)
            os.makedirs(os.path.join(output_path,class_name), exist_ok=True)
            os.makedirs(os.path.join(json_path,class_name), exist_ok=True)
            input_files = [os.path.join(class_folder, f) for f in os.listdir(class_folder) if f.endswith(('.avi','.mp4', '.jpg', '.png'))]
            for (i,input_path) in enumerate(input_files):
                print(f"****************{i}/{len(input_files)}: {input_path} ****************")
                
                if os.path.isdir(output_path):
                    og_ext = input_path[input_path.rfind('.'):]
                    save_name_img = os.path.basename(input_path).replace(og_ext, f"_result{ext}")
                    save_name_json = os.path.basename(input_path).replace(og_ext, "_result.json")
                    output_path_img = os.path.join(output_path,class_name, save_name_img)
                    output_path_json = os.path.join(json_path, class_name, save_name_json)
                else:
                    output_path_img = output_path + f'{ext}'
                    output_path_json = output_path + '.json'

                wait = 0
                total_frames = 1
                if is_video:
                    reader = VideoReader(input_path, args.rotate)
                    cap = cv2.VideoCapture(input_path)  # type: ignore
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    wait = 15
                    if args.save_img:
                        cap = cv2.VideoCapture(input_path)  # type: ignore
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        ret, frame = cap.read()
                        cap.release()
                        assert ret
                        assert fps > 0
                        output_size = frame.shape[:2][::-1]

                        # Check if we have X264 otherwise use default MJPG
                        try:
                            temp_video = cv2.VideoWriter('/tmp/checkcodec.mp4',
                                                        cv2.VideoWriter_fourcc(*'h264'), 30, (32, 32))
                            opened = temp_video.isOpened()
                        except Exception:
                            opened = False
                        codec = 'h264' if opened else 'MJPG'
                        out_writer = cv2.VideoWriter(output_path_img,
                                                    cv2.VideoWriter_fourcc(*codec),  # More efficient codec
                                                    fps, output_size)  # type: ignore
                else:
                    reader = [np.array(Image.open(input_path).rotate(args.rotate))]  # type: ignore



                print(f'>>> Running inference on {input_path}')
                keypoints = []
                fps = []
                tot_time = 0.
                for (ith, img) in enumerate(reader):
                    t0 = time.time()

                    # Run inference
                    frame_keypoints = model.inference(img)
                    keypoints.append(frame_keypoints)

                    delta = time.time() - t0
                    tot_time += delta
                    fps.append(delta)

                    # Draw the poses and save the output img
                    if args.show or args.save_img:
                        # Draw result and transform to BGR
                        img = model.draw(args.show_yolo, args.show_raw_yolo, args.conf_threshold)[..., ::-1]

                        if args.save_img:
                            # TODO: If exists add (1), (2), ...
                            if is_video:
                                out_writer.write(img)
                            else:
                                print('>>> Saving output image')
                                cv2.imwrite(output_path_img, img)

                        if args.show:
                            cv2.imshow('preview', img)
                            cv2.waitKey(wait)

                if is_video:
                    tot_poses = sum(len(k) for k in keypoints)
                    print(f'>>> Mean inference FPS: {1 / np.mean(fps):.2f}')
                    print(f'>>> Total poses predicted: {tot_poses} mean per frame: '
                        f'{(tot_poses / (ith + 1)):.2f}')
                    print(f'>>> Mean FPS per pose: {(tot_poses / tot_time):.2f}')

                if args.save_json:
                    print('>>> Saving output json')
                    with open(output_path_json, 'w') as f:
                        out = {'keypoints': keypoints,
                            'skeleton': joints_dict()[model.dataset]['keypoints']}
                        json.dump(out, f, cls=NumpyEncoder)

                if is_video and args.save_img:
                    out_writer.release()
                cv2.destroyAllWindows()
    else:
            os.makedirs(output_path,exist_ok=True)
            os.makedirs(json_path,exist_ok=True)
            input_files = [ f for f in os.listdir(src_dir) if f.endswith(('.avi','.mp4', '.jpg', '.png'))]
            for (i,video_pth) in enumerate(input_files):
                input_path = os.path.join(src_dir,video_pth)
                video_id = video_pth.split('.')[0]
                print(f"****************{i}/{len(input_files)}: {video_pth} ****************")
                # mat_file_path = f'/data/dataset/Penn_Action_skeleton/Penn_Action/labels/{video_id}.mat'
                # # 파일 읽기
                # gt_bboxes = loadmat(mat_file_path)['bbox']
                if os.path.isdir(output_path):
                    og_ext = input_path[input_path.rfind('.'):]
                    save_name_img = os.path.basename(input_path).replace(og_ext, f"_result{ext}")
                    save_name_json = os.path.basename(input_path).replace(og_ext, "_result.json")
                    output_path_img = os.path.join(output_path, save_name_img)
                    output_path_json = os.path.join(json_path, save_name_json)
                else:
                    output_path_img = output_path + f'{ext}'
                    output_path_json = output_path + '.json'

                wait = 0
                total_frames = 1
                if is_video:
                    reader = VideoReader(input_path, args.rotate)
                    cap = cv2.VideoCapture(input_path)  # type: ignore
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    wait = 15
                    if args.save_img:
                        cap = cv2.VideoCapture(input_path)  # type: ignore
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        ret, frame = cap.read()
                        cap.release()
                        assert ret
                        assert fps > 0
                        output_size = frame.shape[:2][::-1]

                        # Check if we have X264 otherwise use default MJPG
                        try:
                            temp_video = cv2.VideoWriter('/tmp/checkcodec.mp4',
                                                        cv2.VideoWriter_fourcc(*'h264'), 30, (32, 32))
                            opened = temp_video.isOpened()
                        except Exception:
                            opened = False
                        codec = 'h264' if opened else 'MJPG'
                        out_writer = cv2.VideoWriter(output_path_img,
                                                    cv2.VideoWriter_fourcc(*codec),  # More efficient codec
                                                    fps, output_size)  # type: ignore
                else:
                    reader = [np.array(Image.open(input_path).rotate(args.rotate))]  # type: ignore



                print(f'>>> Running inference on {input_path}')
                keypoints = []
                fps = []
                tot_time = 0.
                # If gt_bboxes is shorter than total_frames, pad with last bbox
                # if gt_bboxes.shape[0] < total_frames:
                #     last_bbox = gt_bboxes[-1:]
                #     pad_count = total_frames - gt_bboxes.shape[0]
                #     gt_bboxes = np.concatenate([gt_bboxes, np.repeat(last_bbox, pad_count, axis=0)], axis=0)
                for (ith, img) in enumerate(reader):
                    t0 = time.time()

                    # Run inference
                    # frame_keypoints = model.inference(img,gt_bboxes[ith:ith+1,:])
                    frame_keypoints = model.inference(img)
                    keypoints.append(frame_keypoints)

                    delta = time.time() - t0
                    tot_time += delta
                    fps.append(delta)

                    # Draw the poses and save the output img
                    if args.show or args.save_img:
                        # Draw result and transform to BGR
                        img = model.draw(args.show_yolo, args.show_raw_yolo, args.conf_threshold)[..., ::-1]

                        if args.save_img:
                            # TODO: If exists add (1), (2), ...
                            if is_video:
                                out_writer.write(img)
                            else:
                                print('>>> Saving output image')
                                cv2.imwrite(output_path_img, img)

                        if args.show:
                            cv2.imshow('preview', img)
                            cv2.waitKey(wait)

                if is_video:
                    tot_poses = sum(len(k) for k in keypoints)
                    print(f'>>> Mean inference FPS: {1 / np.mean(fps):.2f}')
                    print(f'>>> Total poses predicted: {tot_poses} mean per frame: '
                        f'{(tot_poses / (ith + 1)):.2f}')
                    print(f'>>> Mean FPS per pose: {(tot_poses / tot_time):.2f}')

                if args.save_json:
                    print('>>> Saving output json')
                    with open(output_path_json, 'w') as f:
                        out = {'keypoints': keypoints,
                            'skeleton': joints_dict()[model.dataset]['keypoints']}
                        json.dump(out, f, cls=NumpyEncoder)

                if is_video and args.save_img:
                    out_writer.release()
                cv2.destroyAllWindows()