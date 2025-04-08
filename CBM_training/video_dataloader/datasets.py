import os
from torchvision import transforms
from transforms import *
from .kinetics import VideoClsDataset, VideoMAE
from .kinetics_scratch import VideoClsDataset_scratch, VideoMAE
from .ssv2 import SSVideoClsDataset
from .ucf import UCFVideoClsDataset
# from haa500 import HAA500VideoClsDataset


def build_dataset(is_train, test_mode, args):
    if args.data_set == 'kinetics400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'train.csv')
        elif test_mode is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1,# if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400
    elif args.data_set == 'kinetics400_scratch':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.video_anno_path, 'train.csv')
        elif test_mode is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv') 

        dataset = VideoClsDataset_scratch(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1,# if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400
    elif args.data_set == 'haa100':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'train.csv')
        elif test_mode is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 100
    elif args.data_set == 'kinetics100':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'train.csv')
        elif test_mode is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1,# if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 100

    elif args.data_set == 'kth':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'train.csv')
        elif test_mode is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 6
    elif args.data_set == 'kth-5':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'train.csv')
        elif test_mode is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 5
    elif args.data_set == 'kth-2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'train.csv')
        elif test_mode is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 2
    elif args.data_set == 'penn-action':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'train.csv')
        elif test_mode is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 15
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.video_anno_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174
    elif args.data_set == 'mini-SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.video_anno_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 87
    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'train.csv')
        elif test_mode is True:
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.video_anno_path, 'val.csv') 

        dataset = UCFVideoClsDataset(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
