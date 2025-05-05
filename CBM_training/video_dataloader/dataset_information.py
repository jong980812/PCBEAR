# dataset_information.py

DATASET_CONFIG = {
    'UCF101_SCUBA': {
        'data_set': 'UCF101_SCUBA',
        'data_path': '/local_datasets/UCF101_place365',
        'video_anno_path': '/data/jongseo/project/PCBEAR/dataset/UCF_scuba'
    },
    'UCF101_VQGAN': {
        'data_set': 'UCF101_SCUBA',
        'data_path': '/local_datasets/UCF101_vqgan',
        'video_anno_path': '/data/jongseo/project/PCBEAR/dataset/UCF_scuba'
    },
    'HAT': {
        'data_set': 'HAT',
        'data_path': '/local_datasets/HAT_dataset',
        'video_anno_path': '/data/jongseo/project/PCBEAR/dataset/HAT_annotations'
    # 필요 시 다른 데이터셋 추가
}
}