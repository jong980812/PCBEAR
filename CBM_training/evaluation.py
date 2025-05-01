import argparse
import sys
from datetime import datetime
import os
import json
import torch
import os
import random
import cbm
import plots
import random
import numpy as np
from video_dataloader import video_utils
import train_video_cbm
from video_dataloader import datasets
import cbm_utils

def load_all_concepts_by_type(load_dir, train_mode):
    concept_dict = {}
    for concept_type in train_mode:
        concept_path = os.path.join(load_dir, concept_type, "concepts.txt")
        if not os.path.exists(concept_path):
            print(f"⚠️ Warning: concepts.txt not found for {concept_type}")
            continue
        with open(concept_path, 'r') as f:
            concepts = f.read().splitlines()
        concept_dict[concept_type] = concepts
    return concept_dict


parser = argparse.ArgumentParser(description='Evaluatin ours')
parser.add_argument('--load_dir',required=True,type=str, default='', help='Trained model path')
parser.add_argument('--batch_size',type=int, default=32, help='Inference batch size')
parser.add_argument('--subset_labels', nargs='*', default=None, help='List of class labels to evaluate subset metrics')
def main(train_args, test_args):
    # video_utils.init_distributed_mode(test_args)
    train_video_cbm.setup_seed(train_args.seed)
    # save_dir = test_args.load_dir #! 같은 폴더에 저장
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    load_dir=test_args.load_dir
    
    import numpy as np

    inference_dir = os.path.join(load_dir, "inference")
    os.makedirs(inference_dir, exist_ok=True)

    # 체크포인트가 존재하면 load
    preds_path = os.path.join(inference_dir, "preds.npy")
    labels_path = os.path.join(inference_dir, "labels.npy")
    report_path = os.path.join(inference_dir, "report.txt")
    class_acc_path = os.path.join(inference_dir, "class_acc.txt")
    model = cbm.load_cbm_dynamic(load_dir, device,train_args)
    cls_file = os.path.join(train_args.video_anno_path, 'class_list.txt')
    with open(cls_file, "r") as f:
        class_names = f.read().split("\n")
    if os.path.exists(preds_path) and os.path.exists(labels_path) and os.path.exists(report_path):
        print("✅ Loading cached inference results...")
        all_preds = np.load(preds_path)
        all_labels = np.load(labels_path)
        with open(report_path, "r") as f:
            report = f.read()
    else:
        # inference 수행
        val_video_dataset,_ =   datasets.build_dataset(False, False, train_args)
        # accuracy = cbm_utils.get_accuracy_cbm(model, val_video_dataset, device,test_args.batch_size,8)
        report, cm, all_labels, all_preds,class_acc = cbm_utils.get_detailed_metrics_cbm(model, val_video_dataset, device, batch_size=test_args.batch_size, class_names=class_names,return_raw=True)
            # 저장
        with open(class_acc_path, "w") as f:
            for class_name, acc in sorted(class_acc.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{class_name}: {acc:.4f}\n")
        print(f"✅ 클래스별 accuracy 저장 완료: {save_path}")
        np.save(preds_path, np.array(all_preds))
        np.save(labels_path, np.array(all_labels))
        with open(report_path, "w") as f:
            f.write(report)
        print("📄 Inference 결과 저장 완료!")
    
    


    # COncept 정보 print
    concept_dict = load_all_concepts_by_type(load_dir, train_args.train_mode)
    if len(train_args.train_mode)<2:
        '''
        Single의 경우 aggregated 이제 안하고, 해당 컨셉 폴더만 만들어짐.
        '''
        concepts = concept_dict[train_args.train_mode[0]]
        if 'class_attributes' in train_args.pose_label.split('/')[-1]:
            '''UCF101 class attributes의 경우 Concpet name 따로 불러와야함. '''
            with open('/data/jongseo/project/PCBEAR/dataset/UCF101/class_attributes/attribute.txt', 'r') as f:
                concepts = f.read().splitlines()
    else:
        ''' train mode가 2개 이상일 경우 aggregated가 무조건 생기기때문에 aggregated로. '''
        with open(os.path.join(load_dir, "aggregated", "concepts.txt"), 'r') as f:
            concepts = f.read().splitlines()
    assert len(concepts) ==model.final.weight.shape[1], f"miss"
    print(f"Concept number: {len(concepts)}")


    if test_args.subset_labels:
        result = cbm_utils.get_off_diagonal_confusion_rate(
            all_labels, all_preds, class_names, test_args.subset_labels
        )
        
        # 저장 경로
        save_path = os.path.join(load_dir, f"off_diag_{'+'.join(test_args.subset_labels)}.txt")
        with open(save_path, 'w') as f:
            f.write(f"Target Labels: {result['subset_labels']}\n")
            f.write(f"Total samples: {result['total']}\n")
            f.write(f"Off-diagonal sum: {result['off_diagonal_sum']}\n")
            f.write(f"Confusion rate: {result['confusion_rate']:.4f}\n")
            f.write("Confusion matrix:\n")
            f.write(np.array2string(result['confusion_matrix'], separator=', '))
        print(f"✅ Saved off-diagonal analysis to {save_path}")
        # If subset labels are given, calculate subset metrics
    if test_args.subset_labels:

        cm, report = cbm_utils.get_class_subset_confusion(all_labels, all_preds, class_names, test_args.subset_labels)
        label_str = '+'.join(test_args.subset_labels)
        # filename = f"confusion_subset_{label_str}.txt"
        report_path=os.path.join(load_dir, f"classification_report_{label_str}.txt")
    else:
        report_path = os.path.join(load_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"📄 Saved classification report to {report_path}")
    


if __name__=='__main__':
    test_args = parser.parse_args()
    start_time = datetime.now()
    # 저장된 경로
    args_dir = os.path.join(test_args.load_dir, "args.txt")
    # json 파일 불러오기
    with open(args_dir, 'r') as f:
        args_dict = json.load(f)

    # dict → argparse.Namespace로 변환
    train_args = argparse.Namespace(**args_dict)
    
    main(train_args,test_args)
    end_time = datetime.now()
    print(f"🚀 Run time: {(end_time-start_time).total_seconds():.2f} seconds")  # 초 단위로 변환
