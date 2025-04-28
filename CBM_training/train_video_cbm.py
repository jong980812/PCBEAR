from video_dataloader import datasets
from datetime import datetime
import torch
import os
import cbm
import random
import cbm_utils
import data_utils
import similarity
import argparse
import json
import numpy as np
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset
from video_dataloader import video_utils
import torch.distributed as dist
from learning_concept_layer import train_aggregated_classification_layer,train_classification_layer, train_cocept_layer, train_pose_cocept_layer
import debugging
from save_features import get_multi_modal_encoder, frozen_all_parameters
parser = argparse.ArgumentParser(description='Settings for creating CBM')
# parser.add_argument('--batch_size', default=64, type=int)
# parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--update_freq', default=1, type=int)
parser.add_argument('--save_ckpt_freq', default=100, type=int)

# Model parameters
parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                    help='Name of model to train')
parser.add_argument('--tubelet_size', type=int, default= 2)
parser.add_argument('--input_size', default=224, type=int,
                    help='videos input size')

parser.add_argument('--fc_drop_rate', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                    help='Attention dropout rate (default: 0.)')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')

parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
parser.add_argument('--model_ema', action='store_true', default=False)
parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

# Optimizer parameters
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
    weight decay. We use a cosine schedule for WD and using a larger decay by
    the end of training improves performance for ViTs.""")

parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--layer_decay', type=float, default=0.75)

parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                    help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

# Augmentation parameters
parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--num_sample', type=int, default=1,
                    help='Repeated_aug (default: 2)')
parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'),
parser.add_argument('--smoothing', type=float, default=0.0,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train_interpolation', type=str, default='bicubic',
                    help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

# Evaluation parameters
parser.add_argument('--crop_pct', type=float, default=None)
parser.add_argument('--short_side_size', type=int, default=224)
parser.add_argument('--test_num_segment', type=int, default=5)
parser.add_argument('--test_num_crop', type=int, default=3)

# Random Erase params
parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=0,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')

# Mixup params
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0.')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0.')
parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup_prob', type=float, default=0.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup_switch_prob', type=float, default=0.0,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup_mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

# Finetuning params
parser.add_argument('--finetune', default='', help='finetune from checkpoint')
parser.add_argument('--model_key', default='model|module', type=str)
parser.add_argument('--model_prefix', default='', type=str)
parser.add_argument('--init_scale', default=0.001, type=float)
parser.add_argument('--use_checkpoint', action='store_true')
parser.set_defaults(use_checkpoint=False)
parser.add_argument('--use_mean_pooling', action='store_true')
parser.set_defaults(use_mean_pooling=True)
parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

# Dataset parameters

parser.add_argument('--eval_data_path', default=None, type=str,
                    help='dataset path for evaluation')
parser.add_argument('--nb_classes', default=400, type=int,
                    help='number of the classification types')
parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
parser.add_argument('--num_segments', type=int, default= 1)
parser.add_argument('--num_frames', type=int, default= 16)
parser.add_argument('--sampling_rate', type=int, default= 4)
parser.add_argument('--data_set', default='Kinetics-400', choices=['kth','kth-5','kth-2','penn-action','haa100','kinetics100','kinetics400','kinetics400_scratch', 'mini-SSV2','SSV2', 'UCF101', 'HMDB51','image_folder','mimetics'],
                    type=str, help='dataset')
parser.add_argument('--output_dir', default='',
                    help='path where to save, empty for no saving')
parser.add_argument('--log_dir', default=None,
                    help='path where to tensorboard log')
# parser.add_argument('--device', default='cuda',
#                     help='device to use for training / testing')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--resume', default='',
                    help='resume from checkpoint')
parser.add_argument('--auto_resume', action='store_true')
parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
parser.set_defaults(auto_resume=True)

parser.add_argument('--save_ckpt', action='store_true')
parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
parser.set_defaults(save_ckpt=True)

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--dist_eval', action='store_true', default=False,
                    help='Enabling distributed evaluation')
parser.add_argument('--num_workers', default=10, type=int)
parser.add_argument('--pin_mem', action='store_true',
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
parser.set_defaults(pin_mem=True)

# distributed training parameters
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local-rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')

parser.add_argument('--enable_deepspeed', action='store_true', default=False)
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--spatial_concept_set", type=str, default=None, 
                    help="path to concept set name")
parser.add_argument("--temporal_concept_set", type=str, default=None, 
                    help="path to concept set name")
parser.add_argument("--place_concept_set", type=str, default=None, 
                    help="path to concept set name")
parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="Which CLIP model to use")

parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
parser.add_argument("--saga_batch_size", type=int, default=256, help="Batch size used when fitting final layer")
parser.add_argument("--proj_batch_size", type=int, default=50000, help="Batch size to use when learning projection layer")

parser.add_argument("--feature_layer", type=str, default='layer4', 
                    help="Which layer to collect activations from. Should be the name of second to last layer in the model")
parser.add_argument("--activation_dir", type=str, default='saved_activations', help="save location for backbone and CLIP activations")
parser.add_argument("--save_dir", type=str, default='saved_models', help="where to save trained models")
parser.add_argument("--clip_cutoff", type=float, default=0.25, help="concepts with smaller top5 clip activation will be deleted")
parser.add_argument("--proj_steps", type=int, default=1000, help="how many steps to train the projection layer for")
parser.add_argument("--interpretability_cutoff", type=float, default=0.45, help="concepts with smaller similarity to target concept will be deleted")
parser.add_argument("--lam", type=float, default=0.0007, help="Sparsity regularization parameter, higher->more sparse")
parser.add_argument("--n_iters", type=int, default=1000, help="How many iterations to run the final layer solver for")
parser.add_argument("--print", action='store_true', help="Print all concepts being deleted in this stage")
parser.add_argument('--data_path', default='data/video_annotation/ucf101', type=str,
                    help='dataset path')
parser.add_argument('--video_anno_path',type=str)
parser.add_argument('--center_frame',action='store_true')
parser.add_argument('--no_aug',type=bool,default=False)
parser.add_argument('--saved_features',action='store_true')
parser.add_argument('--dual_encoder', default='clip', choices=['clip', 'lavila', 'internvid','internvid_200m','internvid_10flt'],
                    type=str, help='dataset')
parser.add_argument('--dual_encoder_frames',type=int,default=16)
parser.add_argument('--lavila_ckpt',type=str,default=None)
parser.add_argument('--internvid_version',type=str,default='200m')
parser.add_argument('--only_s',action='store_true')
parser.add_argument('--multiview',action='store_true')
parser.add_argument('--pose_label',type=str,default=None)
parser.add_argument('--sp_clip',action='store_true')
parser.add_argument('--use_mlp',action='store_true')
parser.add_argument('--debug',default=None)
parser.add_argument('--loss_mode',default='concept',choices=['concept','sample','second','first_concept','first_sample'])
#!
#!
parser.add_argument('--backbone_features',type=str,default=None)
parser.add_argument('--learn_each_cls',action='store_true')
parser.add_argument('--with_cls_attr',action='store_true')
parser.add_argument('--vlm_features',type=str,default=None)
# parser.add_argument('--train_mode', default='pose',type=str, help='set concept type')
parser.add_argument('--train_mode', nargs='+', default=['pose'], choices=['pose', 'spatial', 'temporal', 'place'],
                    help='Concept types to train on')
#!
parser.add_argument('--no_cbm',action='store_true')
parser.add_argument('--dump_concept_num', type=int, default=None, help='Number of concepts to dump when no_cbm is set. Must be a positive integer.')
parser.add_argument('--mode_no_cbm', type=str, default='only_cls', choices=['only_cls', 'only_sparse_cls', 'dump_linear_cls', 'dump_linear_sparse_cls'],
                    help='Mode to run when no_cbm is enabled')
parser.add_argument('--no_cbm_epochs', type=int, default=30, help='Number of epochs for no_cbm mode')
parser.add_argument('--no_cbm_batchsize', type=int, default=256, help='Batch size for no_cbm mode')
parser.add_argument('--no_cbm_lr', type=float, default=1e-3, help='Learning rate for no_cbm mode')
parser.add_argument('--no_cbm_weight_decay', type=float, default=0.05, help='Weight decay for no_cbm mode')
def setup_seed(seed):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_dynamic_save_name(args):
    def strip_txt(path):
        if path is None:
            return None
        return os.path.splitext(os.path.basename(path))[0]

    concept_tags = []
    if 'pose' in args.train_mode:
        pose_tag = strip_txt(args.pose_label)
        concept_tags.append(f"{pose_tag}")
    if 'spatial' in args.train_mode:
        s_tag = strip_txt(args.spatial_concept_set)
        concept_tags.append(f"{s_tag}")
    if 'place' in args.train_mode:
        p_tag = strip_txt(args.place_concept_set)
        concept_tags.append(f"{p_tag}")
    if 'temporal' in args.train_mode:
        t_tag = strip_txt(args.temporal_concept_set)
        concept_tags.append(f"{t_tag}")

    tag_str = "+".join(concept_tags)
    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    return os.path.join(args.save_dir, f"{args.data_set}_{tag_str}_{timestamp}")

def train_cbm_and_save(args):
    video_utils.init_distributed_mode(args)
    setup_seed(args.seed)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    device = torch.device(args.device)
    
    #! To check consistency between backbone features and target_data. 
    assert args.data_set in args.backbone_features, f"Error: '{args.data_set}' not found in args.backbone_features ('{args.backbone_features}')"
    
    backbone_features = torch.load(args.backbone_features,map_location="cpu").float()
    val_backbone_features=torch.load(args.backbone_features.replace(f'{args.data_set}_train',f'{args.data_set}_val'),map_location="cpu").float()

    if args.no_cbm:
        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        save_name = os.path.join(args.save_dir, f"nocbm_{timestamp}_dump{args.dump_concept_num}")
        os.makedirs(save_name, exist_ok=True)
        from learning_dump_concept_layer import train_non_cbm
        train_non_cbm(args, backbone_features, val_backbone_features, save_name)
        return
    else:
        save_name = get_dynamic_save_name(args)
        os.makedirs(save_name, exist_ok=True)
    cbm_utils.save_args(args,save_name)

    # aggregated_train_c_features = []
    # aggregated_val_c_features = []
    


        

    
    concept_save_paths = {}
    concept_names = {}
    concept_matrix = {}
    val_concept_matrix = {}
    aggregated_concepts = []
    # aggregated_train_c_features = []
    # aggregated_val_c_features = []
    aggregated_W_c = []
    
    if 'pose' in args.train_mode:
        print("🧍‍♂️ Learning pose concepts...")
        pose_save_path = os.path.join(save_name, 'pose')
        os.makedirs(pose_save_path, exist_ok=True)
        pose_W_c, best_val_loss = train_pose_cocept_layer(args, backbone_features, val_backbone_features, pose_save_path)
        pose_concepts = [str(i) for i in range(pose_W_c.shape[0])]
        print(f'Pose concept num: {len(pose_concepts)}')
        if len(args.train_mode)<2 or args.learn_each_cls:
            pose_train_c, pose_val_c =  train_classification_layer(args,
                                W_c=pose_W_c,
                                pre_concepts=None,
                                concepts =pose_concepts ,
                                target_features=backbone_features,
                                val_target_features=val_backbone_features,
                                save_name=pose_save_path,
                                best_val_loss=best_val_loss
                                )
        
        aggregated_concepts.append(pose_concepts)
        aggregated_W_c.append(pose_W_c)
        # pose 학습 완료 후
        concepts_txt_path = os.path.join(pose_save_path, "concepts.txt")
        with open(concepts_txt_path, "w") as f:
            f.write(pose_concepts[0])
            for concept in pose_concepts[1:]:
                f.write('\n'+concept)
        if args.train_mode == 'pose':
            return
        
    # Define concept types
    # concept_types = {
    #     'spatial': 'spatial',
    #     'place': 'place',
    #     'temporal': 'temporal'
    # }

    # Filter active concept keys based on train_mode
    active_concepts = [m for m in args.train_mode if m != 'pose'] #[k for k in concept_types if k in args.train_mode]
    
    print("🚀 Loading dual encoder...")
    dual_encoder = get_multi_modal_encoder(args,device).to(device).eval()
    frozen_all_parameters(dual_encoder,args.dual_encoder)

    print("📦 Loading VLM visual features...")
    with torch.no_grad():
        vlm_features = torch.load(args.vlm_features, map_location="cpu").float()
        val_vlm_features = torch.load(args.vlm_features.replace(f'{args.data_set}_train', f'{args.data_set}_val'), map_location="cpu").float()
        vlm_features /= torch.norm(vlm_features, dim=1, keepdim=True)
        val_vlm_features /= torch.norm(val_vlm_features, dim=1, keepdim=True)


            



    # Load and process each active concept
    for key in active_concepts:
        set_path = getattr(args, f"{key}_concept_set")
        with open(set_path, 'r') as f:
            concept_names[key] = f.read().split('\n')

        print(f"📚 Encoding text features for {key} concepts...")
        concept_save_paths[key] = cbm_utils.save_text_features(set_path, args, dual_encoder)

    # Compute concept matrices
    with torch.no_grad():
        for key in active_concepts:
            text_feat = torch.load(concept_save_paths[key], map_location="cpu").float()
            text_feat /= torch.norm(text_feat, dim=1, keepdim=True)

            mat = vlm_features @ text_feat.T
            val_mat = val_vlm_features @ text_feat.T

            topk_mean = torch.mean(torch.topk(mat, dim=0, k=5)[0], dim=0)
            original_len = len(concept_names[key])
            concept_names[key] = [concept_names[key][i] for i in range(original_len) if topk_mean[i] > args.clip_cutoff]
            print(f"🔍 {key} concept: {original_len} -> {len(concept_names[key])}")

            mat = mat[:, topk_mean > args.clip_cutoff]
            val_mat = val_mat[:, topk_mean > args.clip_cutoff]

            concept_matrix[key] = mat
            val_concept_matrix[key] = val_mat



    # Textual concepts 학습
    for key in active_concepts:
        save_path = os.path.join(save_name, key)
        os.makedirs(save_path, exist_ok=True)

        print(f"🎓 Learning {key} concept classifier...")
        text_W_c, updated_concepts, best_val_loss = train_cocept_layer(
            args=args,
            concepts=concept_names[key],
            target_features=backbone_features,
            val_target_features=val_backbone_features,
            clip_feature=concept_matrix[key],
            val_clip_features=val_concept_matrix[key],
            save_name=save_path
        )
        if len(args.train_mode)<2 or args.learn_each_cls:
            train_c, val_c = train_classification_layer(
                args=args,
                W_c=text_W_c,
                pre_concepts=None,
                concepts=updated_concepts,
                target_features=backbone_features,
                val_target_features=val_backbone_features,
                save_name=save_path,
                joint=None,
                best_val_loss=best_val_loss
            )

        aggregated_concepts.append(updated_concepts)
        # aggregated_train_c_features.append(train_c)
        # aggregated_val_c_features.append(val_c)
        aggregated_W_c.append(text_W_c)
        
        
# Aggregated classification 학습
    if len(args.train_mode)<2: 
        return
    print("🧠 Training aggregated concept classifier...")
    train_aggregated_classification_layer(
        args=args,
        target_features=backbone_features,
        val_target_features=val_backbone_features,
        # aggregated_train_c_features=aggregated_train_c_features,
        # aggregated_val_c_features=aggregated_val_c_features,
        W_c = aggregated_W_c,
        concepts=aggregated_concepts,
        save_name=save_name
    )
    print("✅ Aggregated classification training complete.")

    
    
    
    
#     with torch.no_grad():
#     #! VLM Textual features
#         s_text_features = torch.load(s_concept_save_name, map_location="cpu").float()
#         s_text_features /= torch.norm(s_text_features, dim=1, keepdim=True)
#         p_text_features = torch.load(p_concept_save_name, map_location="cpu").float()
#         p_text_features /= torch.norm(p_text_features, dim=1, keepdim=True)
#     #! Concept matrix         
#         s_concept_matrix = vlm_features @ s_text_features.T
#         s_val_concept_matrix = val_vlm_features @ s_text_features.T
#         p_concept_matrix = vlm_features @ p_text_features.T
#         p_val_concept_matrix = val_vlm_features @ p_text_features.T
        

    
#     #filter concepts not activating highly
#     s_highest = torch.mean(torch.topk(s_concept_matrix, dim=0, k=5)[0], dim=0)
#     p_highest = torch.mean(torch.topk(p_concept_matrix, dim=0, k=5)[0], dim=0)


#     if args.print:
#         for i, concept in enumerate(s_concepts):
#             if s_highest[i]<=args.clip_cutoff:
#                 print("!**Spatial** Deleting {}, CLIP top5:{:.3f}".format(concept, s_highest[i]))
#         for i, concept in enumerate(p_concepts):
#             if p_highest[i]<=args.clip_cutoff:
#                 print("!**Place** Deleting {}, CLIP top5:{:.3f}".format(concept, p_highest[i]))
#     original_n_concept = len(s_concepts)
#     s_concepts = [s_concepts[i] for i in range(len(s_concepts)) if s_highest[i]>args.clip_cutoff]
#     print(f"!**Spatial** Num concept: {original_n_concept} -> {len(s_concepts)}")
#     original_p_concept = len(p_concepts)
#     p_concepts = [p_concepts[i] for i in range(len(p_concepts)) if p_highest[i]>args.clip_cutoff]
#     print(f"!**Place** Num concept: {original_p_concept} -> {len(p_concepts)}")


#     s_concept_matrix = s_concept_matrix[:, s_highest>args.clip_cutoff]
#     p_concept_matrix = p_concept_matrix[:, p_highest>args.clip_cutoff]
#     s_val_concept_matrix = s_val_concept_matrix[:, s_highest>args.clip_cutoff]
#     p_val_concept_matrix = p_val_concept_matrix[:, p_highest>args.clip_cutoff]
    

    
#     s_save_name = os.path.join(save_name,'spatial');os.makedirs(s_save_name,exist_ok=True)
#     p_save_name = os.path.join(save_name,'place');os.makedirs(p_save_name,exist_ok=True)

# #! Learning Spatial concepts
#     s_W_c, s_concepts, s_best_val_loss= train_cocept_layer(args=args,
#                                                             concepts=s_concepts,
#                                                             target_features=backbone_features,
#                                                             val_target_features=val_backbone_features,
#                                                             clip_feature=s_concept_matrix,
#                                                             val_clip_features=s_val_concept_matrix,
#                                                             save_name=s_save_name)
#     spatial_train_c,spatial_val_c=train_classification_layer(args=args,
#                                W_c=s_W_c,
#                                pre_concepts= None, 
#                                concepts=s_concepts,
#                                target_features=backbone_features,
#                                val_target_features=val_backbone_features,
#                                save_name=s_save_name,
#                                joint=None,
#                                best_val_loss=s_best_val_loss)
# #! Learning Place concepts
    
#     p_W_c, p_concepts, p_best_val_loss= train_cocept_layer(args=args,
#                                                             concepts=p_concepts,
#                                                             target_features=backbone_features,
#                                                             val_target_features=val_backbone_features,
#                                                             clip_feature=p_concept_matrix,
#                                                             val_clip_features=p_val_concept_matrix,
#                                                             save_name=p_save_name)
#     place_train_c,place_val_c=train_classification_layer(args=args,
#                                W_c=p_W_c,
#                                pre_concepts= None, 
#                                concepts=p_concepts,
#                                target_features=backbone_features,
#                                val_target_features=val_backbone_features,
#                                save_name=p_save_name,
#                                joint=None,
#                                best_val_loss=p_best_val_loss)
    
#     aggregated_concepts = [str(i) for i in range(pose_train_c.shape[1])]+s_concepts+p_concepts
#     train_c_features=torch.cat(aggregated_train_c_features,dim=1)
    
    
#     #! 얻어진 컨셉들은 interpretability cutoff적용된 새로운 것.

#     train_aggregated_classification_layer(args=args,
#                                           aggregated_train_c_features=aggregated_train_c_features)


if __name__=='__main__':
    args = parser.parse_args()
    start_time = datetime.now()
    train_cbm_and_save(args)
    end_time = datetime.now()
    print(f"🚀 Run time: {(end_time-start_time).total_seconds():.2f} seconds")  # 초 단위로 변환