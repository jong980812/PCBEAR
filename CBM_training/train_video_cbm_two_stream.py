import torch
import os
import cbm
import random
import cbm_utils
import data_utils
import similarity
import argparse
import datetime
import json
import numpy as np
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset
from video_dataloader import video_utils
import torch.distributed as dist
from learning_concept_layer import spatio_temporal_parallel,spatio_temporal_serial, spatio_temporal_attention,spatio_temporal_joint,spatio_temporal_three_joint,hard_label,soft_label, spatio_temporal_single
import debugging
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
parser.add_argument('--data_set', default='Kinetics-400', choices=['kth','kth-5','kth-2','penn-action','haa49','kinetics100','kinetics400','kinetics400_scratch', 'mini-SSV2','SSV2', 'UCF101', 'HMDB51','image_folder'],
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
parser.add_argument("--s_concept_set", type=str, default=None, 
                    help="path to concept set name")
parser.add_argument("--t_concept_set", type=str, default=None, 
                    help="path to concept set name")
parser.add_argument("--p_concept_set", type=str, default=None, 
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
parser.add_argument('--train_mode',type=str,default='para')
parser.add_argument('--internvid_version',type=str,default='200m')
parser.add_argument('--only_s',action='store_true')
parser.add_argument('--multiview',action='store_true')
parser.add_argument('--hard_label',type=str,default=None)
parser.add_argument('--sp_clip',action='store_true')
parser.add_argument('--use_mlp',action='store_true')
parser.add_argument('--debug',default=None)
parser.add_argument('--loss_mode',default='concept',choices=['concept','sample','second','first_concept','first_sample'])


def train_cbm_and_save(args):
    video_utils.init_distributed_mode(args)
    
    seed = args.seed
    random.seed(seed)  # Python random seed 설정
    np.random.seed(seed)  # NumPy random seed 설정
    torch.manual_seed(seed)  # PyTorch random seed 설정 (CPU)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch random seed 설정 (CUDA)
        torch.cuda.manual_seed_all(seed)  # 모든 GPU에 적용
        
    # cuDNN의 비결정적 동작을 방지 (성능에 약간의 영향을 줄 수 있음)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    # args.video_anno_path=os.path.join(os.getcwd(),'data/video_annotation',args.data_set)
    # similarity_fn = similarity.cos_similarity_cubed_single
    device = torch.device(args.device)
    
    
    
    d_train = args.data_set + "_train"
    d_val = args.data_set + "_val"
    #!저장되는 feature이름이 데이터셋이랑 모델이름 조합.
    #! 만약에 kth-5, kth-2
    #get concept set
    cls_file = os.path.join(args.video_anno_path, 'class_list.txt')
    # with open(cls_file, "r") as f:
    #     classes = f.read().split("\n")
    
    with open(args.s_concept_set) as f:
        s_concepts = f.read().split("\n")
        # s_concepts = list(set(s_concepts))
    with open(args.t_concept_set) as f:
        t_concepts = f.read().split("\n")
        # t_concepts = list(set(t_concepts))
    with open(args.p_concept_set) as f:
        p_concepts = f.read().split("\n")
        # p_concepts = list(set(p_concepts))
    if args.debug is not None:
        debugging.debug(args,args.debug)
        return
        
    #save activations and get save_paths
    for d_probe in [d_train, d_val]:
        cbm_utils.save_activations(clip_name = args.dual_encoder, target_name = args.backbone, 
                               target_layers = [args.feature_layer], d_probe = d_probe,
                               concept_set = (args.s_concept_set, args.t_concept_set,args.p_concept_set), batch_size = args.batch_size, 
                               device =device, pool_mode = "avg", save_dir = args.activation_dir,
                               args=args)
        
    target_save_name, clip_save_name, s_text_save_name, t_text_save_name,p_text_save_name = cbm_utils.get_save_names(args.dual_encoder, args.backbone, 
                                            args.feature_layer,d_train, (args.s_concept_set, args.t_concept_set, args.p_concept_set), "avg", args.activation_dir)
    val_target_save_name, val_clip_save_name, s_text_save_name, t_text_save_name,_ =  cbm_utils.get_save_names(args.dual_encoder, args.backbone,
                                            args.feature_layer, d_val, (args.s_concept_set, args.t_concept_set,args.p_concept_set), "avg", args.activation_dir)
    
    feature_storage = '/data/dataset/videocbm/features'
    if args.saved_features:
        target_save_name = os.path.join(feature_storage,args.data_set,args.backbone,target_save_name.split('/')[-1])
        val_target_save_name = os.path.join(feature_storage,args.data_set,args.backbone,val_target_save_name.split('/')[-1])
        clip_save_name = os.path.join(feature_storage,args.data_set,args.dual_encoder,clip_save_name.split('/')[-1])
        val_clip_save_name = os.path.join(feature_storage,args.data_set,args.dual_encoder,val_clip_save_name.split('/')[-1])
        if args.sp_clip:
            sp_clip_save_name = '/data/jongseo/project/Video-CBM-two-stream/new_results/clip_features/activation/kinetics400_train_clip.pt'
            sp_val_clip_save_name = '/data/jongseo/project/Video-CBM-two-stream/new_results/clip_features/activation/kinetics400_val_clip.pt'
    if args.multiview:
        target_save_name = os.path.join(feature_storage,args.data_set,args.backbone,target_save_name.split('/')[-1]).replace('.pt','_view4_concat.pt')
        # val_target_save_name = os.path.join(feature_storage,args.data_set,args.backbone,val_target_save_name.split('/')[-1]).replace('.pt','_view4_concat.pt')
        clip_save_name = os.path.join(feature_storage,args.data_set,args.dual_encoder,clip_save_name.split('/')[-1]).replace('.pt','_view4_concat.pt')
        # val_clip_save_name = os.path.join(feature_storage,args.data_set,args.dual_encoder,val_clip_save_name.split('/')[-1]).replace('.pt','_view4_concat.pt')
    
    #load features
    with torch.no_grad():
        if args.sp_clip:
            target_features = torch.load(target_save_name, map_location="cpu").float()
            
            val_target_features = torch.load(val_target_save_name, map_location="cpu").float()
        
        #! Internvid
            image_features = torch.load(clip_save_name, map_location="cpu").float()#B,T,D
            image_features /= torch.norm(image_features, dim=1, keepdim=True)
            val_image_features = torch.load(val_clip_save_name, map_location="cpu").float()
            val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)
        #! CLIP
            sp_image_features = torch.load(sp_clip_save_name, map_location="cpu").float()#B,T,D
            sp_image_features /= torch.norm(sp_image_features, dim=2, keepdim=True)
            sp_val_image_features = torch.load(sp_val_clip_save_name, map_location="cpu").float()
            sp_val_image_features /= torch.norm(sp_val_image_features, dim=2, keepdim=True)

            s_text_features = torch.load(s_text_save_name, map_location="cpu").float()
            s_text_features /= torch.norm(s_text_features, dim=1, keepdim=True)
            
            t_text_features = torch.load(t_text_save_name, map_location="cpu").float()
            t_text_features /= torch.norm(t_text_features, dim=1, keepdim=True)
            
            p_text_features = torch.load(p_text_save_name, map_location="cpu").float()
            p_text_features /= torch.norm(p_text_features, dim=1, keepdim=True)
            
            s_clip_features = (sp_image_features @ s_text_features.T).mean(1)
            s_val_clip_features = (sp_val_image_features @ s_text_features.T).mean(1)
            t_clip_features = image_features @ t_text_features.T
            t_val_clip_features = val_image_features @ t_text_features.T
            p_clip_features = (sp_image_features @ p_text_features.T).mean(1)
            p_val_clip_features = (sp_val_image_features @ p_text_features.T).mean(1)
        else:
            target_features = torch.load(target_save_name, map_location="cpu").float()
            
            val_target_features = torch.load(val_target_save_name, map_location="cpu").float()
        
            image_features = torch.load(clip_save_name, map_location="cpu").float()
            image_features /= torch.norm(image_features, dim=1, keepdim=True)
            

            val_image_features = torch.load(val_clip_save_name, map_location="cpu").float()
            val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)

            s_text_features = torch.load(s_text_save_name, map_location="cpu").float()
            s_text_features /= torch.norm(s_text_features, dim=1, keepdim=True)
            
            t_text_features = torch.load(t_text_save_name, map_location="cpu").float()
            t_text_features /= torch.norm(t_text_features, dim=1, keepdim=True)
            
            p_text_features = torch.load(p_text_save_name, map_location="cpu").float()
            p_text_features /= torch.norm(p_text_features, dim=1, keepdim=True)
            
            s_clip_features = image_features @ s_text_features.T
            s_val_clip_features = val_image_features @ s_text_features.T
            t_clip_features = image_features @ t_text_features.T
            t_val_clip_features = val_image_features @ t_text_features.T
            p_clip_features = image_features @ p_text_features.T
            p_val_clip_features = val_image_features @ p_text_features.T
        
        del image_features, s_text_features, t_text_features,p_text_features, val_image_features
    
    #filter concepts not activating highly
    s_highest = torch.mean(torch.topk(s_clip_features, dim=0, k=5)[0], dim=0)
    t_highest = torch.mean(torch.topk(t_clip_features, dim=0, k=5)[0], dim=0)
    p_highest = torch.mean(torch.topk(p_clip_features, dim=0, k=5)[0], dim=0)


    if args.print:
        for i, concept in enumerate(s_concepts):
            if s_highest[i]<=args.clip_cutoff:
                print("!**Spatial** Deleting {}, CLIP top5:{:.3f}".format(concept, s_highest[i]))

    original_n_concept = len(s_concepts)
    s_concepts = [s_concepts[i] for i in range(len(s_concepts)) if s_highest[i]>args.clip_cutoff]
    print(f"!**Spatial** Num concept {len(s_concepts)} from {original_n_concept}")
    if args.print:
        for i, concept in enumerate(t_concepts):
            if t_highest[i]<=args.clip_cutoff:
                print("?**Temporal** Deleting {}, CLIP top5:{:.3f}".format(concept, t_highest[i]))

    original_n_concept = len(t_concepts)
    t_concepts = [t_concepts[i] for i in range(len(t_concepts)) if t_highest[i]>args.clip_cutoff]
    print(f"?**Temporal** Num concept {len(t_concepts)} from {original_n_concept}")
    
    if args.print:
        for i, concept in enumerate(p_concepts):
            if p_highest[i]<=args.clip_cutoff:
                print("!**Place** Deleting {}, CLIP top5:{:.3f}".format(concept, p_highest[i]))

    original_n_concept = len(p_concepts)
    p_concepts = [p_concepts[i] for i in range(len(p_concepts)) if p_highest[i]>args.clip_cutoff]
    print(f"!**Place** Num concept {len(p_concepts)} from {original_n_concept}")
    
    #save memory by recalculating
    del s_clip_features, t_clip_features, p_clip_features
    with torch.no_grad():
        if args.sp_clip:
            image_features = torch.load(clip_save_name, map_location="cpu").float()
            image_features /= torch.norm(image_features, dim=1, keepdim=True)
            sp_image_features = torch.load(sp_clip_save_name, map_location="cpu").float()
            sp_image_features /= torch.norm(sp_image_features, dim=2, keepdim=True)

            s_text_features = torch.load(s_text_save_name, map_location="cpu").float()[s_highest>args.clip_cutoff]
            s_text_features /= torch.norm(s_text_features, dim=1, keepdim=True)
            
            t_text_features = torch.load(t_text_save_name, map_location="cpu").float()[t_highest>args.clip_cutoff]
            t_text_features /= torch.norm(t_text_features, dim=1, keepdim=True)
            
            p_text_features = torch.load(p_text_save_name, map_location="cpu").float()[p_highest>args.clip_cutoff]
            p_text_features /= torch.norm(p_text_features, dim=1, keepdim=True)
            
            s_clip_features = (sp_image_features @ s_text_features.T).mean(1)
            t_clip_features = image_features @ t_text_features.T
            p_clip_features = (sp_image_features @ p_text_features.T).mean(1)
        else:
            image_features = torch.load(clip_save_name, map_location="cpu").float()
            image_features /= torch.norm(image_features, dim=1, keepdim=True)

            s_text_features = torch.load(s_text_save_name, map_location="cpu").float()[s_highest>args.clip_cutoff]
            s_text_features /= torch.norm(s_text_features, dim=1, keepdim=True)
            
            t_text_features = torch.load(t_text_save_name, map_location="cpu").float()[t_highest>args.clip_cutoff]
            t_text_features /= torch.norm(t_text_features, dim=1, keepdim=True)
            
            p_text_features = torch.load(p_text_save_name, map_location="cpu").float()[p_highest>args.clip_cutoff]
            p_text_features /= torch.norm(p_text_features, dim=1, keepdim=True)
            
            s_clip_features = image_features @ s_text_features.T
            t_clip_features = image_features @ t_text_features.T
            p_clip_features = image_features @ p_text_features.T
        del image_features, s_text_features,t_text_features,p_text_features
    
    s_val_clip_features = s_val_clip_features[:, s_highest>args.clip_cutoff]
    t_val_clip_features = t_val_clip_features[:, t_highest>args.clip_cutoff]
    p_val_clip_features = p_val_clip_features[:, p_highest>args.clip_cutoff]
    #! Learning Concept Layer
    #learn projection layer
    from datetime import datetime
    if args.pose_label:
        save_name = "{}/{}_cbm_{}{}".format(args.save_dir, args.data_set, args.pose_label.split("/")[-1],datetime.now().strftime("%M%S")) 
    else :
        save_name = "{}/{}_cbm_{}".format(args.save_dir, args.data_set,args.s_concept_set.split("/")[-1])
    os.makedirs(save_name,exist_ok=True)
    # save_spatial = os.path.join(save_name,'spatial')
    # save_temporal = os.path.join(save_name,'temporal')

    
    #!
    #target_feat -> Backbone feat
    #clip_feat -> dual encoder feat
    #!
    if args.train_mode=='serial':
        spatio_temporal_serial(args,
                            s_concepts,
                            target_features,
                            val_target_features,
                            s_clip_features,
                            s_val_clip_features,
                            t_concepts,
                            t_clip_features,
                            t_val_clip_features,
                            save_name)
    elif args.train_mode=='single':
        spatio_temporal_single(args,
                            s_concepts,
                            target_features,
                            val_target_features,
                            s_clip_features,
                            s_val_clip_features,
                            save_name)
    elif args.train_mode =='attention':
        # spatio_temporal_attention()
        spatio_temporal_attention(args,
                            s_concepts,
                            target_features,
                            val_target_features,
                            s_clip_features,
                            s_val_clip_features,
                            t_concepts,
                            t_clip_features,
                            t_val_clip_features,
                            save_name)
    elif args.train_mode =='joint':
        spatio_temporal_joint(args,
                            s_concepts,
                            target_features,
                            val_target_features,
                            s_clip_features,
                            s_val_clip_features,
                            t_concepts,
                            t_clip_features,
                            t_val_clip_features,
                            save_name)
    elif args.train_mode =='triple':
        spatio_temporal_three_joint(
                            args,
                            s_concepts,
                            target_features,
                            val_target_features,
                            s_clip_features,
                            s_val_clip_features,
                            t_concepts,
                            t_clip_features,
                            t_val_clip_features,
                            p_concepts,
                            p_clip_features,
                            p_val_clip_features,
                            save_name
        )
    elif args.train_mode =='para':
        spatio_temporal_parallel(args,
                            s_concepts,
                            target_features,
                            val_target_features,
                            s_clip_features,
                            s_val_clip_features,
                            t_concepts,
                            t_clip_features,
                            t_val_clip_features,
                            save_name)
    elif args.train_mode == 'hard_label':
        hard_label(args,target_features,val_target_features,save_name)
    else :
        soft_label(args,target_features,val_target_features,save_name)
    # s_W_c,s_concepts = train_cocept_layer(args,
    #                            s_concepts,
    #                            target_features,
    #                            val_target_features,
    #                            s_clip_features,
    #                            s_val_clip_features,
    #                            save_spatial)
    # t_W_c,t_concepts = train_cocept_layer(args,
    #                            t_concepts,
    #                            target_features,
    #                            val_target_features,
    #                            t_clip_features,
    #                            t_val_clip_features,
    #                            save_temporal)

    
    # del s_clip_features, s_val_clip_features,t_clip_features, t_val_clip_featuresㅌㅌㅌㅌㅌㅌ
    



    # train_classification_layer(args,
    #                            W_c=s_W_c,
    #                            concepts = s_concepts,
    #                            target_features=target_features,
    #                            val_target_features=val_target_features,
    #                             save_name=save_spatial
    #                            )
    # train_classification_layer(args,
    #                            W_c=t_W_c,
    #                            concepts = t_concepts,
    #                            target_features=target_features,
    #                            val_target_features=val_target_features,
    #                             save_name=save_temporal
    #                            )

    
    
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # val_data_t = data_utils.get_data(d_val,args=args)
    # s_model,t_model = cbm.load_cbm_two_stream(save_name, device,args)
    # cbm_utils.analysis_backbone_dim(s_model,val_data_t,args,5,300)
    # cbm_utils.analysis_backbone_dim(t_model,val_data_t,args,5,300)


    # accuracy = cbm_utils.get_accuracy_cbm(s_model, val_data_t, device,32,8)
    # print("?****? Spatio Accuracy: {:.2f}%".format(accuracy*100))
    
    # accuracy = cbm_utils.get_accuracy_cbm(t_model, val_data_t, device,32,8)
    # print("!****! Temporal Accuracy: {:.2f}%".format(accuracy*100))

if __name__=='__main__':
    args = parser.parse_args()
    train_cbm_and_save(args)