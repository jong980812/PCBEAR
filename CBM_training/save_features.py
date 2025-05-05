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
from cbm_utils import *
from video_dataloader import datasets
from modeling_clip import CLIP
parser = argparse.ArgumentParser(description='Settings for creating CBM')

# Model parameters
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
parser.add_argument('--data_set', default='Kinetics-400', choices=['kth','kth-5','kth-2','penn-action','haa100','kinetics100','kinetics400','kinetics400_scratch', 'mini-SSV2','SSV2', 'UCF101', 'HMDB51','image_folder'],
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
parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="Which CLIP model to use")
parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
parser.add_argument("--activation_dir", type=str, default='saved_activations', help="save location for backbone and CLIP activations")
parser.add_argument("--save_dir", type=str, default='saved_models', help="where to save trained models")
parser.add_argument('--data_path', default='data/video_annotation/ucf101', type=str,
                    help='dataset path')
parser.add_argument('--video_anno_path',type=str)
parser.add_argument('--center_frame',action='store_true')
parser.add_argument('--no_aug',action='store_true')
# parser.add_argument('--no_aug',type=bool,default=False)
parser.add_argument('--dual_encoder', default='clip', choices=['clip', 'lavila', 'internvid','internvid_200m','internvid_10flt'],
                    type=str, help='dataset')
parser.add_argument('--dual_encoder_frames',type=int,default=16)
parser.add_argument('--lavila_ckpt',type=str,default=None)
parser.add_argument('--internvid_version',type=str,default='200m')
parser.add_argument('--only_backbone',action='store_true')
parser.add_argument('--only_vlm',action='store_true')
parser.add_argument('--TA', action='store_true', default=False)



def frozen_all_parameters(model, name="model"):
    
    for param in model.parameters():
        param.requires_grad = False

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] Total params: {total}, Trainable params: {trainable}")
    if trainable == 0:
        print(f"✅ {name} is properly frozen.")
    else:
        print(f"❌ {name} is NOT frozen. ({trainable} trainable parameters)")

def get_multi_modal_encoder(args,device):
    if args.dual_encoder == "lavila": 
        #META Video-text model
        dual_encoder_model, _ = get_lavila(args,device=device) 
    elif args.dual_encoder == "clip":
        #OpenAI 나온 image-text model
        name = 'ViT-B/16'
        dual_encoder_model, _ = clip.load(name, device=device)
    elif "internvid" in args.dual_encoder:
        #MGC lab Video-text model
        dual_encoder_model, _ = get_intervid(args,device)
        # name = 'ViT-B/16'
        # clip_model, _ = clip.load(name, device=device)
    return dual_encoder_model

def get_video_encoder(args,device):
    if args.backbone.startswith("clip"):
        # target_model, target_preprocess = clip.load(args.backbone[5:], device=device)
        target_model=CLIP(
            input_resolution=224,
            patch_size=16,
            num_frames=args.num_frames,
            width=768,
            layers=12,
            heads=12,
            drop_path_rate=args.drop_path,
            num_classes=args.nb_classes,
            args=args
        )
        finetune = torch.load(args.finetune, map_location='cpu')['model']
        print(target_model.load_state_dict(finetune))
        
    elif args.backbone =='timesformer':
        target_model = Timesformer(
            img_size=224, patch_size=16,
        embed_dim=768, depth=12, num_heads=12,
        num_frames=8,
        time_init='zeros',
        attention_style='frozen-in-time',
        ln_pre=False,
        act_layer=QuickGELU,
        is_tanh_gating=False,
        drop_path_rate=0.,)
        finetune = torch.load(args.finetune, map_location='cpu')['model_state']
        msg = target_model.load_state_dict(finetune,True)
    elif args.backbone =='r3d':
        from torchvision.models.video import r3d_18
        target_model = r3d_18(pretrained=True)
    else:#! VideoMAE는 여기로.
        target_model, target_preprocess = data_utils.get_target_model(args.backbone, device,args)
    return target_model



def main(args):
    print("=" * 50)
    print(" Feature Extraction Process")
    print(f" Dataset: {args.data_set}")
    print(f" Backbone Model: {args.backbone}")
    print(f" Dual Encoder Model: {args.dual_encoder}")
    print(f" Device: {args.device}")
    print("=" * 50)
    
    device = torch.device(args.device)
    video_utils.init_distributed_mode(args)
    
    seed = args.seed
    random.seed(seed)  # Python random seed 설정
    np.random.seed(seed)  # NumPy random seed 설정
    torch.manual_seed(seed)  # PyTorch random seed 설정 (CPU)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch random seed 설정 (CUDA)
        torch.cuda.manual_seed_all(seed)  # 모든 GPU에 적용
        
    #! Load model
    dual_encoder = get_multi_modal_encoder(args,device).to(device).eval() 
    video_encoder = get_video_encoder(args,device).to(device).eval()
    frozen_all_parameters(dual_encoder,args.dual_encoder);frozen_all_parameters(video_encoder,args.backbone)
    
    for mode in ["train","val"]:
        
        is_train=True if mode=="train" else False
        test_mode = False
        data_name = args.data_set + "_"+mode #! _train or _val
            
        target_save_name, vlm_save_name,  = get_save_backbone_name(args.dual_encoder, args.backbone, data_name,args.activation_dir)
    
        # Load video dataset
        video_dataset, nb_classes = datasets.build_dataset(is_train, test_mode, args)
        video_dataloader = DataLoader(video_dataset, args.batch_size, num_workers=args.num_workers, pin_memory=True,shuffle=False)
        if not args.only_backbone:
            print(f"Extractin features from VLM {args.dual_encoder}")
            if args.dual_encoder =='clip':
                save_clip_image_features(dual_encoder, video_dataloader, vlm_save_name, args.batch_size, device=device,args=args)
            elif args.dual_encoder =='lavila':
                save_lavila_video_features(dual_encoder, video_dataloader, vlm_save_name, args.batch_size, device=device,args=args)
            elif 'internvid' in args.dual_encoder:
                save_internvid_video_features(dual_encoder, video_dataloader, vlm_save_name, args.batch_size, device=device,args=args)
        if args.only_vlm:
            continue
        print(f"Extractin features from Video backbone {args.backbone}")

        if args.backbone.startswith("clip"):
            save_clip_video_features(video_encoder, video_dataloader, target_save_name, args.batch_size, device,args)
        elif args.backbone.startswith("vmae_") or args.backbone=='AIM':
            save_vmae_video_features(video_encoder,video_dataloader,target_save_name,args.batch_size,device,args)
        elif args.backbone.startswith("r3d"):
            args.short_side_size=112
            args.input_size=112
            video_dataset, nb_classes = datasets.build_dataset(is_train, test_mode, args)
            video_dataloader = DataLoader(video_dataset, args.batch_size, num_workers=args.num_workers, pin_memory=True,shuffle=False)
            save_r3d_video_features(video_encoder, video_dataloader, target_save_name,args.batch_size,device)
    
        #!Now we have only VMAE video backbone. Has to be added more Backbone.

    return
    
    




if __name__=='__main__':
    args = parser.parse_args()
    main(args)