import os
import torch.distributed as dist
import math
import torch
import clip
import data_utils
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import DataLoader
import lavila.models as models
from lavila.models import Timesformer
from lavila.utils import inflate_positional_embeds
from lavila.openai_model import QuickGELU
from transforms import Permute
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
import video_utils
PM_SUFFIX = {"max":"_max", "avg":""}

def save_target_activations(target_model, dataset, save_name, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg'):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """
    _make_save_dir(save_name)
    save_names = {}    
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)
        
    if _all_saved(save_names):
        return
    
    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        hooks[target_layer] = eval(command)
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = target_model(images.to(device))
    
    for target_layer in target_layers:
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_image_features(model, dataset, save_name, batch_size=1000 , device = "cuda",args = None):
    _make_save_dir(save_name)
    all_features = []

    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            # t = (images.shape)[2]
            # if args.center_frame:
            #     images = images.squeeze(2)
            features = model.encode_video(images.to(device))# B,T, D
            all_features.append(features.cpu())

    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return
def save_lavila_video_features(model, dataset, save_name, batch_size=1000 , device = "cuda",args = None):
    _make_save_dir(save_name)
    all_features = []

    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            # t = (images.shape)[2]
            # if args.center_frame:
            #     images = images.squeeze(2)
            features = model.encode_image(images.to(device))# B, D
            all_features.append(features.cpu())

    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return
def save_clip_text_features(model, text, save_name, batch_size=1000):
    
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return

def save_activations(clip_name, target_name, target_layers, d_probe, 
                     concept_set, batch_size, device, pool_mode, save_dir,args=None):
    
    s_concept_set, t_concept_set,p_concept_set = concept_set

    target_save_name, clip_save_name, s_text_save_name, t_text_save_name,p_text_save_name = get_save_names(clip_name, target_name, 
                                                                    "{}", d_probe, concept_set, 
                                                                      pool_mode, save_dir)
    # save_names = {"clip": clip_save_name, "text": text_save_name}
    # for target_layer in target_layers:
    #     save_names[target_layer] = target_save_name.format(target_layer)
        
    # if _all_saved(save_names):
    #     return
    
    if args.dual_encoder == "lavila":
        dual_encoder_model, clip_preprocess = get_lavila(args,device=device) 
    elif args.dual_encoder == "clip":
        name = 'ViT-B/16'
        dual_encoder_model, clip_preprocess = clip.load(name, device=device)
    elif "internvid" in args.dual_encoder:
        dual_encoder_model, _ = get_intervid(args,device)
        clip_preprocess = None
        name = 'ViT-B/16'
        clip_model, clip_preprocess = clip.load(name, device=device)

    #! Load backbone 
    if target_name.startswith("clip_"):
        target_model, target_preprocess = clip.load(target_name[5:], device=device)
    elif target_name =='timesformer':
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
    else:
        target_model, target_preprocess = data_utils.get_target_model(target_name, device,args)
    target_model.to(device)
    target_model.eval()
    dual_encoder_model.to(device)
    dual_encoder_model.eval()
    
    #setup data
    #! Video Dataset은 embedded preprocess 
    data_c = data_utils.get_data(d_probe, clip_preprocess,args)
    data_c.end_point=2
    # data_c = data_utils.get_data(d_probe, target_preprocess,args)

    with open(s_concept_set, 'r') as f: 
        s_words = (f.read()).split('\n')
    with open(t_concept_set, 'r') as f: 
        t_words = (f.read()).split('\n')
    with open(p_concept_set, 'r') as f: 
        p_words = (f.read()).split('\n')
    
    if args.dual_encoder =='clip':
        s_text = clip.tokenize(["{}".format(word) for word in s_words]).to(device)
        t_text = clip.tokenize(["{}".format(word) for word in t_words]).to(device)
        p_text = clip.tokenize(["{}".format(word) for word in p_words]).to(device)
        save_clip_text_features(dual_encoder_model , s_text, s_text_save_name, batch_size)
        save_clip_text_features(dual_encoder_model , t_text, t_text_save_name, batch_size)
        save_clip_text_features(dual_encoder_model , p_text, p_text_save_name, batch_size)
        if not args.saved_features:
            save_clip_image_features(dual_encoder_model, data_c, clip_save_name, batch_size, device=device,args=args)
    elif args.dual_encoder =='lavila':
        s_text = clip.tokenize(["{}".format(word) for word in s_words]).to(device)
        t_text = clip.tokenize(["{}".format(word) for word in t_words]).to(device)
        p_text = clip.tokenize(["{}".format(word) for word in p_words]).to(device)
        save_clip_text_features(dual_encoder_model , s_text, s_text_save_name, batch_size)
        save_clip_text_features(dual_encoder_model , t_text, t_text_save_name, batch_size)
        save_clip_text_features(dual_encoder_model , p_text, p_text_save_name, batch_size)
        if not args.saved_features:
            save_lavila_video_features(dual_encoder_model, data_c, clip_save_name, batch_size, device=device,args=args)
    elif 'internvid' in args.dual_encoder:
        s_text = clip.tokenize(["A photo of {}.".format(word) for word in s_words]).to(device)
        t_text = dual_encoder_model.text_encoder.tokenize(["{}".format(word) for word in t_words], context_length=32).to(device)
        p_text = clip.tokenize(["A person {}.".format(word) for word in p_words]).to(device)
        # save_internvid_text_features(dual_encoder_model , s_text, s_text_save_name, batch_size)
        save_clip_text_features(clip_model , s_text, s_text_save_name, batch_size)
        
        save_internvid_text_features(dual_encoder_model , t_text, t_text_save_name, batch_size)
        # save_internvid_text_features(dual_encoder_model , p_text, p_text_save_name, batch_size)
        save_clip_text_features(clip_model , p_text, p_text_save_name, batch_size)
        
        if not args.saved_features:
            save_internvid_video_features(dual_encoder_model, data_c, clip_save_name, batch_size, device=device,args=args)
    if args.saved_features:# 이 아래는 saved_feature이면 안해도됌.
        return
    
    if target_name.startswith("clip_"):
        save_clip_image_features(target_model, data_c, target_save_name, batch_size, device,args)
    elif target_name.startswith("vmae_") or target_name=='AIM':
        save_vmae_video_features(target_model,data_c,target_save_name,batch_size,device,args)
    else:
        save_target_activations(target_model, data_c, target_save_name, target_layers,
                                batch_size, device, pool_mode)
    return
    
    
def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, similarity_fn, 
                                   return_target_feats=True):
    image_features = torch.load(clip_save_name)
    text_features = torch.load(text_save_name)
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True).float()
        text_features /= text_features.norm(dim=-1, keepdim=True).float()
        clip_feats = (image_features @ text_features.T)
    del image_features, text_features
    torch.cuda.empty_cache()
    
    target_feats = torch.load(target_save_name)
    similarity = similarity_fn(clip_feats, target_feats)
    
    del clip_feats
    torch.cuda.empty_cache()
    
    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity
    
def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.mean(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output.detach().cpu())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.amax(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output.detach().cpu())
    return hook

    
def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir):
    s_concept, t_concept,p_concept = concept_set    
    # if target_name.startswith("clip_"):
    target_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, target_name.replace('/', ''))
    # else:
    #     target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer,
    #                                              PM_SUFFIX[pool_mode])
    clip_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    s_concept_set_name = (s_concept.split("/")[-1]).split(".")[0] if s_concept is not None else None
    t_concept_set_name = (t_concept.split("/")[-1]).split(".")[0] if t_concept is not None else None
    p_concept_set_name = (p_concept.split("/")[-1]).split(".")[0] if p_concept is not None else None
    s_text_save_name = "{}/{}_{}.pt".format(save_dir, s_concept_set_name, clip_name.replace('/', ''))
    t_text_save_name = "{}/{}_{}.pt".format(save_dir, t_concept_set_name, clip_name.replace('/', ''))
    p_text_save_name = "{}/{}_{}.pt".format(save_dir, p_concept_set_name, clip_name.replace('/', ''))
    
    return target_save_name, clip_save_name, s_text_save_name, t_text_save_name,p_text_save_name

    
def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return

def get_accuracy_cbm(model, dataset, device, batch_size=250, num_workers=10):
    correct = 0
    total = 0
    model.eval()
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=10,
                                           pin_memory=True)):
        with torch.no_grad():
            #outs = target_model(images.to(device))
            outs, concept_activation = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            correct += torch.sum(pred.cpu()==labels)
            total += len(labels)
    return correct/total
def get_accuracy_and_concept_distribution_cbm(model,k,dataset, device, batch_size=250, num_workers=10,save_name=None):
    correct = 0
    total = 0
    num_object,num_action,num_scene=model.s_proj_layer.weight.shape[0],model.t_proj_layer.weight.shape[0],model.p_proj_layer.weight.shape[0]
    num_concept = num_object + num_action + num_scene  
    
    # concept set의 인덱스 범위
    object_range = (0, num_object)  
    action_range = (num_object, num_object + num_action)
    scene_range = (num_object + num_action, num_concept)
    
    # concept set별 개수 기록용 변수 초기화
    total_object_count = 0
    total_action_count = 0
    total_scene_count = 0
    model.eval()
    # with open(os.path.join(save_name,"class_concept_contribution.csv"), "w") as f:
        # f.write("label,topk_indices\n")
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=10,
                                           pin_memory=True)):
        with torch.no_grad():
            #outs = target_model(images.to(device))
            outs, concept_activation = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            correct += torch.sum(pred.cpu()==labels)
            total += len(labels)
            pred_weights = model.final.weight[pred]  # shape: (batch, num_concept)
            concept_contribution = concept_activation * pred_weights  
            topk_indices = torch.topk(concept_contribution, k, dim=1).indices  # shape: (batch, topk)

            # 각 샘플에 대해 concept set의 개수 세기
            for i in range(topk_indices.size(0)):
                object_count = action_count = scene_count = 0
                for idx in topk_indices[i]:
                    if object_range[0] <= idx < object_range[1]:
                        object_count += 1
                    elif action_range[0] <= idx < action_range[1]:
                        action_count += 1
                    elif scene_range[0] <= idx < scene_range[1]:
                        scene_count += 1
                
                # 전체 개수 합산
                total_object_count += object_count
                total_action_count += action_count
                total_scene_count += scene_count
                
                # with open(os.path.join(save_name,"class_concept_contribution.csv"), "a") as f:
                    # topk_indices_str = ",".join(map(str, topk_indices[i].cpu().tolist()))
                    # f.write(f"{labels[i].item()},{topk_indices_str}\n")
    
    return correct/total,[total_object_count,total_action_count,total_scene_count]


def get_preds_cbm(model, dataset, device, batch_size=250, num_workers=2):
    preds = []
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers,
                                           pin_memory=True)):
        with torch.no_grad():
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    return preds

def get_concept_act_by_pred(model, dataset, device):
    preds = []
    concept_acts = []
    for images, labels in tqdm(DataLoader(dataset, 500, num_workers=8, pin_memory=True)):
        with torch.no_grad():
            outs, concept_act = model(images.to(device))
            concept_acts.append(concept_act.cpu())
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    concept_acts = torch.cat(concept_acts, dim=0)
    concept_acts_by_pred=[]
    for i in range(torch.max(pred)+1):
        concept_acts_by_pred.append(torch.mean(concept_acts[preds==i], dim=0))
    concept_acts_by_pred = torch.stack(concept_acts_by_pred, dim=0)
    return concept_acts_by_pred


def save_vmae_video_features(model, dataset, save_name, batch_size=1000 , device = "cuda",args = None):
    _make_save_dir(save_name)
    
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # for end_point in range(5):
    all_features = []
    #     dataset.end_point = end_point
    with torch.no_grad():
        for videos, labels in tqdm(DataLoader(dataset, batch_size, num_workers=10, pin_memory=True,shuffle=False)):
            # t = (images.shape)[2]
            # if args.center_frame:
            #     images = images.squeeze(2)
            features = model.forward_features(videos.to(device))
            all_features.append(features.cpu())
    torch.save(torch.cat(all_features), save_name)

    #free memory
    del all_features
    torch.cuda.empty_cache()
    return





def get_lavila(args,device):
    if args.lavila_ckpt:
        ckpt_path = args.lavila_ckpt
    else:
        raise Exception('no checkpoint found')
    ckpt = torch.load(ckpt_path, map_location='cpu')

    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))
    model = getattr(models, old_args.model)(
        pretrained=old_args.load_visual_pretrained,
        pretrained2d=old_args.load_visual_pretrained is not None,
        text_use_cls_token=old_args.use_cls_token,
        project_embed_dim=old_args.project_embed_dim,
        timesformer_gated_xattn=False,
        timesformer_freeze_space=False,
        num_frames=args.dual_encoder_frames,
        drop_path_rate=0.,
    )
    if 'TIMESFORMER' in old_args.model or 'EGOVLP' in old_args.model:
        # inflate weight
        print('=> inflating PE in models due to different frame numbers')
        state_dict = inflate_positional_embeds(
            model.state_dict(), state_dict,
            num_frames=args.dual_encoder_frames,
            load_temporal_fix='bilinear',
        )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    
    crop_size = 224 if '336PX' not in old_args.model else 336
    val_transform = transforms.Compose([
            Permute([3, 0, 1, 2]),    # T H W C -> C T H W
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            (transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) if 'OPENAI' not in old_args.model else
             transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])),
    ])
    print("\n\n\n***********LAVILA Load***************")
    print("=> loaded resume checkpoint '{}' (epoch {})".format(ckpt_path, ckpt['epoch']))
    return model,val_transform



def get_intervid(args,device):
    from viclip import get_viclip, retrieve_text, _frame_from_video
    model_cfgs = {
    'viclip-l-internvid-10m-flt': {
        'size': 'l',
        'pretrained': 'xxx/ViCLIP-L_InternVid-FLT-10M.pth',
    },
    'viclip-l-internvid-200m': {
        'size': 'l',
        'pretrained': 'xxx/ViCLIP-L_InternVid-200M.pth',
    },
    'viclip-b-internvid-10m-flt': {
        'size': 'b',
        'pretrained': '/data/dataset/video_checkpoint/ViCLIP-B_InternVid-FLT-10M.pth',
    },
    'viclip-b-internvid-200m': {
        'size': 'b',
        'pretrained': '/data/dataset/video_checkpoint/ViCLIP-B_InternVid-200M.pth',
    },
    }
    cfg = model_cfgs['viclip-b-internvid-200m'] if '200m' in args.dual_encoder else model_cfgs['viclip-b-internvid-10m-flt']
    model_l = get_viclip(cfg['size'], cfg['pretrained'])
    assert(type(model_l)==dict and model_l['viclip'] is not None and model_l['tokenizer'] is not None)
    clip, tokenizer = model_l['viclip'], model_l['tokenizer']
    clip = clip.to(device)
    clip = clip.eval()
    return clip, tokenizer
    



def save_internvid_video_features(model, dataset, save_name, batch_size=1000 , device = "cuda",args = None):
    _make_save_dir(save_name)
    all_labels = []
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # for end_point in range(5):
    all_features = []
        # dataset.end_point = end_point
        # dl=DataLoader(dataset, batch_size, num_workers=10, pin_memory=True,shuffle=False)
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=10, pin_memory=True,shuffle=False)):
            # t = (images.shape)[2]
            # if args.center_frame:
            #     images = images.squeeze(2)
            features = model.encode_vision(images.to(device))
            all_features.append(features.cpu())
            # all_labels+=(labels.tolist())
    torch.save(torch.cat(all_features), save_name)
        # torch.save(torch.cat(all_labels), os.path.join(save_dir,'label.pt'))
        # with open(os.path.join(save_dir,"label.txt"), "w") as file:
        #     for item in all_labels:
        #         file.write(f"{item}\n") 
        #free memory
    del all_features
    torch.cuda.empty_cache()
    return
def save_internvid_text_features(model, text, save_name, batch_size=1000):
    
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.text_encoder(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return

def analysis_backbone_dim(model,dataset,args,number_act=5,view_num=50):
    import numpy as np
    distribution = np.zeros(768, dtype=int)
    for i in range(len(dataset)):
        if i%100==0:
            print(f'iteration ** {i}/{len(dataset)}')
        # image,label ,path= val_pil_data[i]
        x, _ = dataset[i]
        x = x.unsqueeze(0).to(args.device)
        backbone_feat,concept_activation,outputs = model.get_feature(x)
        # outputs, _ = s_model(x)
        _, top_classes = torch.topk(outputs[0], dim=0, k=2)
        contributions = concept_activation[0]*model.final.weight[top_classes[0], :]
        high_spatial_concept = torch.topk(contributions, dim=0, k=2)[1][0]
        # top_logit_vals, top_classes = torch.topk(concept_activation[0], dim=0, k=5)
        contributions = backbone_feat[0]*model.proj_layer.weight[high_spatial_concept, :]
        _,top_dim = torch.topk(contributions, dim=0, k=number_act)
        dims = top_dim.cpu().numpy()
        distribution += np.bincount(dims, minlength=768)  # 분포 카운트 누적
        sorted_indices = np.argsort(distribution)
        N = view_num  # 예: 상위 10개, 하위 10개의 차원을 출력
        bottom_N_indices = sorted_indices[:N]  # 하위 N개
        top_N_indices = sorted_indices[-N:][::-1]  # 상위 N개 (큰 값부터 출력)

        # 해당 차원의 활성화 횟수
        bottom_N_values = distribution[bottom_N_indices]
        top_N_values = distribution[top_N_indices]

    # 결과 출력
    print(f"Top {N} most activated dimensions: {top_N_indices}")
    print(f"Activation counts for most activated dimensions: {top_N_values}\n")

    print(f"Top {N} least activated dimensions: {bottom_N_indices}")
    print(f"Activation counts for least activated dimensions: {bottom_N_values}")
    
    

