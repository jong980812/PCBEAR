import json
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
from video_dataloader import video_utils
from PIL import Image
from IPython.display import display, Image as IPImage
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

def save_clip_image_features(model, dataloader, save_name, batch_size=1000 , device = "cuda",args = None):
    _make_save_dir(save_name)
    all_features = []

    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, labels in (dataloader):
            # t = (images.shape)[2]
            # if args.center_frame:'
            '''
            로더에서 나온 비디오 센터 프레임을 골라 고를떄 
            '''
            #     images = images.squeeze(2)
            features = model.encode_video(images.to(device))# B,T, D
            features = features.mean(dim=1)  
            all_features.append(features.cpu())

    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_video_features(model, dataloader, save_name, batch_size=1000 , device = "cuda",args = None):
    _make_save_dir(save_name)
    all_features = []

    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for videos, labels in (dataloader):
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
        s_text = clip.tokenize(["A video in which a {} object is present.".format(word) for word in s_words]).to(device)
        t_text = dual_encoder_model.text_encoder.tokenize(["{}".format(word) for word in t_words], context_length=32).to(device)
        p_text = clip.tokenize(["A scene set in a {}".format(word) for word in p_words]).to(device)
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
    


def save_text_features(concept_set, args, dual_encoder_model,key):
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0] if concept_set is not None else None
    concepts_save_name = "{}/{}_{}.pt".format(args.activation_dir, concept_set_name, args.dual_encoder)    
    with open(concept_set, 'r') as f: 
        words = (f.read()).split('\n')
    if args.dual_encoder =='clip':
        concepts = clip.tokenize(["A photo of {}".format(word) for word in words]).to(args.device)
        save_clip_text_features(dual_encoder_model , concepts, concepts_save_name, args.batch_size)
    elif args.dual_encoder =='lavila':
        concepts = clip.tokenize(["A video of {}".format(word) for word in words]).to(args.device)
        save_clip_text_features(dual_encoder_model , concepts, concepts_save_name, args.batch_size)
    elif 'internvid' in args.dual_encoder:
        # concepts = clip.tokenize(["A photo of {}.".format(word) for word in words]).to(args.device)
        if key == "spatial":
            concepts = dual_encoder_model.text_encoder.tokenize(["A video of {}".format(word) for word in words], context_length=32).to(args.device)
        elif key == "place":
            concepts = dual_encoder_model.text_encoder.tokenize(["A video of {}".format(word) for word in words], context_length=32).to(args.device)
        elif key == "temporal":
            concepts = dual_encoder_model.text_encoder.tokenize(["A video of {}".format(word) for word in words], context_length=32).to(args.device)
        save_internvid_text_features(dual_encoder_model , concepts, concepts_save_name, args.batch_size)

    return concepts_save_name
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
def get_save_backbone_name(clip_name, target_name, d_probe, save_dir):
    target_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, target_name.replace('/', ''))
    # else:
    #     target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer,
    #                                              PM_SUFFIX[pool_mode])
    clip_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    
    return target_save_name, clip_save_name
    
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


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import numpy as np
from sklearn.metrics import confusion_matrix

def get_off_diagonal_confusion_rate(y_true, y_pred, class_names, subset_labels):
    """
    주어진 클래스 subset에 대해 confusion matrix의 off-diagonal 합과 혼동률 계산

    Args:
        y_true (List[int] or np.array): ground-truth 레이블들
        y_pred (List[int] or np.array): 예측값들
        class_names (List[str]): 전체 클래스 이름 리스트
        subset_labels (List[str]): 보고 싶은 클래스 이름들

    Returns:
        dict: {
            'subset_labels': [...],
            'off_diagonal_sum': int,
            'total': int,
            'confusion_rate': float,
            'confusion_matrix': np.ndarray
        }
    """
    subset_indices = [class_names.index(cls) for cls in subset_labels]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # subset만 필터링
    mask = np.isin(y_true, subset_indices)
    y_true_subset = y_true[mask]
    y_pred_subset = y_pred[mask]

    cm = confusion_matrix(y_true_subset, y_pred_subset, labels=subset_indices)

    correct = np.trace(cm)
    total = cm.sum()
    off_diagonal = total - correct
    confusion_rate = off_diagonal / total if total > 0 else 0.0

    return {
        'subset_labels': subset_labels,
        'off_diagonal_sum': off_diagonal,
        'total': total,
        'confusion_rate': confusion_rate,
        'confusion_matrix': cm
    }
def get_class_subset_confusion(y_true, y_pred, class_names, target_classes):
    """
    주어진 클래스 subset에 대해 confusion matrix 및 classification report 반환
    
    Args:
        y_true, y_pred: 전체 예측 결과
        class_names: 전체 클래스 이름 리스트
        target_classes: ['pushup', 'pullup', 'situp'] 등 관심 클래스
    
    Returns:
        cm (N x N confusion matrix), report (str)
    """
    target_indices = [class_names.index(cls) for cls in target_classes]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 해당 클래스만 필터링
    mask = np.isin(y_true, target_indices)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=target_indices)
    report = classification_report(y_true_filtered, y_pred_filtered, labels=target_indices, target_names=target_classes, digits=3)

    return cm, report
# def get_detailed_metrics_cbm(model, dataset, device, batch_size=250, num_workers=10, class_names=None, return_raw=False):
#     all_preds = []
#     all_labels = []
#     model.eval()
#     for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)):
#         with torch.no_grad():
#             outputs, _ = model(images.to(device))
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.numpy())

#     cm = confusion_matrix(all_labels, all_preds)
#     report = classification_report(all_labels, all_preds, target_names=class_names, digits=3)

#     if return_raw:
#         return report, cm, all_labels, all_preds
#     else:
#         return report, cm
def get_detailed_metrics_cbm(model, dataset, device, batch_size=250, num_workers=10, class_names=None, return_raw=False):
    all_preds = []
    all_labels = []

    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    for images, labels in tqdm(dataloader):
        with torch.no_grad():
            outputs, _ = model(images.to(device))
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 전체 confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    correct = sum([pred == label for pred, label in zip(all_preds, all_labels)])
    total = len(all_labels)
    accuracy = correct / total

    print(f"Accuracy: {accuracy:.4f}")
    # classification report
    report = classification_report(all_labels, all_preds,labels=[i for i in range(dataloader.dataset.nb_classes)],target_names=class_names, digits=3)

    # 클래스별 accuracy 계산
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    class_accuracy_dict = {}
    if class_names is not None:
        for idx, class_name in enumerate(class_names):
            class_accuracy_dict[class_name] = class_accuracies[idx]
    else:
        for idx in range(len(class_accuracies)):
            class_accuracy_dict[f"Class_{idx}"] = class_accuracies[idx]

    if return_raw:
        return report, cm, all_labels, all_preds, class_accuracy_dict
    else:
        return report, cm, class_accuracy_dict

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


def save_vmae_video_features(model, dataloader, save_name, batch_size=1000 , device = "cuda",args = None):
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
        for videos, labels in (dataloader):
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
def save_r3d_video_features(model, dataloader, save_name, batch_size=1000, device="cuda"):
    model.eval()
    model.to(device)

    # fc 제거하고 feature만 받기
    model.fc = torch.nn.Identity()

    if os.path.exists(save_name):
        return

    save_dir = os.path.dirname(save_name)
    os.makedirs(save_dir, exist_ok=True)

    all_features = []

    with torch.no_grad():
        for videos, _ in dataloader:
            videos = videos.to(device)
            features = model(videos)  # (B, 512)
            all_features.append(features.cpu())

    torch.save(torch.cat(all_features), save_name)

    del all_features
    torch.cuda.empty_cache()
    return
def save_lavila_video_features(model, dataloader, save_name, batch_size=1000 , device = "cuda",args = None):
    _make_save_dir(save_name)
    all_features = []

    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, labels in (dataloader):
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
def save_internvid_video_features(model, dataloader, save_name, batch_size=1000 , device = "cuda",args = None):
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
        for images, labels in (dataloader):
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

def save_args(args,save_name):
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    

import torch
import numpy as np
from PIL import Image
from IPython.display import display, Image as IPImage
# Dataloader로부터 얻은 tensor (C, T, H, W)

def visualize_gif(image,label,path,index,img_ind):
    tensor = image
    if len(tensor.shape)>4:
        tensor = tensor[index]
    video_name = path.split('/')[-1].split('.')[0]
    # image_folder = f'./gif/{img_ind}_{video_name}'
    # os.makedirs(image_folder, exist_ok=True)
    gif_path = f'./gif/{img_ind}_{video_name}_{index}.gif'

    # if not os.path.exists(gif_path):
    # 텐서를 (T, H, W, C)로 변환
    tensor = tensor.permute(1, 2, 3, 0)  # (T, H, W, C)

    # 텐서를 numpy 배열로 변환
    tensor_np = tensor.numpy()

    # 이미지 리스트 생성
    images = []
    for i in range(tensor_np.shape[0]):
        # 각 프레임을 (H, W, C) 형태로 변환 후 0~255 범위로 스케일링
        frame = ((tensor_np[i] - tensor_np[i].min()) / (tensor_np[i].max() - tensor_np[i].min()) * 255).astype(np.uint8)
        frame_image = Image.fromarray(frame)
        images.append(frame_image)
        # frame_image_path = os.path.join(image_folder, f'77688_0000{i+10}.png')
        # frame_image.save(frame_image_path)
    # GIF로 저장 (duration은 각 프레임 사이의 시간, 100ms = 0.1초)
    

    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)
    # else:
    #     pass
    # Jupyter에서 GIF 표시
    display(IPImage(filename=gif_path))