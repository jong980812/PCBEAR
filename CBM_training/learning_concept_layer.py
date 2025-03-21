from video_dataloader import datasets
import cbm
import pickle
from collections import OrderedDict
from typing import Tuple, Union
from einops import rearrange
from sparselinear import SparseLinear
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import cbm
import torch
from tqdm import tqdm
import debugging
from timm.utils import accuracy
import os
import random
import cbm_utils
import data_utils
import similarity
import argparse
import datetime
import torch.optim as optim
import json
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset
from video_dataloader import video_utils
import torch.distributed as dist

def spatio_temporal_joint(args,
                            s_concepts,
                            target_features,
                            val_target_features,
                            s_clip_features,
                            s_val_clip_features,
                            t_concepts,
                            t_clip_features,
                            t_val_clip_features,
                            save_name):
    save_spatial = os.path.join(save_name,'spatial')
    save_temporal = os.path.join(save_name,'temporal')
    save_spatio_temporal = os.path.join(save_name,'spatio_temporal')
    # save_spatio_temporal = os.path.join(save_name,'spatio_temporal')
    os.mkdir(save_spatial)
    os.mkdir(save_temporal)
    os.mkdir(save_spatio_temporal)
    s_W_c,s_concepts = train_cocept_layer(args,
                               s_concepts,
                               target_features,
                               val_target_features,
                               s_clip_features,
                               s_val_clip_features,
                               save_spatial)
    t_W_c,t_concepts = train_cocept_layer(args,
                               t_concepts,
                               target_features,
                               val_target_features,
                               t_clip_features,
                               t_val_clip_features,
                               save_temporal)

    
    del s_clip_features, s_val_clip_features,t_clip_features, t_val_clip_features
    

    st_W_c = torch.cat([s_W_c,t_W_c],dim=0)
    st_concepts = s_concepts+t_concepts
    train_classification_layer(args,
                               W_c=st_W_c,
                               pre_concepts=None,
                               concepts = st_concepts,
                               target_features=target_features,
                               val_target_features=val_target_features,
                                save_name=save_spatio_temporal
                               )

    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # d_val = args.data_set + "_test"
    # val_data_t = data_utils.get_data(d_val,args=args)
    # device = torch.device(args.device)
    
    # s_model,t_model = cbm.load_cbm_two_stream(save_name, device,args)

    # print("!****! Start test Spatio")
    # accuracy = cbm_utils.get_accuracy_cbm(s_model, val_data_t, device,32,10)
    # print("!****! Spatio Accuracy: {:.2f}%".format(accuracy*100))
    
    # print("?***? Start test Temporal")
    # accuracy = cbm_utils.get_accuracy_cbm(t_model, val_data_t, device,32,10)
    # print("?****? Temporal Accuracy: {:.2f}%".format(accuracy*100))
    
    return

def spatio_temporal_three_joint(args,
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
                            save_name):
    save_spatial = os.path.join(save_name,'spatial')
    save_temporal = os.path.join(save_name,'temporal')
    save_place = os.path.join(save_name,'place')
    save_spatio_temporal_place = os.path.join(save_name,'spatio_temporal_place')
    # save_spatio_temporal = os.path.join(save_name,'spatio_temporal')
    os.mkdir(save_spatial)
    os.mkdir(save_temporal)
    os.mkdir(save_place)
    os.mkdir(save_spatio_temporal_place)
    train_cs,val_cs = [],[]
    s_W_c,s_concepts = train_cocept_layer(args,
                               s_concepts,
                               target_features,
                               val_target_features,
                               s_clip_features, 
                               s_val_clip_features,
                               save_spatial)
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(s_concepts), bias=False)
    proj_layer.load_state_dict({"weight":s_W_c})
    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std

        val_c -= train_mean
        val_c /= train_std
        
        
        train_cs.append(train_c)
        val_cs.append(val_c)
    torch.save(train_mean, os.path.join(save_spatial, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_spatial, "proj_std.pt"))
        
    t_W_c,t_concepts = train_cocept_layer(args,
                               t_concepts,
                               target_features,
                               val_target_features,
                               t_clip_features,
                               t_val_clip_features,
                               save_temporal)
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(t_concepts), bias=False)
    proj_layer.load_state_dict({"weight":t_W_c})
    with torch.no_grad():

        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std

        val_c -= train_mean
        val_c /= train_std

        
        train_cs.append(train_c)
        val_cs.append(val_c)
    torch.save(train_mean, os.path.join(save_temporal, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_temporal, "proj_std.pt"))
    p_W_c,p_concepts = train_cocept_layer(args,
                               p_concepts,
                               target_features,
                               val_target_features,
                               p_clip_features,
                               p_val_clip_features,
                               save_place)
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(p_concepts), bias=False)
    proj_layer.load_state_dict({"weight":p_W_c})
    with torch.no_grad():

        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std

        val_c -= train_mean
        val_c /= train_std
        
        
        train_cs.append(train_c)
        val_cs.append(val_c)
    torch.save(train_mean, os.path.join(save_place, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_place, "proj_std.pt"))
    train_c = torch.cat(train_cs,dim=1)
    val_c = torch.cat(val_cs,dim=1)
    del s_clip_features, s_val_clip_features,t_clip_features, t_val_clip_features,p_clip_features, p_val_clip_features
    

    stp_W_c = torch.cat([s_W_c,t_W_c,p_W_c],dim=0)
    stp_concepts = s_concepts+t_concepts+p_concepts
    train_classification_layer(args,
                               W_c=stp_W_c,
                               pre_concepts=None,
                               concepts = stp_concepts,
                               target_features=target_features,
                               val_target_features=val_target_features,
                                save_name=save_spatio_temporal_place,
                                joint=(train_c,val_c)
                               )

    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
    # d_val = args.data_set + "_test"
    # val_data_t = data_utils.get_data(d_val,args=args)
    # val_data_t.end_point = 2
    # device = torch.device(args.device)
    
    if args.debug:
        debugging.debug(args,save_name)
    
    # model,_ = cbm.load_cbm_triple(save_name, device,args)
    # print("?***? Start test")
    # accuracy = cbm_utils.get_accuracy_cbm(model, val_data_t, device,128,10)
    # print("?****? Accuracy: {:.2f}%".format(accuracy*100))
    # d_val = args.data_set + "_test"
    # val_data_t = data_utils.get_data(d_val,args=args)
    # device = torch.device(args.device)
    
    # s_model,t_model = cbm.load_cbm_two_stream(save_name, device,args)

    # print("!****! Start test Spatio")
    # accuracy = cbm_utils.get_accuracy_cbm(s_model, val_data_t, device,32,10)
    # print("!****! Spatio Accuracy: {:.2f}%".format(accuracy*100))
    
    # print("?***? Start test Temporal")
    # accuracy = cbm_utils.get_accuracy_cbm(t_model, val_data_t, device,32,10)
    # print("?****? Temporal Accuracy: {:.2f}%".format(accuracy*100))
    
    return
def spatio_temporal_three_ensemble(args,
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
                            save_name):
    save_spatial = os.path.join(save_name,'spatial')
    save_temporal = os.path.join(save_name,'temporal')
    save_place = os.path.join(save_name,'place')
    save_spatio_temporal_place = os.path.join(save_name,'spatio_temporal_place')
    # save_spatio_temporal = os.path.join(save_name,'spatio_temporal')
    os.mkdir(save_spatial)
    os.mkdir(save_temporal)
    os.mkdir(save_place)
    os.mkdir(save_spatio_temporal_place)
    train_cs,val_cs = [],[]
    s_W_c,s_concepts = train_cocept_layer(args,
                               s_concepts,
                               target_features,
                               val_target_features,
                               s_clip_features, 
                               s_val_clip_features,
                               save_spatial)
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(s_concepts), bias=False)
    proj_layer.load_state_dict({"weight":s_W_c})
    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std

        val_c -= train_mean
        val_c /= train_std
        
        
        train_cs.append(train_c)
        val_cs.append(val_c)
    torch.save(train_mean, os.path.join(save_spatial, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_spatial, "proj_std.pt"))
        
    t_W_c,t_concepts = train_cocept_layer(args,
                               t_concepts,
                               target_features,
                               val_target_features,
                               t_clip_features,
                               t_val_clip_features,
                               save_temporal)
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(t_concepts), bias=False)
    proj_layer.load_state_dict({"weight":t_W_c})
    with torch.no_grad():

        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std

        val_c -= train_mean
        val_c /= train_std

        
        train_cs.append(train_c)
        val_cs.append(val_c)
    torch.save(train_mean, os.path.join(save_temporal, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_temporal, "proj_std.pt"))
    p_W_c,p_concepts = train_cocept_layer(args,
                               p_concepts,
                               target_features,
                               val_target_features,
                               p_clip_features,
                               p_val_clip_features,
                               save_place)
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(p_concepts), bias=False)
    proj_layer.load_state_dict({"weight":p_W_c})
    with torch.no_grad():

        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std

        val_c -= train_mean
        val_c /= train_std
        
        
        train_cs.append(train_c)
        val_cs.append(val_c)
    torch.save(train_mean, os.path.join(save_place, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_place, "proj_std.pt"))
    # train_c = torch.cat(train_cs,dim=1)
    # val_c = torch.cat(val_cs,dim=1)
    del s_clip_features, s_val_clip_features,t_clip_features, t_val_clip_features,p_clip_features, p_val_clip_features
    cls_file = data_utils.LABEL_FILES[args.data_set]
    d_train = args.data_set + "_train"
    d_val = args.data_set + "_val"
    train_targets = data_utils.get_targets_only(d_train,args)
    val_targets = data_utils.get_targets_only(d_val,args)
    train_y = torch.LongTensor(train_targets)
    val_y = torch.LongTensor(val_targets)
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    s_train_c,t_train_c,p_train_c = train_cs
    s_val_c,t_val_c,p_val_c = val_cs
        
        


    indexed_train_ds = IndexedTensorDataset(train_c, train_y)
    val_ds = TensorDataset(val_c,val_y)
        
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)

    # Make linear model and zero initialize
    linear = torch.nn.Linear(train_c.shape[1],len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    
    STEP_SIZE = 0.05
    ALPHA = 0.99
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = args.lam

    # Solve the GLM path
    #concept layer to classification
    output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.n_iters, ALPHA, epsilon=1, k=1,
                    val_loader=val_loader, do_zero=False, metadata=metadata, n_ex=len(target_features), n_classes = len(classes),verbose=10)
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']
    
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))
    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        out_dict = {}
        for key in ('lam', 'lr', 'alpha', 'time'):
            out_dict[key] = float(output_proj['path'][0][key])
        out_dict['metrics'] = output_proj['path'][0]['metrics']
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict['sparsity'] = {"Non-zero weights":nnz, "Total weights":total, "Percentage non-zero":nnz/total}
        json.dump(out_dict, f, indent=2)

    # stp_W_c = torch.cat([s_W_c,t_W_c,p_W_c],dim=0)
    # stp_concepts = s_concepts+t_concepts+p_concepts


    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        
    # d_val = args.data_set + "_test"
    # val_data_t = data_utils.get_data(d_val,args=args)
    # val_data_t.end_point = 2
    # device = torch.device(args.device)
    
    
    debugging.debug(args,save_name)
    
    # model,_ = cbm.load_cbm_triple(save_name, device,args)
    # print("?***? Start test")
    # accuracy = cbm_utils.get_accuracy_cbm(model, val_data_t, device,128,10)
    # print("?****? Accuracy: {:.2f}%".format(accuracy*100))
    # d_val = args.data_set + "_test"
    # val_data_t = data_utils.get_data(d_val,args=args)
    # device = torch.device(args.device)
    
    # s_model,t_model = cbm.load_cbm_two_stream(save_name, device,args)

    # print("!****! Start test Spatio")
    # accuracy = cbm_utils.get_accuracy_cbm(s_model, val_data_t, device,32,10)
    # print("!****! Spatio Accuracy: {:.2f}%".format(accuracy*100))
    
    # print("?***? Start test Temporal")
    # accuracy = cbm_utils.get_accuracy_cbm(t_model, val_data_t, device,32,10)
    # print("?****? Temporal Accuracy: {:.2f}%".format(accuracy*100))
    
    return
def spatio_temporal_parallel(args,
                            s_concepts,
                            target_features,
                            val_target_features,
                            s_clip_features,
                            s_val_clip_features,
                            t_concepts,
                            t_clip_features,
                            t_val_clip_features,
                            save_name):
    save_spatial = os.path.join(save_name,'spatial')
    save_temporal = os.path.join(save_name,'temporal')
    # save_spatio_temporal = os.path.join(save_name,'spatio_temporal')
    os.mkdir(save_spatial)
    os.mkdir(save_temporal)
    s_W_c,s_concepts = train_cocept_layer(args,
                               s_concepts,
                               target_features,
                               val_target_features,
                               s_clip_features,
                               s_val_clip_features,
                               save_spatial)
    t_W_c,t_concepts = train_cocept_layer(args,
                               t_concepts,
                               target_features,
                               val_target_features,
                               t_clip_features,
                               t_val_clip_features,
                               save_temporal)

    
    del s_clip_features, s_val_clip_features,t_clip_features, t_val_clip_features
    



    train_classification_layer(args,
                               W_c=s_W_c,
                               pre_concepts=None,
                               concepts = s_concepts,
                               target_features=target_features,
                               val_target_features=val_target_features,
                                save_name=save_spatial
                               )
    if not args.only_s:
        train_classification_layer(args,
                                W_c=t_W_c,
                                pre_concepts=None,
                                concepts = t_concepts,
                                target_features=target_features,
                                val_target_features=val_target_features,
                                    save_name=save_temporal
                                )
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    d_val = args.data_set + "_test"
    val_data_t = data_utils.get_data(d_val,args=args)
    val_data_t.end_point = 2
    device = torch.device(args.device)
    
    s_model,t_model = cbm.load_cbm_two_stream(save_name, device,args)

    print("!****! Start test Spatio")
    accuracy = cbm_utils.get_accuracy_cbm(s_model, val_data_t, device,32,10)
    print("!****! Spatio Accuracy: {:.2f}%".format(accuracy*100))
    
    print("?***? Start test Temporal")
    accuracy = cbm_utils.get_accuracy_cbm(t_model, val_data_t, device,32,10)
    print("?****? Temporal Accuracy: {:.2f}%".format(accuracy*100))
    
    return


def spatio_temporal_serial(args,
                            s_concepts,
                            target_features,
                            val_target_features,
                            s_clip_features,
                            s_val_clip_features,
                            t_concepts,
                            t_clip_features,
                            t_val_clip_features,
                            save_name):
    save_spatial = os.path.join(save_name,'spatial')
    save_temporal = os.path.join(save_name,'temporal')
    save_classification = os.path.join(save_name,'classification')
    os.mkdir(save_spatial)
    os.mkdir(save_temporal)
    os.mkdir(save_classification)
    s_W_c,s_concepts = train_cocept_layer(args,
                               s_concepts,
                               target_features,
                               val_target_features,
                               s_clip_features,
                               s_val_clip_features,
                               save_spatial)
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(s_concepts), bias=False)
    proj_layer.load_state_dict({"weight":s_W_c})
    s_target_features = proj_layer(target_features.detach())
    s_val_target_features = proj_layer(val_target_features.detach())
    s_train_mean = torch.mean(s_target_features, dim=0, keepdim=True)
    s_train_std = torch.std(s_target_features, dim=0, keepdim=True)
    s_target_features -= s_train_mean
    s_target_features /= s_train_std
    s_val_target_features -= s_train_mean
    s_val_target_features /= s_train_std
    torch.save(s_train_mean, os.path.join(save_name,'temporal', "proj_mean.pt"))
    torch.save(s_train_std, os.path.join(save_name,'temporal', "proj_std.pt"))
    t_W_c,t_concepts = train_cocept_layer(args,
                               t_concepts,
                               s_target_features,
                               s_val_target_features,
                               t_clip_features,
                               t_val_clip_features,
                               save_temporal)

    del s_clip_features, s_val_clip_features,t_clip_features, t_val_clip_features
    



    train_classification_layer(args,
                               W_c=t_W_c,
                               pre_concepts=s_concepts,
                               concepts = t_concepts,
                               target_features=s_target_features,
                               val_target_features=s_val_target_features,
                                save_name=save_classification
                               )
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    d_val = args.data_set + "_test"
    val_data_t = data_utils.get_data(d_val,args=args)
    device = torch.device(args.device)
    
    model = cbm.load_cbm_serial(save_name, device,args)
    
    # print("?****? Start test")
    # accuracy = cbm_utils.get_accuracy_cbm(model, val_data_t, device,32,10)
    # print("?****? Accuracy: {:.2f}%".format(accuracy*100))
def spatio_temporal_single(args,
                            s_concepts,
                            target_features,
                            val_target_features,
                            s_clip_features,
                            s_val_clip_features,
                            save_name):
    # save_spatial = os.path.join(save_name,'spatial')
    # save_temporal = os.path.join(save_name,'temporal')
    # save_classification = os.path.join(save_name,'classification')
    # os.mkdir(save_spatial)
    # os.mkdir(save_temporal)
    # os.mkdir(save_classification)
    s_W_c,s_concepts,best_val_loss = train_cocept_layer(args,
                               s_concepts,
                               target_features,
                               val_target_features,
                               s_clip_features,
                               s_val_clip_features,
                               save_name)
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(s_concepts), bias=False)
    proj_layer.load_state_dict({"weight":s_W_c})
    s_target_features = proj_layer(target_features.detach())
    s_val_target_features = proj_layer(val_target_features.detach())
    s_train_mean = torch.mean(s_target_features, dim=0, keepdim=True)
    s_train_std = torch.std(s_target_features, dim=0, keepdim=True)
    s_target_features -= s_train_mean
    s_target_features /= s_train_std
    s_val_target_features -= s_train_mean
    s_val_target_features /= s_train_std
    torch.save(s_train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(s_train_std, os.path.join(save_name, "proj_std.pt"))
    # t_W_c,t_concepts = train_cocept_layer(args,
    #                            t_concepts,
    #                            s_target_features,
    #                            s_val_target_features,
    #                            t_clip_features,
    #                            t_val_clip_features,
    #                            save_temporal)

    del s_clip_features, s_val_clip_features
    



    train_classification_layer(args,
                               W_c=s_W_c,
                               pre_concepts=s_concepts,
                               concepts = s_concepts,
                               target_features=target_features,
                               val_target_features=val_target_features,
                                save_name=save_name,
                                best_val_loss=best_val_loss
                               )
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # d_val = args.data_set + "_test"
    # val_data_t = data_utils.get_data(d_val,args=args)
    # device = torch.device(args.device)
    
    # model = cbm.load_cbm_serial(save_name, device,args)
    
    # print("?****? Start test")
    # accuracy = cbm_utils.get_accuracy_cbm(model, val_data_t, device,32,10)
    # print("?****? Accuracy: {:.2f}%".format(accuracy*100))

def spatio_temporal_attention(args,
                            s_concepts,
                            target_features,
                            val_target_features,
                            s_clip_features,
                            s_val_clip_features,
                            t_concepts,
                            t_clip_features,
                            t_val_clip_features,
                            save_name):
    save_spatial = os.path.join(save_name,'spatial')
    save_temporal = os.path.join(save_name,'temporal')
    os.mkdir(save_spatial)
    os.mkdir(save_temporal)
    s_W_c,s_concepts = train_cocept_layer(args,
                               s_concepts,
                               target_features,
                               val_target_features,
                               s_clip_features,
                               s_val_clip_features,
                               save_spatial)
    t_W_c,t_concepts = train_cocept_layer(args,
                               t_concepts,
                               target_features,
                               val_target_features,
                               t_clip_features,
                               t_val_clip_features,
                               save_temporal)

    
    del s_clip_features, s_val_clip_features,t_clip_features, t_val_clip_features
    train_attention_layer(args,s_concepts,t_concepts,s_W_c,t_W_c,target_features,val_target_features,save_name)


def hard_label(args,target_features, val_target_features,save_name):
    if args.loss_mode =='concept':
        similarity_fn = similarity.cos_similarity_cubed_single_concept
    elif args.loss_mode =='sample':
        similarity_fn = similarity.cos_similarity_cubed_single_sample
    elif args.loss_mode =='second':
        similarity_fn = similarity.cos_similarity_cubed_single_secondpower
    elif args.loss_mode =='first_concept':
        similarity_fn = similarity.cos_similarity_cubed_single_firstpower_concept
    elif args.loss_mode =='first_sample':
        similarity_fn = similarity.cos_similarity_cubed_single_firstpower_sample
    else:
        exit()
    # similarity_fn = similarity.cos_similarity_cubed_single_sample
    if args.pose_label is not None:
        # pkl 파일 경로
        file_path = args.pose_label

        # 파일 로드
        with open(os.path.join(file_path,'hard_label_train.pkl'), "rb") as file:
            hard_label_train = pickle.load(file)  # data는 리스트이며, 각 요소는 dict로 구성
        with open(os.path.join(file_path,'hard_label_val.pkl'), "rb") as file:
            hard_label_val = pickle.load(file)  # data는 리스트이며, 각 요소는 dict로 구성
        # 'attribute_label' 키의 value를 추출하고 모두 모음
        train_attribute = [item['attribute_label'] for item in hard_label_train]
        
        # torch.tensor로 변환
        train_result_tensor = torch.tensor(train_attribute,dtype=target_features.dtype)
        
        val_attribute = [item['attribute_label'] for item in hard_label_val]
        
        # torch.tensor로 변환
        
        val_result_tensor = torch.tensor(val_attribute,dtype=target_features.dtype)
        # if not args.use_mlp:
        train_result_tensor[train_result_tensor == 0.] = 0.05
        train_result_tensor[train_result_tensor == 1.] = 0.3
        val_result_tensor[val_result_tensor == 0.] = 0.05
        val_result_tensor[val_result_tensor == 1.] = 0.3
    
    

    if args.use_mlp:
        proj_layer = cbm.ModelOracleCtoY(n_class_attr=2, input_dim=target_features.shape[1],
                                    num_classes=train_result_tensor.shape[1])
        proj_layer = proj_layer.cuda()
        opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
    else:
        proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=train_result_tensor.shape[1],
                                  bias=False).to(args.device)
        # proj_layer.weight.data.zero_()
        # proj_layer.bias.data.zero_()
        opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)    
    indices = [ind for ind in range(len(target_features))]
    
    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(target_features))
    # import pickle
    # result_tensor /= torch.norm(result_tensor,dim=1,keepdim=True)

    # 결과 출력
    for i in range(args.proj_steps):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = proj_layer(target_features[batch].to(args.device).detach())
        # if args.pose_label is not None:
        if args.use_mlp:
            loss = criterion(outs, train_result_tensor[batch].to(args.device).detach())
        else:
            loss = -similarity_fn(train_result_tensor[batch].to(args.device).detach(), outs)
   
        
        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i%10==0 or i==args.proj_steps-1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                if args.use_mlp:
                    val_loss = criterion(val_output, val_result_tensor.to(args.device).detach())
                else:
                    val_loss = -similarity_fn(val_result_tensor.to(args.device).detach(), val_output)
                

                val_loss = torch.mean(val_loss)
            if i==0:
                best_val_loss = val_loss
                best_step = i
                if args.use_mlp:
                    best_weights = {
                    "linear.weight": proj_layer.linear.weight.clone(),
                    "linear.bias": proj_layer.linear.bias.clone(),
                    "linear2.weight": proj_layer.linear2.weight.clone(),
                    "linear2.bias": proj_layer.linear2.bias.clone()
                }
                else:
                    best_weights = proj_layer.weight.clone()

            elif val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_step = i
                if args.use_mlp:
                    best_weights = {
                    "linear.weight": proj_layer.linear.weight.clone(),
                    "linear.bias": proj_layer.linear.bias.clone(),
                    "linear2.weight": proj_layer.linear2.weight.clone(),
                    "linear2.bias": proj_layer.linear2.bias.clone()
                }
                else:
                    best_weights = proj_layer.weight.clone()
            else: #stop if val loss starts increasing
                # break
                # print(loss)
                pass
            print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(i, -loss.cpu(),
                                                                                            -val_loss.cpu()))
        opt.zero_grad()
    if args.use_mlp:
        proj_layer.load_state_dict(best_weights)
    else:
        proj_layer.load_state_dict({"weight":best_weights})
    print("Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))
    W_c = proj_layer.weight[:]
    torch.save(W_c, os.path.join(save_name ,"W_c.pt"))

    # save_classification = os.path.join(save_name,'classification')

    # os.mkdir(save_classification)
    train_classification_layer(args,
                               W_c=W_c,
                               pre_concepts=None,
                               concepts = train_result_tensor[0],
                               target_features=target_features,
                               val_target_features=val_target_features,
                                save_name=save_name,
                                best_val_loss=best_val_loss
                               )


def soft_label(args,target_features, val_target_features,save_name):
    if args.loss_mode =='concept':
        similarity_fn = similarity.cos_similarity_cubed_single_concept
    elif args.loss_mode =='sample':
        similarity_fn = similarity.cos_similarity_cubed_single_sample
    elif args.loss_mode =='second':
        similarity_fn = similarity.cos_similarity_cubed_single_secondpower
    elif args.loss_mode =='first_concept':
        similarity_fn = similarity.cos_similarity_cubed_single_firstpower_concept
    elif args.loss_mode =='first_sample':
        similarity_fn = similarity.cos_similarity_cubed_single_firstpower_sample
    else:
        exit()
    # similarity_fn = similarity.cos_similarity_cubed_single_sample
    if args.pose_label is not None:
        # pkl 파일 경로
        file_path = args.pose_label

        # 파일 로드
        with open(os.path.join(file_path,'soft_label_train.pkl'), "rb") as file:
            hard_label_train = pickle.load(file)  # data는 리스트이며, 각 요소는 dict로 구성
        with open(os.path.join(file_path,'soft_label_val.pkl'), "rb") as file:
            hard_label_val = pickle.load(file)  # data는 리스트이며, 각 요소는 dict로 구성
        # 'attribute_label' 키의 value를 추출하고 모두 모음
        train_attribute = [item['attribute_label'] for item in hard_label_train]
        
        # torch.tensor로 변환
        train_result_tensor = torch.tensor(train_attribute,dtype=target_features.dtype)
        
        val_attribute = [item['attribute_label'] for item in hard_label_val]
        
        # torch.tensor로 변환
        
        val_result_tensor = torch.tensor(val_attribute,dtype=target_features.dtype)

    if args.use_mlp:
        proj_layer = cbm.ModelOracleCtoY(n_class_attr=2, input_dim=target_features.shape[1],
                                    num_classes=train_result_tensor.shape[1])
        proj_layer = proj_layer.cuda()
        opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
    else:
        proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=train_result_tensor.shape[1],
                                  bias=False).to(args.device)
        # proj_layer.weight.data.zero_()
        # proj_layer.bias.data.zero_()
        opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)    
    indices = [ind for ind in range(len(target_features))]
    
    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(target_features))
    # import pickle
    # result_tensor /= torch.norm(result_tensor,dim=1,keepdim=True)

    # 결과 출력
    for i in range(args.proj_steps):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = proj_layer(target_features[batch].to(args.device).detach())
        # if args.pose_label is not None:
        if args.use_mlp:
            loss = criterion(outs, train_result_tensor[batch].to(args.device).detach())
        else:
            loss = -similarity_fn(train_result_tensor[batch].to(args.device).detach(), outs)
   
        
        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i%10==0 or i==args.proj_steps-1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                if args.use_mlp:
                    val_loss = criterion(val_output, val_result_tensor.to(args.device).detach())
                else:
                    val_loss = -similarity_fn(val_result_tensor.to(args.device).detach(), val_output)
                

                val_loss = torch.mean(val_loss)
            if i==0:
                best_val_loss = val_loss
                best_step = i
                if args.use_mlp:
                    best_weights = {
                    "linear.weight": proj_layer.linear.weight.clone(),
                    "linear.bias": proj_layer.linear.bias.clone(),
                    "linear2.weight": proj_layer.linear2.weight.clone(),
                    "linear2.bias": proj_layer.linear2.bias.clone()
                }
                else:
                    best_weights = proj_layer.weight.clone()

            elif val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_step = i
                if args.use_mlp:
                    best_weights = {
                    "linear.weight": proj_layer.linear.weight.clone(),
                    "linear.bias": proj_layer.linear.bias.clone(),
                    "linear2.weight": proj_layer.linear2.weight.clone(),
                    "linear2.bias": proj_layer.linear2.bias.clone()
                }
                else:
                    best_weights = proj_layer.weight.clone()
            else: #stop if val loss starts increasing
                # break
                # print(loss)
                pass
            print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(i, -loss.cpu(),
                                                                                            -val_loss.cpu()))
        opt.zero_grad()
    if args.use_mlp:
        proj_layer.load_state_dict(best_weights)
    else:
        proj_layer.load_state_dict({"weight":best_weights})
    print("Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))
    W_c = proj_layer.weight[:]
    torch.save(W_c, os.path.join(save_name ,"W_c.pt"))

    # save_classification = os.path.join(save_name,'classification')

    # os.mkdir(save_classification)
    train_classification_layer(args,
                               W_c=W_c,
                               pre_concepts=None,
                               concepts = train_result_tensor[0],
                               target_features=target_features,
                               val_target_features=val_target_features,
                                save_name=save_name,
                                best_val_loss=best_val_loss
                               )

def train_cocept_layer(args,concepts, target_features,val_target_features,clip_feature,val_clip_features,save_name):
    similarity_fn = similarity.cos_similarity_cubed_single_concept
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts),
                                bias=False).to(args.device)
    opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)

    indices = [ind for ind in range(len(target_features))]

    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(target_features))
    for i in range(args.proj_steps):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = proj_layer(target_features[batch].to(args.device).detach())
        loss = -similarity_fn(clip_feature[batch].to(args.device).detach(), outs)
        
        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i%50==0 or i==args.proj_steps-1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                val_loss = -similarity_fn(val_clip_features.to(args.device).detach(), val_output)
                val_loss = torch.mean(val_loss)
            if i==0:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
                print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(best_step, -loss.cpu(),
                                                                                            -best_val_loss.cpu()))
                
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
            else: #stop if val loss starts increasing
                break
            print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(i, -loss.cpu(),
                                                                                            -val_loss.cpu()))
        opt.zero_grad()
    proj_layer.load_state_dict({"weight":best_weights})
    print("**Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))

    #delete concepts that are not interpretable
    with torch.no_grad():
        outs = proj_layer(val_target_features.to(args.device).detach())
        sim = similarity_fn(val_clip_features.to(args.device).detach(), outs)
        interpretable = sim > args.interpretability_cutoff
        
    if args.print:
        for i, concept in enumerate(concepts):
            if sim[i]<=args.interpretability_cutoff:
                print("Deleting {}, Iterpretability:{:.3f}".format(concept, sim[i]))
    original_n_concept = len(concepts)

    concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i]]
    print(f"Num concept {len(concepts)} from {original_n_concept}")
    W_c = proj_layer.weight[interpretable]
    torch.save(W_c, os.path.join(save_name ,"W_c.pt"))
    with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write('\n'+concept)
    return W_c,concepts,best_val_loss

def train_cocept_layer_scratch(args,concepts, target_features,val_target_features,clip_feature,val_clip_features,save_name):
    similarity_fn = similarity.cos_similarity_cubed_single
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts),
                                bias=False).to(args.device)
    opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)

    indices = [ind for ind in range(len(target_features))]

    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(target_features))
    for i in range(args.proj_steps):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = proj_layer(target_features[batch].to(args.device).detach())
        loss = -similarity_fn(clip_feature[batch].to(args.device).detach(), outs)
        
        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i%50==0 or i==args.proj_steps-1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                val_loss = -similarity_fn(val_clip_features.to(args.device).detach(), val_output)
                val_loss = torch.mean(val_loss)
            if i==0:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
                print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(best_step, -loss.cpu(),
                                                                                            -best_val_loss.cpu()))
                
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
            else: #stop if val loss starts increasing
                break
        opt.zero_grad()
    proj_layer.load_state_dict({"weight":best_weights})
    print("**Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))

    #delete concepts that are not interpretable
    with torch.no_grad():
        outs = proj_layer(val_target_features.to(args.device).detach())
        sim = similarity_fn(val_clip_features.to(args.device).detach(), outs)
        interpretable = sim > args.interpretability_cutoff
        
    if args.print:
        for i, concept in enumerate(concepts):
            if sim[i]<=args.interpretability_cutoff:
                print("Deleting {}, Iterpretability:{:.3f}".format(concept, sim[i]))
    original_n_concept = len(concepts)

    concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i]]
    print(f"Num concept {len(concepts)} from {original_n_concept}")
    W_c = proj_layer.weight[interpretable]
    torch.save(W_c, os.path.join(save_name ,"W_c.pt"))
    with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write('\n'+concept)
    return W_c,concepts
def train_attention_layer(args,s_concepts,t_concepts, s_W_c, t_W_c,target_features,val_target_features,save_name ):
    
    
    s_proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(s_concepts), bias=False)
    s_proj_layer.load_state_dict({"weight":s_W_c})
    t_proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(t_concepts), bias=False)
    t_proj_layer.load_state_dict({"weight":t_W_c})
    cls_file = data_utils.LABEL_FILES[args.data_set]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
        
    
    sparse_linear_attention = Sparse_attention(1,1,None, len(t_concepts),len(classes))
    sparse_linear_attention.to(args.device)
    optimizer = optim.Adam(
    list(sparse_linear_attention.parameters()),  # 두 모듈의 파라미터를 함께 학습
    lr=0.001)  # Learning rate
    print((sparse_linear_attention.parameters()))
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=target_features.shape[0], gamma=0.1)
    
    d_train = args.data_set + "_train"
    d_val = args.data_set + "_val"
    train_targets = data_utils.get_targets_only(d_train,args)
    val_targets = data_utils.get_targets_only(d_val,args)
    with torch.no_grad():
        s_train_c = s_proj_layer(target_features.detach())
        s_val_c = s_proj_layer(val_target_features.detach())
        s_train_mean = torch.mean(s_train_c, dim=0, keepdim=True)
        s_train_std = torch.std(s_train_c, dim=0, keepdim=True)
        s_train_c -= s_train_mean
        s_train_c /= s_train_std
        s_val_c -= s_train_mean
        s_val_c /= s_train_std
  
        train_y = torch.LongTensor(train_targets)
        val_y = torch.LongTensor(val_targets)

        t_train_c = t_proj_layer(target_features.detach())
        t_val_c = t_proj_layer(val_target_features.detach())
        t_train_mean = torch.mean(t_train_c, dim=0, keepdim=True)
        t_train_std = torch.std(t_train_c, dim=0, keepdim=True)
        t_train_c -= t_train_mean
        t_train_c /= t_train_std
        t_val_c -= t_train_mean
        t_val_c /= t_train_std
    train_ds = TensorDataset(s_train_c,t_train_c,train_y)
    val_ds = TensorDataset(s_val_c,t_val_c,val_y)
    train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True,num_workers=10,pin_memory=True)
    val_loader = DataLoader(val_ds,batch_size=32,shuffle=False,num_workers=10,pin_memory=True)
    loss_fn=nn.CrossEntropyLoss()
    num_epochs = 5
    for epoch in range(num_epochs):
        sparse_linear_attention.train()
        for batch in tqdm(train_dataloader):
        # 학습 코드 (forward, backward, optimizer.step 등)
            x1,x2,y = batch
            x1=x1.to(args.device)
            x2=x2.to(args.device)
            y=y.to(args.device)
            optimizer.zero_grad()
            # 예시 데이터와 로스 계산 과정
            out = sparse_linear_attention(x2,x1)
            loss = loss_fn(out,y)
            loss.backward()
            optimizer.step()
            # 매 epoch마다 스케줄러 업데이트
            # scheduler.step()
            
            # 현재 학습률 출력
        print(f"Epoch {epoch+1}, Loss: {loss}")#, Current learning rate: {scheduler.get_last_lr()}")
        correct = 0
        total = 0
        sparse_linear_attention.eval()
        for batch in val_loader:
            with torch.no_grad():
                #outs = target_model(images.to(device))
                x1,x2,y = batch
                x1=x1.to(args.device)
                x2=x2.to(args.device)
                # y=y.to(args.device)
                outs = sparse_linear_attention(x2,x1)
                pred = torch.argmax(outs, dim=1)
                correct += torch.sum(pred.cpu()==y)
                total += len(y)
        print(f'acc:{correct/total}')

def train_new_classification_layer(args,W_c,pre_concepts,concepts, target_features,val_target_features,save_name):
    proj_layer = torch.nn.Linear(in_features=len(pre_concepts) if args.train_mode=='serial' else target_features.shape[1], out_features=len(concepts), bias=False)
    proj_layer.load_state_dict({"weight":W_c})
    d_train = args.data_set + "_train"
    d_val = args.data_set + "_val"
    train_targets = data_utils.get_targets_only(d_train,args)
    val_targets = data_utils.get_targets_only(d_val,args)
    cls_file = data_utils.LABEL_FILES[args.data_set]
    target_features = target_features.view(target_features.shape[0]//5,5,-1)
    val_target_features = val_target_features.view(val_target_features.shape[0]//5,5,-1)
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    with torch.no_grad():
        train_cs, val_cs = [],[]
        for i in range(5):
            train_c = proj_layer(target_features[:,i,:].detach())
            val_c = proj_layer(val_target_features[:,i,:].detach())
            
            train_mean = torch.mean(train_c, dim=0, keepdim=True)
            train_std = torch.std(train_c, dim=0, keepdim=True)
            
            train_c -= train_mean
            train_c /= train_std
            val_c -= train_mean
            val_c /= train_std
            train_cs.append(train_c.unsqueeze(1))
            val_cs.append(val_c.unsqueeze(1))
        train_cs=torch.cat(train_cs,dim=1)
        val_cs=torch.cat(val_cs,dim=1)

            
        train_y = torch.LongTensor(train_targets)
        val_y = torch.LongTensor(val_targets)
        indexed_train_ds = IndexedTensorDataset(train_cs, train_y)
        val_ds = TensorDataset(val_cs,val_y)
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)
    
    sparse_linear_attention = Sparse_attention(len(concepts),1,None, len(concepts),len(classes))
    sparse_linear_attention.to(args.device)
    optimizer = optim.Adam(
    list(sparse_linear_attention.parameters()),  # 두 모듈의 파라미터를 함께 학습
    lr=0.0001)  # Learning rate
    print((sparse_linear_attention.parameters()))
    loss_fn=nn.CrossEntropyLoss()
    num_epochs = 50
    for epoch in range(num_epochs):
        sparse_linear_attention.train()
        for batch in (indexed_train_loader):
        # 학습 코드 (forward, backward, optimizer.step 등)
            x1,y,_ = batch
            x1=x1.to(args.device)
            y=y.to(args.device)
            optimizer.zero_grad()
            # 예시 데이터와 로스 계산 과정
            out,_ = sparse_linear_attention(x1)
            loss = loss_fn(out,y)
            loss.backward()
            optimizer.step()
            # 매 epoch마다 스케줄러 업데이트
            # scheduler.step()
            
            # 현재 학습률 출력
        print(f"Epoch {epoch+1}, Loss: {loss}")#, Current learning rate: {scheduler.get_last_lr()}")
        correct = 0
        total = 0
        sparse_linear_attention.eval()
        for batch in val_loader:
            with torch.no_grad():
                #outs = target_model(images.to(device))
                x1,y = batch
                x1=x1.to(args.device)
                # x2=x2.to(args.device)
                # y=y.to(args.device)
                outs ,_= sparse_linear_attention(x1)
                pred = torch.argmax(outs, dim=1)
                correct += torch.sum(pred.cpu()==y)
                total += len(y)
        print(f'acc:{correct/total}')
    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(sparse_linear_attention.state_dict(), os.path.join(save_name, "W.pt"))
def train_classification_multiview(args,W_c,pre_concepts,concepts, target_features,val_target_features,save_name):
    proj_layer = torch.nn.Linear(in_features=len(pre_concepts) if args.train_mode=='serial' else target_features.shape[1], out_features=len(concepts), bias=False)
    proj_layer.load_state_dict({"weight":W_c})
    d_train = args.data_set + "_train"
    d_val = args.data_set + "_val"
    train_targets = data_utils.get_targets_only(d_train,args)
    val_targets = data_utils.get_targets_only(d_val,args)
    cls_file = data_utils.LABEL_FILES[args.data_set]
    target_features = target_features.view(target_features.shape[0]//5,5,-1)
    # val_target_features = val_target_features.view(val_target_features.shape[0]//5,5,-1)
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    with torch.no_grad():
        train_cs=[]
        val_c = proj_layer(val_target_features.detach())

        for i in range(5):
            train_c = proj_layer(target_features[:,i,:].detach())
            # val_c = proj_layer(val_target_features[:,i,:].detach())
            
            train_mean = torch.mean(train_c, dim=0, keepdim=True)
            train_std = torch.std(train_c, dim=0, keepdim=True)
            
            train_c -= train_mean
            train_c /= train_std
            train_cs.append(train_c.unsqueeze(1))

        train_cs=torch.cat(train_cs,dim=1)
        # val_cs=torch.cat(val_cs,dim=1)
        ind2_train_c = proj_layer(target_features[:,2,:].detach())
        
        train_mean = torch.mean(ind2_train_c, dim=0, keepdim=True)
        train_std = torch.std(ind2_train_c, dim=0, keepdim=True)
        val_c -= train_mean
        val_c /= train_std
            
        train_y = torch.LongTensor(train_targets)
        val_y = torch.LongTensor(val_targets)
        indexed_train_ds = IndexedTensorDataset(train_cs, train_y)
        val_ds = TensorDataset(val_c,val_y)
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)
    # Make linear model and zero initialize
    linear = torch.nn.Linear(train_c.shape[1],len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    
    STEP_SIZE = 0.01
    ALPHA = 0.99
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = args.lam

    # Solve the GLM path
    #concept layer to classification
    output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.n_iters, ALPHA, epsilon=1, k=1,
                    val_loader=val_loader, do_zero=False, metadata=metadata, n_ex=len(target_features), n_classes = len(classes),verbose=10,multiview=True)
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']
    

    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))
    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        out_dict = {}
        for key in ('lam', 'lr', 'alpha', 'time'):
            out_dict[key] = float(output_proj['path'][0][key])
        out_dict['metrics'] = output_proj['path'][0]['metrics']
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict['sparsity'] = {"Non-zero weights":nnz, "Total weights":total, "Percentage non-zero":nnz/total}
        json.dump(out_dict, f, indent=2)
def train_classification_layer(args=None,W_c=None,pre_concepts=None,concepts=None, target_features=None,val_target_features=None,save_name=None,joint=None,best_val_loss=None):
    # cls_file = data_utils.LABEL_FILES[args.data_set]
    cls_file = os.path.join(args.video_anno_path, 'class_list.txt')
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    assert args.nb_classes == len(classes), f"Error: args.nb_classes ({args.nb_classes}) != len(classes) ({len(classes)})"
    
    
    train_video_dataset, _ = datasets.build_dataset(True, False, args)
    val_video_dataset,_ =   datasets.build_dataset(False, False, args)
    # d_train = args.data_set + "_train"
    # d_val = args.data_set + "_val"
    train_targets = train_video_dataset.label_array
    val_targets = val_video_dataset.label_array
    train_y = torch.LongTensor(train_targets)
    val_y = torch.LongTensor(val_targets)
        
    proj_layer = torch.nn.Linear(target_features.shape[1], out_features=len(concepts), bias=False)
    proj_layer.load_state_dict({"weight":W_c})
    with torch.no_grad():
        train_c = proj_layer(target_features.detach())
        val_c = proj_layer(val_target_features.detach())
        
        train_mean = torch.mean(train_c, dim=0, keepdim=True)
        train_std = torch.std(train_c, dim=0, keepdim=True)
        
        train_c -= train_mean
        train_c /= train_std

        val_c -= train_mean
        val_c /= train_std

        
        


    indexed_train_ds = IndexedTensorDataset(train_c, train_y)
    val_ds = TensorDataset(val_c,val_y)
        
    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)

    # Make linear model and zero initialize
    linear = torch.nn.Linear(train_c.shape[1],len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    
    STEP_SIZE = 0.05
    ALPHA = 0.99
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = args.lam

    # Solve the GLM path
    #concept layer to classification
    output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.n_iters, ALPHA, epsilon=1, k=1,
                    val_loader=val_loader, do_zero=False, metadata=metadata, n_ex=len(target_features), n_classes = len(classes),verbose=500)
    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']
    
    if joint is None:
        torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
        torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))
    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        out_dict = {}
        for key in ('lam', 'lr', 'alpha', 'time'):
            out_dict[key] = float(output_proj['path'][0][key])
        out_dict['metrics'] = output_proj['path'][0]['metrics']
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict['sparsity'] = {"Non-zero weights":nnz, "Total weights":total, "Percentage non-zero":nnz/total}
        out_dict['concept_layer_best_loss'] = -(best_val_loss.item())
        json.dump(out_dict, f, indent=2)



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int=1, n_head: int=1, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        # self.ln_1 = LayerNorm(d_model)
        # self.mlp = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(d_model, d_model * 4)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(d_model * 4, d_model))
        # ]))
        # self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, x2:torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        return self.attn(x, x2, x2, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor,x2:torch.Tensor):
        x = self.attention(x,x2)+x
        # x = x + self.mlp((x))
        return x


class Sparse_attention(nn.Module):
    def __init__(self, d_model: int=1, n_head: int=1, attn_mask: torch.Tensor = None, linear_in=1, linear_out=1):
        super().__init__()
        
        self.attention = ResidualAttentionBlock(d_model,n_head,attn_mask)
        self.spatial_token = nn.Parameter(torch.randn(1,linear_in))
        self.sparse_linear = SparseLinear(linear_in,linear_out,bias=True,sparsity=0.9,alpha=0.1)
        
    def forward(self,x):
        # x1 = x1.unsqueeze(-1)
        # x2 = x2.unsqueeze(-1)
        # spatial_token = self.spatial_token.expand(1,x1.shape[0],-1)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # x2 = x2.permute(1, 0, 2)  # NLD -> LND
        
        x= self.attention(x,x)
        x = x.permute(1, 0, 2)  #NLD
        # final_feat=x.mean(1)
        x = self.sparse_linear(x)
        x = x.mean(1)
        return x,None#final_feat
    def forward_feat(self,x):
    
        spatial_token = self.spatial_token.expand(1,x.shape[0],-1)
        x1 = x.permute(1, 0, 2)  # NLD -> LND
        # x2 = x2.permute(1, 0, 2)  # NLD -> LND

        sub_act= self.attention(spatial_token,x1).permute(1,0,2).squeeze(1)#N,1,D
        mean_act = x[:,2,:]#.mean(1)  #NLD
        # final_feat=x.squeeze(1)
        x = self.sparse_linear(mean_act)
        # sub_x = self.sparse_linear(sub_act)
        final = x#+sub_x*0.1
        return final,sub_act
    def forward_feat_ensemble(self,x):
    
        spatial_token = self.spatial_token.expand(1,x.shape[0],-1)
        x1 = x.permute(1, 0, 2)  # NLD -> LND
        # x2 = x2.permute(1, 0, 2)  # NLD -> LND

        sub_act= self.attention(spatial_token,x1).permute(1,0,2).squeeze(1)#N,1,D
        # mean_act = x.mean(1)  #NLD
        # final_feat=x.squeeze(1)
        x = self.sparse_linear(x)
        sub_x = self.sparse_linear(sub_act)
        final = x.mean(1)+(0.1*sub_x)
        return final,sub_act
    # def forward_feat_self(self,x1):
    
    #     x=x1[]
    #     return x,final_feat