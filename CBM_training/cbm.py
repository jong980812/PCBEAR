# from modeling_finetune import MLP
import os
import json
import torch
import data_utils
from sparselinear import SparseLinear
import torch.nn as nn
from learning_concept_layer import Sparse_attention
class CBM_model(torch.nn.Module):
    def __init__(self, backbone_name, W_c, W_g, b_g, proj_mean, proj_std, device="cuda",args=None):
        super().__init__()
        model, _ = data_utils.get_target_model(backbone_name, device,args)
        model.eval()
        #remove final fully connected layer
        if "clip" in backbone_name:
            self.backbone = model
        elif "cub" in backbone_name:
            self.backbone = lambda x: model.features(x)
        elif "vmae" or 'AIM' in backbone_name:
            self.backbone = lambda x: model.forward_features(x)
        else:
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
            
        self.proj_layer = torch.nn.Linear(in_features=W_c.shape[1], out_features=W_c.shape[0], bias=False).to(device)
        self.proj_layer.load_state_dict({"weight":W_c})
            
        self.proj_mean = proj_mean
        self.proj_std = proj_std
        
        self.final = torch.nn.Linear(in_features = W_g.shape[1], out_features=W_g.shape[0]).to(device)
        self.final.load_state_dict({"weight":W_g, "bias":b_g})
        self.concepts = None
    def get_feature(self,x):
        backbone_feat = self.backbone(x)
        backbone_feat = torch.flatten(backbone_feat, 1)
        x = self.proj_layer(backbone_feat)
        proj_c = (x-self.proj_mean)/self.proj_std
        final = self.final(proj_c)
        
        return backbone_feat,proj_c,final
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.proj_layer(x)
        proj_c = (x-self.proj_mean)/self.proj_std
        x = self.final(proj_c)
        return x, proj_c
class CBM_model_joint(torch.nn.Module):
    def __init__(self, backbone_name, s_W_c,t_W_c, W_g, b_g, proj_mean, proj_std, device="cuda",args=None):
        super().__init__()
        model, _ = data_utils.get_target_model(backbone_name, device,args)
        model.eval()
        #remove final fully connected layer
        if "clip" in backbone_name:
            self.backbone = model
        elif "cub" in backbone_name:
            self.backbone = lambda x: model.features(x)
        elif "vmae" or 'AIM' in backbone_name:
            self.backbone = lambda x: model.forward_features(x)
        else:
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
        st_W_c = torch.cat([s_W_c,t_W_c],dim=0)
            
        self.proj_layer = torch.nn.Linear(in_features=st_W_c.shape[1], out_features=st_W_c.shape[0], bias=False).to(device)
        self.proj_layer.load_state_dict({"weight":st_W_c})
        self.proj_mean = proj_mean
        self.proj_std = proj_std
        
        self.final = torch.nn.Linear(in_features = W_g.shape[1], out_features=W_g.shape[0]).to(device)
        self.final.load_state_dict({"weight":W_g, "bias":b_g})
        self.concepts = None
    def get_feature(self,x):
        backbone_feat = self.backbone(x)
        backbone_feat = torch.flatten(backbone_feat, 1)
        x = self.proj_layer(backbone_feat)
        proj_c = (x-self.proj_mean)/self.proj_std
        final = self.final(proj_c)
        
        return backbone_feat,proj_c,final
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.proj_layer(x)
        proj_c = (x-self.proj_mean)/self.proj_std
        x = self.final(proj_c)
        return x, proj_c
class CBM_model_triple(torch.nn.Module):
    def __init__(self, backbone_name, s_W_c,t_W_c,p_W_c, W_g, b_g, proj_mean, proj_std, device="cuda",args=None):
        super().__init__()
        self.backbone_model, _ = data_utils.get_target_model(backbone_name, device,args)
        self.backbone_model.eval()
        #remove final fully connected layer
        if "clip" in backbone_name:
            self.backbone = self.backbone_model
        elif "cub" in backbone_name:                    
            self.backbone = lambda x: self.backbone_model.features(x)
        elif "vmae" or 'AIM' in backbone_name:
            self.backbone = lambda x: self.backbone_model.forward_features(x)
        else:
            self.backbone = torch.nn.Sequential(*list(self.backbone_model.children())[:-1])
        # stp_W_c = torch.cat([s_W_c,t_W_c,p_W_c],dim=0)
            
        self.s_proj_layer = torch.nn.Linear(in_features=s_W_c.shape[1], out_features=s_W_c.shape[0], bias=False).to(device)
        self.s_proj_layer.load_state_dict({"weight":s_W_c})
        self.t_proj_layer = torch.nn.Linear(in_features=t_W_c.shape[1], out_features=t_W_c.shape[0], bias=False).to(device)
        self.t_proj_layer.load_state_dict({"weight":t_W_c})
        self.p_proj_layer = torch.nn.Linear(in_features=p_W_c.shape[1], out_features=p_W_c.shape[0], bias=False).to(device)
        self.p_proj_layer.load_state_dict({"weight":p_W_c})
        self.s_proj_mean = proj_mean[0]
        self.s_proj_std = proj_std[0]
        self.t_proj_mean = proj_mean[1]
        self.t_proj_std = proj_std[1]
        self.p_proj_mean = proj_mean[2]
        self.p_proj_std = proj_std[2]
        self.final = torch.nn.Linear(in_features = W_g.shape[1], out_features=W_g.shape[0]).to(device)
        self.final.load_state_dict({"weight":W_g, "bias":b_g})
        self.concepts = None
    def get_feature(self,x):
        backbone_feat = self.backbone_model(x,only_feat= True)
        backbone_feat = torch.flatten(backbone_feat, 1)
        s_x = self.s_proj_layer(backbone_feat)
        s_proj_c = (s_x-self.s_proj_mean)/self.s_proj_std
        
        t_x = self.t_proj_layer(backbone_feat)
        t_proj_c = (t_x-self.t_proj_mean)/self.t_proj_std
        
        p_x = self.p_proj_layer(backbone_feat)
        p_proj_c = (p_x-self.p_proj_mean)/self.p_proj_std
        
        proj_c = torch.cat([s_proj_c,t_proj_c,p_proj_c],dim=1)

        
        return backbone_feat,proj_c
    def forward(self, x,masking_sp=False):
        x = self.backbone_model(x,True)
        x = torch.flatten(x, 1)
        # x = self.proj_layer(x)
        s_x = self.s_proj_layer(x)
        s_proj_c = (s_x-self.s_proj_mean)/self.s_proj_std
        t_x = self.t_proj_layer(x)
        t_proj_c = (t_x-self.t_proj_mean)/self.t_proj_std
        
        p_x = self.p_proj_layer(x)
        p_proj_c = (p_x-self.p_proj_mean)/self.p_proj_std
        
        proj_c = torch.cat([s_proj_c,t_proj_c,p_proj_c],dim=1)
        x = self.final(proj_c)
        return x, proj_c
class CBM_model_serial(torch.nn.Module):
    def __init__(self, backbone_name, pre_W_c, post_W_c, W_g, b_g, pre_mean,pre_std,post_mean, post_std, device="cuda",args=None):
        super().__init__()
        model, _ = data_utils.get_target_model(backbone_name, device,args)
        #remove final fully connected layer
        if "clip" in backbone_name:
            self.backbone = model
        elif "cub" in backbone_name:
            self.backbone = lambda x: model.features(x)
        elif "vmae" or 'AIM' in backbone_name:
            self.backbone = lambda x: model.forward_features(x)
        else:
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
            
        self.pre_proj_layer = torch.nn.Linear(in_features=pre_W_c.shape[1], out_features=pre_W_c.shape[0], bias=False).to(device)
        self.pre_proj_layer.load_state_dict({"weight":pre_W_c})

        self.post_proj_layer = torch.nn.Linear(in_features=post_W_c.shape[1], out_features=post_W_c.shape[0], bias=False).to(device)
        self.post_proj_layer.load_state_dict({"weight":post_W_c})
            
        self.pre_proj_mean = pre_mean
        self.pre_proj_std = pre_std
        
        self.post_proj_mean = post_mean
        self.post_proj_std = post_std
        
        self.final = torch.nn.Linear(in_features = W_g.shape[1], out_features=W_g.shape[0]).to(device)
        self.final.load_state_dict({"weight":W_g, "bias":b_g})
        self.concepts = None
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.pre_proj_layer(x)
        pre_proj_c = (x-self.pre_proj_mean)/self.pre_proj_std
        
        post_proj_c = self.post_proj_layer(pre_proj_c)
        post_proj_c = (post_proj_c-self.post_proj_mean)/self.post_proj_std
        x = self.final(post_proj_c)
        return x, pre_proj_c,post_proj_c
class standard_model(torch.nn.Module):
    def __init__(self, backbone_name, W_g, b_g, proj_mean, proj_std, device="cuda"):
        super().__init__()
        model, _ = data_utils.get_target_model(backbone_name, device)
        #remove final fully connected layer
        if "clip" in backbone_name:
            self.backbone = model
        elif "cub" in backbone_name:
            self.backbone = lambda x: model.features(x)
        else:
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
            
        self.proj_mean = proj_mean
        self.proj_std = proj_std
        
        self.final = torch.nn.Linear(in_features = W_g.shape[1], out_features=W_g.shape[0]).to(device)
        self.final.load_state_dict({"weight":W_g, "bias":b_g})
        self.concepts = None
        
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        proj_c = (x-self.proj_mean)/self.proj_std
        x = self.final(proj_c)
        return x, proj_c

class CBM_model_attention(torch.nn.Module):
    def __init__(self, backbone_name, W_c, W, proj_mean, proj_std, device="cuda",args=None, d_model=None, n_classes=None,):
        super().__init__()
        model, _ = data_utils.get_target_model(backbone_name, device,args)
        model.eval()
        #remove final fully connected layer
        if "clip" in backbone_name:
            self.backbone = model
        elif "cub" in backbone_name:
            self.backbone = lambda x: model.features(x)
        elif "vmae" or 'AIM' in backbone_name:
            self.backbone = lambda x: model.forward_features(x)
        else:
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
        self.sparse_linear_attention = Sparse_attention(d_model,1,None, d_model,n_classes).to(device)
        self.sparse_linear_attention.load_state_dict(W)
        self.sparse_linear_attention.eval()
        
        self.proj_layer = torch.nn.Linear(in_features=W_c.shape[1], out_features=W_c.shape[0], bias=False).to(device)
        self.proj_layer.load_state_dict({"weight":W_c})
            
        self.proj_mean = proj_mean
        self.proj_std = proj_std
        
    def get_feature(self,x):
        backbone_feat = self.backbone(x)
        backbone_feat = torch.flatten(backbone_feat, 1)
        x = self.proj_layer(backbone_feat)
        proj_c = (x-self.proj_mean)/self.proj_std
        x = self.sparse_linear_attention(proj_c)
        
        return x,proj_c
    def forward(self, x):
        num_t = x.shape[0]
        features = []
        for i in range(num_t):
            feat = self.backbone(x[i])
            feat = torch.flatten(feat, 1)
            feat = self.proj_layer(feat)
            proj_c = (feat-self.proj_mean)/self.proj_std
            features.append(proj_c)
        proj_c = torch.cat(features,dim=0)#
        x,sub_act = self.sparse_linear_attention(proj_c.unsqueeze(0))
        return x, proj_c,sub_act
def load_cbm_two_stream_attention(load_dir, device,argument):
    print('**********Load Spatio model***************')
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)
    W_c = torch.load(os.path.join(load_dir,'spatial' ,"W_c.pt"), map_location=device)
    W = torch.load(os.path.join(load_dir,'spatial', "W.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir,'spatial', "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir,'spatial', "proj_std.pt"), map_location=device)
    s_model = CBM_model_attention(args['backbone'], W_c, W, proj_mean, proj_std, device,argument,proj_mean.shape[1],args['nb_classes'])
    print('**********Load Temporal model***************')
    W_c = torch.load(os.path.join(load_dir,'temporal' ,"W_c.pt"), map_location=device)
    W = torch.load(os.path.join(load_dir,'temporal', "W.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir,'temporal', "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir,'temporal', "proj_std.pt"), map_location=device)
    t_model = CBM_model_attention(args['backbone'], W_c, W, proj_mean, proj_std, device,argument,proj_mean.shape[1],args['nb_classes'])
    return s_model,t_model
def load_cbm(load_dir, device,argument):
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)

    W_c = torch.load(os.path.join(load_dir ,"W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir, "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir, "b_g.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir, "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir, "proj_std.pt"), map_location=device)

    model = CBM_model(args['backbone'], W_c, W_g, b_g, proj_mean, proj_std, device,argument)
    return model
def load_cbm_joint(load_dir, device,argument):
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)

    s_W_c = torch.load(os.path.join(load_dir,'spatial',"W_c.pt"), map_location=device)
    t_W_c = torch.load(os.path.join(load_dir,'temporal',"W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir,'spatio_temporal', "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir,'spatio_temporal', "b_g.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir,'spatio_temporal', "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir,'spatio_temporal', "proj_std.pt"), map_location=device)

    model = CBM_model_joint(args['backbone'], s_W_c,t_W_c, W_g, b_g, proj_mean, proj_std, device,argument)
    return model,model
def load_cbm_triple(load_dir, device,argument=None):
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)

    s_W_c = torch.load(os.path.join(load_dir,'spatial',"W_c.pt"), map_location=device)
    s_proj_mean = torch.load(os.path.join(load_dir,'spatial', "proj_mean.pt"), map_location=device)
    s_proj_std = torch.load(os.path.join(load_dir,'spatial', "proj_std.pt"), map_location=device)
    
    t_W_c = torch.load(os.path.join(load_dir,'temporal',"W_c.pt"), map_location=device)
    t_proj_mean = torch.load(os.path.join(load_dir,'temporal', "proj_mean.pt"), map_location=device)
    t_proj_std = torch.load(os.path.join(load_dir,'temporal', "proj_std.pt"), map_location=device)
    
    p_W_c = torch.load(os.path.join(load_dir,'place',"W_c.pt"), map_location=device)
    p_proj_mean = torch.load(os.path.join(load_dir,'place', "proj_mean.pt"), map_location=device)
    p_proj_std = torch.load(os.path.join(load_dir,'place', "proj_std.pt"), map_location=device)
    
    W_g = torch.load(os.path.join(load_dir,'spatio_temporal_place', "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir,'spatio_temporal_place', "b_g.pt"), map_location=device)


    model = CBM_model_triple(args['backbone'], s_W_c,t_W_c,p_W_c, W_g, b_g, [s_proj_mean,t_proj_mean,p_proj_mean], [s_proj_std,t_proj_std,p_proj_std], device,argument)
    return model,model
def load_cbm_serial(load_dir, device,argument):
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)

    s_W_c = torch.load(os.path.join(load_dir,"spatial","W_c.pt"), map_location=device)
    t_W_c = torch.load(os.path.join(load_dir,"temporal","W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir,"classification", "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir,"classification", "b_g.pt"), map_location=device)

    pre_proj_mean = torch.load(os.path.join(load_dir,'temporal' ,"proj_mean.pt"), map_location=device)
    pre_proj_std = torch.load(os.path.join(load_dir,'temporal', "proj_std.pt"), map_location=device)
    post_proj_mean = torch.load(os.path.join(load_dir,'classification' ,"proj_mean.pt"), map_location=device)
    post_proj_std = torch.load(os.path.join(load_dir,'classification', "proj_std.pt"), map_location=device)

    model = CBM_model_serial(args['backbone'], s_W_c,t_W_c, W_g, b_g,pre_proj_mean,pre_proj_std, post_proj_mean, post_proj_std, device,argument)
    return model
def load_cbm_two_stream(load_dir, device,argument):
    print('**********Load Spatio model***************')
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)
    W_c = torch.load(os.path.join(load_dir,'spatial' ,"W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir,'spatial', "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir,'spatial', "b_g.pt"), map_location=device)
    proj_mean = torch.load(os.path.join(load_dir,'spatial', "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir,'spatial', "proj_std.pt"), map_location=device)
    s_model = CBM_model(args['backbone'], W_c, W_g, b_g, proj_mean, proj_std, device,argument)
    print('**********Load Temporal model***************')
    W_c = torch.load(os.path.join(load_dir,'temporal' ,"W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir,'temporal', "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir,'temporal', "b_g.pt"), map_location=device)
    proj_mean = torch.load(os.path.join(load_dir,'temporal', "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir,'temporal', "proj_std.pt"), map_location=device)
    t_model = CBM_model(args['backbone'], W_c, W_g, b_g, proj_mean, proj_std, device,argument)
    return s_model,t_model
def load_std(load_dir, device):
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)

    W_g = torch.load(os.path.join(load_dir, "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir, "b_g.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir, "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir, "proj_std.pt"), map_location=device)

    model = standard_model(args['backbone'], W_g, b_g, proj_mean, proj_std, device)
    model.eval()
    return model

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, expand_dim):
        super(MLP, self).__init__()
        self.expand_dim = expand_dim
        if self.expand_dim is not None:
            self.linear = nn.Linear(input_dim, expand_dim)
            self.activation = torch.nn.ReLU()
            self.linear2 = nn.Linear(expand_dim, num_classes) #softmax is automatically handled by loss function
        else:
            self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        if self.expand_dim is not None:
            x = self.activation(x)
            x = self.linear2(x)
        return x
def ModelOracleCtoY(n_class_attr, input_dim, num_classes):
    # X -> C part is separate, this is only the C -> Y part
    import math
    expand_dim = int(math.sqrt(input_dim * num_classes))
    if n_class_attr == 3:
        model = MLP(input_dim=input_dim * n_class_attr, num_classes=num_classes, expand_dim=expand_dim)
    else:
        model = MLP(input_dim=input_dim, num_classes=num_classes, expand_dim=expand_dim)
    return model
