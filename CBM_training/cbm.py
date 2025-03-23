# from modeling_finetune import MLP
import os
import json
import torch
import data_utils
from sparselinear import SparseLinear
import torch.nn as nn
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
class CBM_model_double(torch.nn.Module):
    def __init__(self, backbone_name, a_W_c, b_W_c, W_g, b_g, proj_mean, proj_std, device="cuda", args=None):
        super().__init__()
        self.backbone_model, _ = data_utils.get_target_model(backbone_name, device, args)
        self.backbone_model.eval()

        if "clip" in backbone_name:
            self.backbone = self.backbone_model
        elif "cub" in backbone_name:
            self.backbone = lambda x: self.backbone_model.features(x)
        elif "vmae" or 'AIM' in backbone_name:
            self.backbone = lambda x: self.backbone_model.forward_features(x)
        else:
            self.backbone = torch.nn.Sequential(*list(self.backbone_model.children())[:-1])

        self.a_proj_layer = torch.nn.Linear(a_W_c.shape[1], a_W_c.shape[0], bias=False).to(device)
        self.a_proj_layer.load_state_dict({"weight": a_W_c})
        self.b_proj_layer = torch.nn.Linear(b_W_c.shape[1], b_W_c.shape[0], bias=False).to(device)
        self.b_proj_layer.load_state_dict({"weight": b_W_c})

        self.a_proj_mean = proj_mean[0]
        self.a_proj_std = proj_std[0]
        self.b_proj_mean = proj_mean[1]
        self.b_proj_std = proj_std[1]

        self.final = torch.nn.Linear(W_g.shape[1], W_g.shape[0]).to(device)
        self.final.load_state_dict({"weight": W_g, "bias": b_g})
        self.concepts = None

    def get_feature(self, x):
        backbone_feat = self.backbone_model(x, only_feat=True)
        backbone_feat = torch.flatten(backbone_feat, 1)
        a_x = self.a_proj_layer(backbone_feat)
        a_proj_c = (a_x - self.a_proj_mean) / self.a_proj_std

        b_x = self.b_proj_layer(backbone_feat)
        b_proj_c = (b_x - self.b_proj_mean) / self.b_proj_std

        proj_c = torch.cat([a_proj_c, b_proj_c], dim=1)
        return backbone_feat, proj_c

    def forward(self, x, masking_sp=False):
        x = self.backbone_model(x, True)
        x = torch.flatten(x, 1)
        a_x = self.a_proj_layer(x)
        a_proj_c = (a_x - self.a_proj_mean) / self.a_proj_std

        b_x = self.b_proj_layer(x)
        b_proj_c = (b_x - self.b_proj_mean) / self.b_proj_std

        proj_c = torch.cat([a_proj_c, b_proj_c], dim=1)
        x = self.final(proj_c)
        return x, proj_c
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
def load_cbm_dynamic(load_dir, device,args):


    train_mode = args.train_mode


    if len(train_mode) == 1:
        load_sub_dir = os.path.join(load_dir,train_mode[0])
        load_cbm(load_sub_dir,device,args)
    if len(train_mode) == 2:
        load_cbm_double(load_dir,device,args)
    elif len(train_mode) ==3:
        return load_cbm_triple(load_dir,device,args)
    else:
        raise ValueError(f"Unsupported train_mode: {train_mode}")



def load_cbm(load_dir, device,args):

    W_c = torch.load(os.path.join(load_dir ,"W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir, "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir, "b_g.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir, "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir, "proj_std.pt"), map_location=device)

    model = CBM_model(args.backbone, W_c, W_g, b_g, proj_mean, proj_std, device,args)
    return model

def load_cbm_double(load_dir, device,args=None):
    # with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
    #     args = json.load(f)
    concepts = args.train_mode

    a_W_c = torch.load(os.path.join(load_dir,concepts[0],"W_c.pt"), map_location=device)
    a_proj_mean = torch.load(os.path.join(load_dir,concepts[0], "proj_mean.pt"), map_location=device)
    a_proj_std = torch.load(os.path.join(load_dir,concepts[0], "proj_std.pt"), map_location=device)
    
    b_W_c = torch.load(os.path.join(load_dir,concepts[1],"W_c.pt"), map_location=device)
    b_proj_mean = torch.load(os.path.join(load_dir,concepts[1], "proj_mean.pt"), map_location=device)
    b_proj_std = torch.load(os.path.join(load_dir,concepts[1], "proj_std.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir,'aggregated', "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir,'aggregated', "b_g.pt"), map_location=device)


    model = CBM_model_double(args.backbone, a_W_c,b_W_c, W_g, b_g, [a_proj_mean,b_proj_mean], [a_proj_std,b_proj_std], device,args)
    return model
def load_cbm_triple(load_dir, device,args=None):
    # with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
    #     args = json.load(f)
    concepts = args.train_mode

    a_W_c = torch.load(os.path.join(load_dir,concepts[0],"W_c.pt"), map_location=device)
    a_proj_mean = torch.load(os.path.join(load_dir,concepts[0], "proj_mean.pt"), map_location=device)
    a_proj_std = torch.load(os.path.join(load_dir,concepts[0], "proj_std.pt"), map_location=device)
    
    b_W_c = torch.load(os.path.join(load_dir,concepts[1],"W_c.pt"), map_location=device)
    b_proj_mean = torch.load(os.path.join(load_dir,concepts[1], "proj_mean.pt"), map_location=device)
    b_proj_std = torch.load(os.path.join(load_dir,concepts[1], "proj_std.pt"), map_location=device)
    
    c_W_c = torch.load(os.path.join(load_dir,concepts[2],"W_c.pt"), map_location=device)
    c_proj_mean = torch.load(os.path.join(load_dir,concepts[2], "proj_mean.pt"), map_location=device)
    c_proj_std = torch.load(os.path.join(load_dir,concepts[2], "proj_std.pt"), map_location=device)
    
    W_g = torch.load(os.path.join(load_dir,'aggregated', "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir,'aggregated', "b_g.pt"), map_location=device)


    model = CBM_model_triple(args.backbone, a_W_c,b_W_c,c_W_c, W_g, b_g, [a_proj_mean,b_proj_mean,c_proj_mean], [a_proj_std,b_proj_std,c_proj_std], device,args)
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
