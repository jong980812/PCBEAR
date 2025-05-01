import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from video_dataloader import datasets
import torch.nn.init as init
import json

def train_non_cbm(args, backbone_features, val_backbone_features, save_name):
    cls_file = os.path.join(args.video_anno_path, 'class_list.txt')
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    assert args.nb_classes == len(classes), f"Error: args.nb_classes ({args.nb_classes}) != len(classes) ({len(classes)})"
    
    
    train_video_dataset, _ = datasets.build_dataset(True, False, args)
    val_video_dataset,_ =   datasets.build_dataset(False, False, args)
    train_targets = train_video_dataset.label_array
    val_targets = val_video_dataset.label_array
    train_y = torch.LongTensor(train_targets)
    val_y = torch.LongTensor(val_targets)
    
    
    # indexed_train_ds = IndexedTensorDataset(backbone_features, train_y)
    # val_ds = TensorDataset(val_backbone_features,val_y)
        
    # indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.no_cbm_batch_size, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=args.no_cbm_batch_size, shuffle=False)
    
    
    if args.mode_no_cbm == "only_cls":
        train_only_cls(args, backbone_features, val_backbone_features, train_y, val_y, save_name)
        return
    elif args.mode_no_cbm == "only_sparse_cls":
        train_only_sparse_cls(args, backbone_features, val_backbone_features, train_y, val_y, save_name)
        return
    elif args.mode_no_cbm == "dump_linear_cls":
        assert args.dump_concept_num is not None and isinstance(args.dump_concept_num, int) and args.dump_concept_num > 0, \
            "args.dump_concept_num must be a positive integer when args.no_cbm is set."
        train_dump_linear_cls(args, backbone_features, val_backbone_features, train_y, val_y, save_name)
        return
    elif args.mode_no_cbm == "dump_linear_sparse_cls":
        assert args.dump_concept_num is not None and isinstance(args.dump_concept_num, int) and args.dump_concept_num > 0, \
            "args.dump_concept_num must be a positive integer when args.no_cbm is set."
        train_dump_linear_sparse_cls(args, backbone_features, val_backbone_features, train_y, val_y, save_name)
        return
    else:
        raise ValueError(f"Unsupported mode_no_cbm: {args.mode_no_cbm}")


# New function: train_only_cls
def train_only_cls(args, backbone_features, val_backbone_features, train_y, val_y, save_name):
    from torch.utils.data import TensorDataset, DataLoader
    import torch.optim as optim
    import torch
    import os

    train_ds = TensorDataset(backbone_features, train_y)
    val_ds = TensorDataset(val_backbone_features, val_y)

    train_loader = DataLoader(train_ds, batch_size=args.no_cbm_batchsize, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.no_cbm_batchsize, shuffle=False, num_workers=4, pin_memory=True)

    model = torch.nn.Linear(backbone_features.shape[1], args.nb_classes).to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.no_cbm_lr, weight_decay=args.no_cbm_weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.no_cbm_epochs // 3), gamma=0.1)

    best_acc = 0
    best_model_state = None

    # Logging
    log_file = open(os.path.join(save_name, "train_only_cls_log.txt"), "w")

    for epoch in range(args.no_cbm_epochs):
        model.train()
        correct = 0
        total = 0
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        scheduler.step()
        train_loss_epoch = total_loss / total if total > 0 else 0.0
        train_acc_epoch = correct / total if total > 0 else 0.0

        # validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(args.device)
                y = y.to(args.device)
                out = model(x)
                loss = criterion(out, y)
                preds = torch.argmax(out, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                val_loss += loss.item() * y.size(0)
                val_total += y.size(0)
        acc = correct / total if total > 0 else 0.0
        val_loss_epoch = val_loss / val_total if val_total > 0 else 0.0
        print(f"[only_cls] Epoch {epoch}: val acc {acc:.4f}")
        log_file.write(f"Epoch {epoch}: train_loss={train_loss_epoch:.4f}, train_acc={train_acc_epoch:.4f}, val_loss={val_loss_epoch:.4f}, val_acc={acc:.4f}\n")
        log_file.flush()
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()

    # Save best model
    save_path = os.path.join(save_name, "only_cls_linear.pt")
    torch.save(best_model_state, save_path)
    print(f"[only_cls] Best val acc: {best_acc:.4f}, model saved to {save_path}")
    log_file.close()


# New function: train_only_sparse_cls
def train_only_sparse_cls(args, backbone_features, val_backbone_features, train_y, val_y, save_name):
    train_ds = IndexedTensorDataset(backbone_features, train_y)
    val_ds = TensorDataset(val_backbone_features, val_y)

    train_loader = DataLoader(train_ds, batch_size=args.no_cbm_batchsize, shuffle=True, num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.no_cbm_batchsize, shuffle=False, num_workers=10, pin_memory=True)

    linear = torch.nn.Linear(backbone_features.shape[1], args.nb_classes).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    STEP_SIZE = 0.05
    ALPHA = 0.99
    metadata = {'max_reg': {'nongrouped': args.lam}}

    output_proj = glm_saga(
        linear, train_loader, STEP_SIZE, args.n_iters, ALPHA, epsilon=1, k=1,
        val_loader=val_loader, do_zero=False, metadata=metadata,
        n_ex=len(backbone_features), n_classes=args.nb_classes, verbose=500
    )

    W_g = output_proj['path'][0]['weight']
    b_g = output_proj['path'][0]['bias']

    torch.save(W_g, os.path.join(save_name, "only_sparse_cls_W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "only_sparse_cls_b_g.pt"))

    # Save metrics
    import json
    metrics = {
        "lam": float(output_proj['path'][0]['lam']),
        "lr": float(output_proj['path'][0]['lr']),
        "alpha": float(output_proj['path'][0]['alpha']),
        "time": float(output_proj['path'][0]['time']),
        "metrics": output_proj['path'][0]['metrics']
    }
    nnz = (W_g.abs() > 1e-5).sum().item()
    total = W_g.numel()
    metrics['sparsity'] = {
        "Non-zero weights": nnz,
        "Total weights": total,
        "Percentage non-zero": nnz / total
    }
    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[only_sparse_cls] Finished training. W_g and b_g saved in {save_name}")
    

# New function: train_dump_linear_cls
def train_dump_linear_cls(args, backbone_features, val_backbone_features, train_y, val_y, save_name):
    from torch.utils.data import TensorDataset, DataLoader
    import torch.optim as optim
    import os
    import json

    train_ds = TensorDataset(backbone_features, train_y)
    val_ds = TensorDataset(val_backbone_features, val_y)

    train_loader = DataLoader(train_ds, batch_size=args.no_cbm_batchsize, shuffle=True, num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.no_cbm_batchsize, shuffle=False, num_workers=10, pin_memory=True)

    # Two-layer model
    dump_linear = torch.nn.Linear(backbone_features.shape[1], args.dump_concept_num, bias=False).to(args.device)
    classifier = torch.nn.Linear(args.dump_concept_num, args.nb_classes, bias=True).to(args.device)

    params = list(dump_linear.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=args.no_cbm_lr, weight_decay=args.no_cbm_weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.no_cbm_epochs // 3), gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    # Initialize dump_linear
    init.kaiming_normal_(dump_linear.weight, mode='fan_in', nonlinearity='linear')

    # Initialize classifier
    init.xavier_uniform_(classifier.weight)
    init.zeros_(classifier.bias)
    best_acc = 0
    best_state = None

    # Logging
    log_file = open(os.path.join(save_name, "train_dump_linear_cls_log.txt"), "w")

    for epoch in range(args.no_cbm_epochs):
        dump_linear.train()
        classifier.train()
        correct = 0
        total = 0
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad()
            feat = dump_linear(x)
            out = classifier(feat)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        scheduler.step()
        train_loss_epoch = total_loss / total if total > 0 else 0.0
        train_acc_epoch = correct / total if total > 0 else 0.0

        # validation
        dump_linear.eval()
        classifier.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(args.device)
                y = y.to(args.device)
                feat = dump_linear(x)
                out = classifier(feat)
                loss = criterion(out, y)
                preds = torch.argmax(out, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                val_loss += loss.item() * y.size(0)
                val_total += y.size(0)
        acc = correct / total if total > 0 else 0.0
        val_loss_epoch = val_loss / val_total if val_total > 0 else 0.0
        print(f"[dump_linear_cls] Epoch {epoch}: val acc {acc:.4f}")
        log_file.write(f"Epoch {epoch}: train_loss={train_loss_epoch:.4f}, train_acc={train_acc_epoch:.4f}, val_loss={val_loss_epoch:.4f}, val_acc={acc:.4f}\n")
        log_file.flush()
        if acc > best_acc:
            best_acc = acc
            best_state = {
                "W_c": dump_linear.weight.detach().cpu(),
                "W_g": classifier.weight.detach().cpu(),
                "b_g": classifier.bias.detach().cpu()
            }

    # Save best model
    torch.save(best_state["W_c"], os.path.join(save_name, "W_c.pt"))
    torch.save(best_state["W_g"], os.path.join(save_name, "W_g.pt"))
    torch.save(best_state["b_g"], os.path.join(save_name, "b_g.pt"))

    metrics = {
        "best_val_acc": best_acc,
        "dump_concept_num": args.dump_concept_num,
        "epochs": args.no_cbm_epochs,
        "lr": args.no_cbm_lr,
        "weight_decay": args.no_cbm_weight_decay
    }
    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[dump_linear_cls] Best val acc: {best_acc:.4f}, model and metrics saved to {save_name}")
    log_file.close()


# New function: train_dump_linear_sparse_cls
def train_dump_linear_sparse_cls(args, backbone_features, val_backbone_features, train_y, val_y, save_name):
    import json

    train_ds = IndexedTensorDataset(backbone_features, train_y)
    val_ds = TensorDataset(val_backbone_features, val_y)

    train_loader = DataLoader(train_ds, batch_size=args.no_cbm_batchsize, shuffle=True, num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.no_cbm_batchsize, shuffle=False, num_workers=10, pin_memory=True)

    dump_linear = torch.nn.Linear(backbone_features.shape[1], args.dump_concept_num, bias=False).to(args.device)
    classifier = torch.nn.Linear(args.dump_concept_num, args.nb_classes, bias=True).to(args.device)

    # Initialize weights
    init.kaiming_normal_(dump_linear.weight, mode='fan_in', nonlinearity='linear')
    init.xavier_uniform_(classifier.weight)
    init.zeros_(classifier.bias)

    # Combine dump_linear and classifier into a single model (flattened as linear with bias)
    model = torch.nn.Sequential(dump_linear, classifier)

    STEP_SIZE = args.no_cbm_lr
    ALPHA = 0.99
    metadata = {'max_reg': {'nongrouped': args.lam}}

    output_proj = glm_saga(
        model, train_loader, STEP_SIZE, args.n_iters, ALPHA, epsilon=1, k=1,
        val_loader=val_loader, do_zero=False, metadata=metadata,
        n_ex=len(backbone_features), n_classes=args.nb_classes, verbose=500
    )

    # Extract the updated weights
    dump_weight = model[0].weight.detach().cpu()
    cls_weight = model[1].weight.detach().cpu()
    cls_bias = model[1].bias.detach().cpu()

    torch.save(dump_weight, os.path.join(save_name, "W_c.pt"))
    torch.save(cls_weight, os.path.join(save_name, "W_g.pt"))
    torch.save(cls_bias, os.path.join(save_name, "b_g.pt"))

    metrics = {
        "lam": float(output_proj['path'][0]['lam']),
        "lr": float(output_proj['path'][0]['lr']),
        "alpha": float(output_proj['path'][0]['alpha']),
        "time": float(output_proj['path'][0]['time']),
        "metrics": output_proj['path'][0]['metrics']
    }
    nnz = (cls_weight.abs() > 1e-5).sum().item() + (dump_weight.abs() > 1e-5).sum().item()
    total = cls_weight.numel() + dump_weight.numel()
    metrics['sparsity'] = {
        "Non-zero weights": nnz,
        "Total weights": total,
        "Percentage non-zero": nnz / total
    }
    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[dump_linear_sparse_cls] Finished training. W_c, W_g, b_g and metrics saved to {save_name}")
    
def train_dump_linear_sparse_cls(args, backbone_features, val_backbone_features, train_y, val_y, save_name):
    import json

    train_ds = IndexedTensorDataset(backbone_features, train_y)
    val_ds = TensorDataset(val_backbone_features, val_y)

    train_loader = DataLoader(train_ds, batch_size=args.no_cbm_batchsize, shuffle=True, num_workers=10, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.no_cbm_batchsize, shuffle=False, num_workers=10, pin_memory=True)

    dump_linear = torch.nn.Linear(backbone_features.shape[1], args.dump_concept_num, bias=False).to(args.device)
    classifier = torch.nn.Linear(args.dump_concept_num, args.nb_classes, bias=True).to(args.device)

    # Initialize weights
    init.kaiming_normal_(dump_linear.weight, mode='fan_in', nonlinearity='linear')
    init.xavier_uniform_(classifier.weight)
    init.zeros_(classifier.bias)

    # Combine dump_linear and classifier into a single equivalent Linear layer
    combined_linear = torch.nn.Linear(backbone_features.shape[1], args.nb_classes).to(args.device)
    with torch.no_grad():
        combined_linear.weight.copy_(torch.matmul(classifier.weight, dump_linear.weight))
        combined_linear.bias.copy_(classifier.bias)

    STEP_SIZE = args.no_cbm_lr
    ALPHA = 0.99
    metadata = {'max_reg': {'nongrouped': args.lam}}

    # Run sparse optimization
    output_proj = glm_saga(
        combined_linear, train_loader, STEP_SIZE, args.n_iters, ALPHA,
        epsilon=1, k=1, val_loader=val_loader, do_zero=False,
        metadata=metadata, n_ex=len(backbone_features),
        n_classes=args.nb_classes, verbose=500
    )

    # Save model parameters
    W_c = dump_linear.weight.detach().cpu()
    W_g = classifier.weight.detach().cpu()
    b_g = classifier.bias.detach().cpu()

    torch.save(W_c, os.path.join(save_name, "W_c.pt"))
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))

    # Save metrics
    metrics = {
        "lam": float(output_proj['path'][0]['lam']),
        "lr": float(output_proj['path'][0]['lr']),
        "alpha": float(output_proj['path'][0]['alpha']),
        "time": float(output_proj['path'][0]['time']),
        "metrics": output_proj['path'][0]['metrics']
    }
    nnz = (combined_linear.weight.abs() > 1e-5).sum().item()
    total = combined_linear.weight.numel()
    metrics['sparsity'] = {
        "Non-zero weights": nnz,
        "Total weights": total,
        "Percentage non-zero": nnz / total
    }
    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[dump_linear_sparse_cls] Finished training. W_c, W_g, b_g and metrics saved to {save_name}")