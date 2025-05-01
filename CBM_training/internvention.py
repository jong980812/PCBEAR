import torch
def analyze_intervention_effect(original_pred, modified_pred, val_y):
    """
    original_pred: intervention 전 logits (N, num_classes)
    modified_pred: intervention 후 logits (N, num_classes)
    val_y: 정답 labels (N,)
    """

    original_labels = original_pred.argmax(dim=1)
    modified_labels = modified_pred.argmax(dim=1)

    # 상태별 구분
    matched_before = (original_labels == val_y)  # intervention 전 맞음
    matched_after = (modified_labels == val_y)   # intervention 후 맞음

    # 케이스 분류
    A_keep_correct = matched_before & matched_after
    B_fix_incorrect = (~matched_before) & matched_after
    C_break_correct = matched_before & (~matched_after)
    D_keep_wrong = (~matched_before) & (~matched_after)

    print("\nIntervention Result Summary:")
    print(f" - A. 유지 (맞던 것 계속 맞음): {A_keep_correct.sum().item()} samples")
    print(f" - B. 개선 (틀리던 것 맞춤): {B_fix_incorrect.sum().item()} samples")
    print(f" - C. 악화 (맞던 것 틀림): {C_break_correct.sum().item()} samples")
    print(f" - D. 무효 (틀리던 것 계속 틀림): {D_keep_wrong.sum().item()} samples")

    total = val_y.size(0)
    print(f" - 전체 sample: {total}")

    # 성능 변화 분석
    original_acc = matched_before.float().mean().item()
    modified_acc = matched_after.float().mean().item()
    print(f"\nAccuracy:")
    print(f" - Before: {original_acc*100:.2f}%")
    print(f" - After : {modified_acc*100:.2f}%")
    print(f" - Change: {(modified_acc - original_acc)*100:+.2f}%")

    return {
        "A_keep_correct": A_keep_correct,
        "B_fix_incorrect": B_fix_incorrect,
        "C_break_correct": C_break_correct,
        "D_keep_wrong": D_keep_wrong
    }
def compare_predictions(original_pred, modified_pred, val_y, gt_class_idx, intervene_class_idx):
    """
    original_pred: intervention 전 (N, num_classes)
    modified_pred: intervention 후 (N, num_classes)
    val_y: (N,) ground truth labels
    gt_class_idx: intervention할 때 ground truth class index
    intervene_class_idx: intervention할 때 intervention class index
    """

    # ✅ CPU로 이동 (안전)
    original_pred = original_pred.cpu()
    modified_pred = modified_pred.cpu()
    val_y = val_y.cpu()

    # ✅ 예측 label
    original_labels = original_pred.argmax(dim=1)
    modified_labels = modified_pred.argmax(dim=1)

    # ✅ 정답 여부
    original_correct = (original_labels == val_y)
    modified_correct = (modified_labels == val_y)

    # ✅ accuracy 계산
    original_acc = original_correct.float().mean().item()
    modified_acc = modified_correct.float().mean().item()

    print(f"\nAccuracy Change:")
    print(f" - Before Intervention: {original_acc*100:.2f}%")
    print(f" - After  Intervention: {modified_acc*100:.2f}%")
    print(f" - Change: {(modified_acc - original_acc)*100:+.2f}%\n")

    # ✅ logit 변화량 (gt class와 intervention class 중심)
    gt_logits_change = (modified_pred[:, gt_class_idx] - original_pred[:, gt_class_idx])
    intervene_logits_change = (modified_pred[:, intervene_class_idx] - original_pred[:, intervene_class_idx])

    print(f"Logit Changes (mean over all samples):")
    print(f" - GT class ({gt_class_idx}) logit change: {gt_logits_change.mean().item():+.4f}")
    print(f" - Intervention class ({intervene_class_idx}) logit change: {intervene_logits_change.mean().item():+.4f}\n")

    # ✅ 추가 분석: 정답 맞춘 sample에서 변화, 틀린 sample에서 변화
    correct_idx = (val_y == gt_class_idx)


    # correct_idx가 bool 타입이면 Tensor로 변환
    if isinstance(correct_idx, torch.Tensor):
        if correct_idx.sum() > 0:
            num_samples = correct_idx.sum().item()
        else:
            num_samples = 0
    else:
        # correct_idx가 그냥 bool일 때
        num_samples = int(correct_idx)

    if num_samples > 0:
        print(f"Among GT class samples ({num_samples} samples):")
        gt_logits_change_correct = gt_logits_change[correct_idx]
        intervene_logits_change_correct = intervene_logits_change[correct_idx]
        
        print(f" - GT class logit change (on correct samples): {gt_logits_change_correct.mean().item():+.4f}")
        print(f" - Intervention class logit change (on correct samples): {intervene_logits_change_correct.mean().item():+.4f}")

        # 정답 유지/변화 분석
        original_correct_gt = (original_labels[correct_idx] == val_y[correct_idx])
        modified_correct_gt = (modified_labels[correct_idx] == val_y[correct_idx])

        print(f"\nCorrect prediction within GT class samples:")
        print(f" - Before: {original_correct_gt.float().mean().item()*100:.2f}%")
        print(f" - After : {modified_correct_gt.float().mean().item()*100:.2f}%")
    else:
        print(f"No samples of ground-truth class {gt_class_idx} in val_y.")


import torch

def intervene_and_predict(val_backbone_features, 
                           proj_layer, 
                           cls_layer, 
                           W_c, 
                           W_g, 
                           b_g, 
                           proj_mean, 
                           proj_std, 
                           gt_class_idx, 
                           intervene_class_idx, 
                           concept_idx, 
                           margin, 
                           device):
    """
    return:
        original_prediction: intervention 전 prediction
        modified_prediction: intervention 후 prediction
    """

    # ✅ 먼저 val_c 계산 (concept activation)
    proj_layer.load_state_dict({"weight": W_c})
    proj_layer = proj_layer.to(device)

    with torch.no_grad():
        val_c = proj_layer(val_backbone_features.to(device).detach())
        val_c -= proj_mean
        val_c /= proj_std

    # ✅ classification layer 준비
    cls_layer.load_state_dict({"weight": W_g, "bias": b_g})
    cls_layer = cls_layer.to(device)

    # ✅ 원래 prediction
    with torch.no_grad():
        original_prediction = cls_layer(val_c.detach())

    # ✅ W_g를 복사해서 intervention
    W_g_modified = W_g.clone()

    # intervention class에서 concept idx weight를 margin만큼 줄임
    W_g_modified[intervene_class_idx, concept_idx] -= margin

    # 정답 class에서 concept idx weight를 margin만큼 늘림
    W_g_modified[gt_class_idx, concept_idx] += margin

    # ✅ 수정된 weight로 새 cls_layer 만들기
    modified_cls_layer = torch.nn.Linear(val_c.shape[1], W_g.shape[0]).to(device)
    with torch.no_grad():
        modified_cls_layer.weight.copy_(W_g_modified)
        modified_cls_layer.bias.copy_(b_g)

    # ✅ intervention 후 prediction
    with torch.no_grad():
        modified_prediction = modified_cls_layer(val_c.detach())

    return original_prediction, modified_prediction