import torch
import torch.nn.functional as F

def _to_float(x):
    """Helper function to convert tensor to float"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()
    return float(x)

def iou_score(output, target, threshold=0.6, smooth=1e-5):
    output_ = (output > threshold).float()
    target_ = (target > threshold).float()
    
    intersection = (output_ * target_).sum()
    union = output_.sum() + target_.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return _to_float(iou)

def dice_coef(output, target, threshold=0.6, smooth=1e-5):
    output_ = (output > threshold).float().flatten()
    target_ = (target > threshold).float().flatten()
    
    intersection = (output_ * target_).sum()
    dice = (2. * intersection + smooth) / (output_.sum() + target_.sum() + smooth)
    
    return _to_float(dice)

def accuracy_score(output, target, threshold=0.6, smooth=1e-5):
    output_ = (output > threshold).float()
    target_ = (target > threshold).float()
    
    tp = (output_ * target_).sum()  # True positives
    tn = ((1 - output_) * (1 - target_)).sum()  # True negatives
    total = output_.numel()
    
    accuracy = (tp + tn + smooth) / (total + smooth)
    return _to_float(accuracy)

def indicators(output, target, threshold=0.6):
    output_ = (output > threshold).float()
    target_ = (target > threshold).float()
    
    # Calculate basic statistics
    tp = (output_ * target_).sum()
    fp = output_.sum() - tp
    fn = target_.sum() - tp
    tn = output_.numel() - (tp + fp + fn)
    
    # Compute metrics
    iou_ = iou_score(output, target, threshold)
    dice_ = dice_coef(output, target, threshold)
    accuracy_ = accuracy_score(output, target, threshold)
    
    # Calculate rates with smoothing
    smooth = 1e-5
    precision_ = _to_float(tp / (tp + fp + smooth))
    recall_ = _to_float(tp / (tp + fn + smooth))
    specificity_ = _to_float(tn / (tn + fp + smooth))
    
    return iou_, dice_, recall_, specificity_, precision_, accuracy_

def evaluate_multiple_thresholds(output, target, thresholds=[0.4, 0.45, 0.5, 0.55, 0.6]):
    results = {}
    for thresh in thresholds:
        iou, dice, recall, specificity, precision, accuracy = indicators(output, target, threshold=thresh)
        results[thresh] = {
            'iou': iou,
            'dice': dice,
            'recall': recall,
            'specificity': specificity,
            'precision': precision,
            'accuracy': accuracy
        }
    return results