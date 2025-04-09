import torch
import torch.nn.functional as F

def _to_float(x):
    """Helper function to convert tensor to float"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()
    return float(x)

def iou_score(output, target, threshold=0.5, smooth=1e-5):
    output_ = (output > threshold).float()
    target_ = (target > threshold).float()
    
    intersection = (output_ * target_).sum()
    union = output_.sum() + target_.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return _to_float(iou)

def dice_coef(output, target, threshold=0.5, smooth=1e-5):
    output_ = (output > threshold).float().flatten()
    target_ = (target > threshold).float().flatten()
    
    intersection = (output_ * target_).sum()
    dice = (2. * intersection + smooth) / (output_.sum() + target_.sum() + smooth)
    
    return _to_float(dice)

def accuracy_score(output, target, threshold=0.5, smooth=1e-5):
    output_ = (output > threshold).float()
    target_ = (target > threshold).float()
    
    tp = (output_ * target_).sum()  # True positives
    tn = ((1 - output_) * (1 - target_)).sum()  # True negatives
    total = output_.numel()
    
    accuracy = (tp + tn + smooth) / (total + smooth)
    return _to_float(accuracy)

def indicators(output, target, threshold=None):

    if threshold is None:
        # Find best threshold if none provided
        thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]
        best_iou = 0
        best_thresh = 0.5
        for thresh in thresholds:
            current_iou = iou_score(output, target, threshold=thresh)
            if current_iou > best_iou:
                best_iou = current_iou
                best_thresh = thresh
        threshold = best_thresh
    
    output_ = (output > threshold).float()
    target_ = (target > threshold).float()
    
    # Calculate basic statistics
    tp = (output_ * target_).sum()
    fp = output_.sum() - tp
    fn = target_.sum() - tp
    tn = output_.numel() - (tp + fp + fn)
    
    # Compute metrics with smoothing
    smooth = 1e-5
    iou_ = iou_score(output, target, threshold)
    dice_ = dice_coef(output, target, threshold)
    accuracy_ = (tp + tn + smooth) / (output_.numel() + smooth)
    precision_ = tp / (tp + fp + smooth)
    recall_ = tp / (tp + fn + smooth)
    specificity_ = tn / (tn + fp + smooth)
    
    return {
        'iou': _to_float(iou_),
        'dice': _to_float(dice_),
        'accuracy': _to_float(accuracy_),
        'precision': _to_float(precision_),
        'recall': _to_float(recall_),
        'specificity': _to_float(specificity_),
        'threshold': threshold  # Return the threshold used
    }

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