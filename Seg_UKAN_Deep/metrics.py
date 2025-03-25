import numpy as np
import torch
import torch.nn.functional as F
from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision

def safe_divide(a, b):
    """Safe division with handling for zero denominator"""
    return a / (b + 1e-10)

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    output_ = output > 0.5
    target_ = target > 0.5
    
    # Handle empty predictions
    if not np.any(output_) and not np.any(target_):
        return 1.0, 1.0, 0.0  # Perfect score if both empty
    elif not np.any(output_):
        return 0.0, 0.0, float('inf')  # No prediction but target exists
    
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    
    iou = safe_divide(intersection + smooth, union + smooth)
    dice = safe_divide(2 * intersection, (output_.sum() + target_.sum() + smooth))
    
    try:
        hd95_ = hd95(output_, target_) if np.any(output_) and np.any(target_) else float('inf')
    except RuntimeError:
        hd95_ = float('inf')
    
    return iou, dice, hd95_

def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    
    # Handle empty cases
    if not np.any(output) and not np.any(target):
        return 1.0
    
    intersection = (output * target).sum()
    return safe_divide(2. * intersection, (output.sum() + target.sum()))

def indicators(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    output_ = output > 0.5
    target_ = target > 0.5
    
    # Initialize defaults
    metrics = {
        'iou': 0.0,
        'dice': 0.0,
        'hd': float('inf'),
        'hd95': float('inf'),
        'recall': 0.0,
        'specificity': 1.0,
        'precision': 0.0,
        'accuracy': 0.0
    }
    
    # Case 1: Both empty
    if not np.any(output_) and not np.any(target_):
        metrics.update({
            'iou': 1.0,
            'dice': 1.0,
            'hd': 0.0,
            'hd95': 0.0,
            'recall': 1.0,
            'specificity': 1.0,
            'precision': 1.0,
            'accuracy': 1.0
        })
        return tuple(metrics.values())
    
    # Case 2: Prediction empty but target not
    elif not np.any(output_):
        metrics['specificity'] = specificity(output_, target_)
        metrics['accuracy'] = (output_ == target_).mean()
        return tuple(metrics.values())
    
    # Case 3: Normal case
    try:
        metrics.update({
            'iou': jc(output_, target_),
            'dice': dc(output_, target_),
            'recall': recall(output_, target_),
            'specificity': specificity(output_, target_),
            'precision': precision(output_, target_),
            'accuracy': (output_ == target_).mean()
        })
        
        # Only calculate HD if both have foreground
        if np.any(output_) and np.any(target_):
            try:
                metrics['hd'] = hd(output_, target_)
                metrics['hd95'] = hd95(output_, target_)
            except RuntimeError:
                pass
    
    except Exception as e:
        print(f"Metric calculation warning: {str(e)}")
    
    return tuple(metrics.values())