import torch
import torch.nn.functional as F

def iou_score(output, target):
    smooth = 1e-5  # Prevents division by zero
    output_ = (output > 0.5).float()  # Threshold at 0.5
    target_ = (target > 0.5).float()
    
    intersection = (output_ * target_).sum()
    union = output_.sum() + target_.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def dice_coef(output, target):
    smooth = 1e-5
    output = output.flatten()  # Flatten tensors
    target = target.flatten()
    
    intersection = (output * target).sum()
    dice = (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)
    
    return dice


def accuracy_score(output, target):
    smooth = 1e-5
    output_ = (output > 0.5).float()
    target_ = (target > 0.5).float()
    
    tp = (output_ * target_).sum()  # True positives
    tn = ((1 - output_) * (1 - target_)).sum()  # True negatives
    total = output_.numel()
    
    accuracy = (tp + tn + smooth) / (total + smooth)
    return accuracy


def indicators(output, target):
    output_ = (output > 0.5).float()  # Binary mask
    target_ = (target > 0.5).float()
    
    # Calculate TP, FP, FN, TN
    tp = (output_ * target_).sum()
    fp = output_.sum() - tp
    fn = target_.sum() - tp
    tn = output_.numel() - (tp + fp + fn)
    
    # Use existing functions for metrics
    iou_ = iou_score(output, target)
    dice_ = dice_coef(output, target)
    accuracy_ = accuracy_score(output, target)
    
    # Calculate remaining metrics
    precision_ = tp / (tp + fp + 1e-5)
    recall_ = tp / (tp + fn + 1e-5)
    specificity_ = tn / (tn + fp + 1e-5)
    
    return iou_, dice_, recall_, specificity_, precision_, accuracy_