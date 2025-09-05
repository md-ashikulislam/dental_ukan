import os
import gc
import argparse
from collections import OrderedDict
from glob import glob
import random
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import matplotlib.pyplot as plt
from thop import profile, clever_format
from scipy.interpolate import BSpline

import albumentations as A
from albumentations import CLAHE

from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize
from albumentations import MedianBlur
from metrics import evaluate_multiple_thresholds 

import archs
import archs_fcsa_fusion_concat
from kan import KANLinear

import losses
from dataset import Dataset
from metrics import iou_score, indicators, dice_coef, accuracy_score
from utils import AverageMeter, str2bool

from tensorboardX import SummaryWriter
from prettytable import PrettyTable

ARCH_NAMES = archs_fcsa_fusion_concat.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--dataseed', default=2981, type=int,
                        help='')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UKAN')
    
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='busi', help='dataset name')      
    parser.add_argument('--data_dir', default='inputs', help='dataset dir')

    parser.add_argument('--output_dir', default='outputs', help='ouput dir')


    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    parser.add_argument('--kan_lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--no_kan', action='store_true')



    config = parser.parse_args()

    return config

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'accuracy': AverageMeter()}  # Add accuracy

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda(non_blocking=True)  # Move input to GPU
        target = target.cuda(non_blocking=True)  # Move target to GPU
        # input = input.cuda()
        # target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            output = outputs[-1]
            
        else:
            output = model(input)
            loss = criterion(output, target)

        # compute metrics
        iou = iou_score(output, target)
        dice = dice_coef(output, target)
        accuracy = accuracy_score(output, target)

        # If you need other metrics later:
        # iou, dice, recall, specificity, precision, accuracy = indicators(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['accuracy'].update(accuracy, input.size(0))  # Add accuracy update

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('accuracy', avg_meters['accuracy'].avg)  # Include accuracy in logs
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('accuracy', avg_meters['accuracy'].avg)])  # Return accuracy

def validate(config, val_loader, model, criterion):
    avg_meters = {
        'loss': AverageMeter(),
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'accuracy': AverageMeter(),
        'recall': AverageMeter(),
        'specificity': AverageMeter(),
        'precision': AverageMeter(),
        # Add meters for each threshold
        'iou_thresholds': {t: AverageMeter() for t in [0.4, 0.45, 0.5, 0.55, 0.6]},
        'dice_thresholds': {t: AverageMeter() for t in [0.4, 0.45, 0.5, 0.55, 0.6]}
    }

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                output = outputs[-1]
            else:
                output = model(input)
                loss = criterion(output, target)

            # Get all metrics at default threshold (0.4)
            iou, dice, recall, specificity, precision, accuracy = indicators(output, target)

            # Get multi-threshold metrics
            threshold_results = evaluate_multiple_thresholds(output, target)

            # Update main metrics
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['accuracy'].update(accuracy, input.size(0))
            avg_meters['recall'].update(recall, input.size(0))
            avg_meters['specificity'].update(specificity, input.size(0))
            avg_meters['precision'].update(precision, input.size(0))

            # Update threshold-specific metrics
            for thresh, metrics in threshold_results.items():
                avg_meters['iou_thresholds'][thresh].update(metrics['iou'], input.size(0))
                avg_meters['dice_thresholds'][thresh].update(metrics['dice'], input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    # Prepare results dictionary
    results = OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg),
        ('accuracy', avg_meters['accuracy'].avg),
        ('recall', avg_meters['recall'].avg),
        ('specificity', avg_meters['specificity'].avg),
        ('precision', avg_meters['precision'].avg),
    ])
    
    # Add threshold results
    for thresh in [0.4, 0.45, 0.5, 0.55, 0.6]:
        results[f'iou_{thresh}'] = avg_meters['iou_thresholds'][thresh].avg
        results[f'dice_{thresh}'] = avg_meters['dice_thresholds'][thresh].avg

    return results

def visualize_single_sample(writer, model, val_loader, epoch):
    """Plot activations, prediction, and GT as separate full-size figures"""    
    # Get sample
    torch.manual_seed(epoch)  # Makes selection consistent per epoch
    inputs, targets, _ = next(iter(val_loader))
    idx = torch.randint(0, inputs.size(0), (1,)).item()
    input_img = inputs[idx].unsqueeze(0).cuda()
    target_mask = targets[idx].unsqueeze(0).cuda()
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        if isinstance(model, torch.nn.DataParallel):
            output, activations = model.module(input_img)
        else:
            output, activations = model(input_img)
        output = torch.sigmoid(output)
    
    # Calculate metrics
    iou = iou_score(output, target_mask)
    dice = dice_coef(output, target_mask)
    piou, _ = calculate_plausibility_iou(activations, target_mask)
    
    # Prepare tensors
    input_img = input_img.cpu().squeeze()
    target_mask = target_mask.cpu().squeeze()
    output = output.cpu().squeeze()
    activations = activations.cpu().squeeze()
    
    # Normalize activations (mean across channels if needed)
    if len(activations.shape) == 3:
        activations = activations.mean(dim=0)
    activations = (activations - activations.min()) / (activations.max() - activations.min() + 1e-6)

    # -----------------------------------------------
    # 1. Activation Heatmap (Full Page)
    plt.figure(figsize=(12, 10))
    plt.imshow(activations, cmap='jet', vmin=0, vmax=1)
    plt.title(f"Activation Map (PIoU: {piou:.2f})", fontsize=16, pad=20)
    plt.axis('off')
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    writer.add_figure('activations/heatmap', plt.gcf(), epoch)
    plt.close()
    
    # 2. Prediction Mask (Full Page)
    plt.figure(figsize=(12, 10))
    plt.imshow((output > 0.6).float(), cmap='gray')
    plt.title(f"Prediction\nIoU: {iou:.2f}  Dice: {dice:.2f}", 
              fontsize=16, pad=20)
    plt.axis('off')
    writer.add_figure('prediction/mask', plt.gcf(), epoch)
    plt.close()
    
    # 3. Ground Truth (Full Page)
    plt.figure(figsize=(12, 10))
    plt.imshow(target_mask, cmap='gray')
    plt.title("Ground Truth", fontsize=16, pad=20)
    plt.axis('off')
    writer.add_figure('ground_truth/mask', plt.gcf(), epoch)
    plt.close()

def calculate_plausibility_iou(activations, gt_mask, threshold_percentile=90):
    """Calculate Plausibility IoU between thresholded activations and GT mask"""
    # Average across channels if multi-channel
    if activations.size(1) > 1:
        activations = torch.mean(activations, dim=1, keepdim=True)
    
    # Calculate threshold value
    flat_acts = activations.view(-1)
    threshold = torch.quantile(flat_acts, threshold_percentile/100)
    
    # Threshold activations
    thresholded = (activations > threshold).float()
    
    # Calculate IoU
    intersection = (thresholded * gt_mask).sum()
    union = (thresholded + gt_mask).clamp(0, 1).sum()
    piou = (intersection / (union + 1e-6)).item()
    
    return piou, thresholded

def visualize_kan_activations(writer, model, epoch):
    """
    Visualize learned B-spline activation functions and control points from the first KANLinear layer (fc1)
    in the first KANLayer of the first KANBlock in the UKAN model.
    Plots functions for output channel 0 and input channels 0-8.
    """
    # Handle DataParallel if used
    model = model.module if isinstance(model, torch.nn.DataParallel) else model

    # Navigate to the first KANLinear layer: model.block1[0].layer.fc1
    kan_layer = None
    try:
        kan_layer = model.block1[0].layer.fc1
        if not isinstance(kan_layer, KANLinear):
            print("First layer is not a KANLinear layer!")
            return
    except (AttributeError, IndexError):
        print("Could not find KANLinear layer in model.block1[0].layer.fc1!")
        return

    # Get the B-spline parameters
    grid = kan_layer.grid.detach().cpu().numpy()  # Shape: (in_features, grid_size + 2 * spline_order + 1)
    spline_weight = kan_layer.spline_weight.detach().cpu().numpy()  # Shape: (out_features, in_features, grid_size + spline_order)
    base_weight = kan_layer.base_weight.detach().cpu().numpy()  # Shape: (out_features, in_features)

    # Handle spline_scaler if enabled
    if kan_layer.enable_standalone_scale_spline:
        spline_scaler = kan_layer.spline_scaler.detach().cpu().numpy()  # Shape: (out_features, in_features)
        # Scale the spline weights
        spline_weight = spline_weight * spline_scaler[:, :, np.newaxis]  # Broadcasting to match dimensions

    # For output channel 0, input channels 0-8
    out_ch = 0
    num_inputs_to_plot = min(8, kan_layer.in_features)

    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()

    for in_ch in range(num_inputs_to_plot):
        # Get the specific spline parameters for this input-output pair
        grid_points = grid[in_ch, :]  # Grid for this input channel
        spline_coeff = spline_weight[out_ch, in_ch, :]  # Spline coefficients for (out_ch, in_ch)
        base_w = base_weight[out_ch, in_ch]  # Base weight for (out_ch, in_ch)

        # Create B-spline function
        spline = BSpline(grid_points, spline_coeff, kan_layer.spline_order, extrapolate=False)

        # Evaluation points
        x = np.linspace(grid_points[0], grid_points[-1], 100)

        # Compute activation function output
        x_tensor = torch.tensor(x, dtype=torch.float32)
        base_component = base_w * kan_layer.base_activation(x_tensor).numpy()
        spline_component = spline(x)
        y = base_component + spline_component

        # Plot
        ax = axes[in_ch]
        ax.plot(x, y, 'b-', label='Activation function')  # Blue line for activation function
        # Plot control points (truncate grid to match spline_coeff size)
        num_control_points = len(spline_coeff)
        control_grid_points = grid_points[:num_control_points]
        ax.scatter(control_grid_points, spline_coeff, color='purple', label='Control points')
        ax.set_title(f'Input {in_ch} → Output {out_ch}')
        ax.set_xlabel('Input')
        ax.set_ylabel('Output')
        ax.grid(True)
        ax.legend()

    # Remove empty subplots if needed
    for i in range(num_inputs_to_plot, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(f'Learned KAN Activation Functions (Epoch {epoch})', y=1.02)
    plt.tight_layout()

    # Add to TensorBoard
    writer.add_figure('kan_activations/first_kan_layer', fig, epoch)
    plt.close(fig)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def print_metrics(metrics):
    """Helper function to print metrics in a readable format"""
    print("\nCurrent Metrics:")
    print("-" * 40)
    print(f"{'Metric':<15}{'Value':>10}")
    print("-" * 40)
    
    # Print core metrics
    core_metrics = ['iou', 'dice', 'accuracy', 'recall', 'specificity', 'precision']
    for metric in core_metrics:
        print(f"{metric:<15}{metrics[metric]:>10.4f}")
    
    # Print threshold metrics
    print("\nThreshold Metrics:")
    for thresh in [0.4, 0.45, 0.5, 0.55, 0.6]:
        print(f"IoU@{thresh:.2f}: {metrics[f'iou_{thresh}']:.4f}  Dice@{thresh:.2f}: {metrics[f'dice_{thresh}']:.4f}")
    print("-" * 40 + "\n")

def count_parameters(model):
    """Count and display trainable parameters using PrettyTable"""
    table = PrettyTable(["Module", "Parameters"])
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            table.add_row([name, f"{param_count:,}"])
            total_params += param_count
    print(table)
    print(f"Total Trainable Parameters: {total_params:,}")
    return total_params

def calculate_gflops(model, input_shape=(1, 1, 512, 512)):
    """Calculate GFLOPS and parameters using thop."""
    device = next(model.parameters()).device  # Get model's device
    dummy_input = torch.randn(*input_shape).to(device)  # Match device
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params

def main():
    seed_torch()
    config = vars(parse_args())

    exp_name = config.get('name')
    output_dir = config.get('output_dir')

    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    
    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True
    criterion = criterion.cuda() 

    # create model
    model = archs_fcsa_fusion_concat.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'], embed_dims=config['input_list'], no_kan=config['no_kan'])
    print(f"no_kan mode is {'ACTIVATED' if config['no_kan'] else 'NOT activated'}")


    # Count parameters and print PrettyTable
    total_params = count_parameters(model)
    config['total_params'] = total_params  # Store in config for yaml

    # Calculate and print GFLOPS
    gflops, params_formatted = calculate_gflops(
        model.module if isinstance(model, nn.DataParallel) else model,
        input_shape=(1, config['input_channels'], config['input_h'], config['input_w'])
    )
    print(f"Model GFLOPS: {gflops}, Params: {params_formatted}")

    model = model.cuda()  # Move to CUDA

    #FOR 2 GPUs
    # Move model to multiple GPUs
    if torch.cuda.device_count() > 1:
      print(f"Using {torch.cuda.device_count()} GPUs!")
      model = torch.nn.DataParallel(model)

    param_groups = []

    kan_fc_params = []
    other_params = []

    for name, param in model.named_parameters():
        # print(name, "=>", param.shape)
        if 'layer' in name.lower() and 'fc' in name.lower(): # higher lr for kan layers
            # kan_fc_params.append(name)
            param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']}) 
        else:
            # other_params.append(name)
            param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})  
    

    # st()
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)


    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(param_groups, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError
    
    # Load the checkpoint
    # checkpoint = torch.load('/kaggle/input/checkukan/UKAN150/model.pth')

    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])


    dataset_name = config['dataset']

    if dataset_name == 'Dental' or 'new_dataset':
       img_ext = '.JPG'       
       mask_ext = '.jpg'
    
    # Data loading code
    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.15, random_state=config['dataseed'])

    train_transform = Compose([
        RandomRotate90(),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        Resize(config['input_h'], config['input_w']),
        # A.ToGray(),
        MedianBlur(blur_limit=3),  # Median filter (3x3 kernel)
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),  # Add CLAHE here
        A.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        # A.ToGray(),
        MedianBlur(blur_limit=3),  # Median filter (3x3 kernel)
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),  # Add CLAHE here
        A.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform)

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'] ,config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)

    # Log the number of training images after transformation
    print(f"Number of training images after transformation: {len(train_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('accuracy', []),  # Add accuracy logging
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('val_accuracy', []),  # Add val_accuracy
    ])

    best_iou = 0
    best_dice= 0
    best_accuracy= 0
    trigger = 0

    thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]
    best_metrics = {}  # Store all metrics when best IoU is achieved

    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))


        train_log = train(config, train_loader, model, criterion, optimizer)
        val_log = validate(config, val_loader, model, criterion)

        # Add this to your main training loop (after validation)
        if epoch % 2 == 0:  # Every 2 epochs
            visualize_single_sample(my_writer, model, val_loader, epoch)
            visualize_kan_activations(my_writer, model, epoch)  # Add this line


        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - accuracy %.4f - val_loss %.4f - val_iou %.4f - val_accuracy %.4f'
              % (train_log['loss'], train_log['iou'], train_log['accuracy'], val_log['loss'], val_log['iou'], val_log['accuracy']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['accuracy'].append(train_log['accuracy'])  # Add train accuracy
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_accuracy'].append(val_log['accuracy'])  # Add val accuracy

        for thresh in thresholds:
            log[f'val_iou_{thresh}'] = val_log[f'iou_{thresh}']
            log[f'val_dice_{thresh}'] = val_log[f'dice_{thresh}']

        pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('train/accuracy', train_log['accuracy'], global_step=epoch)  # Add accuracy
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)
        my_writer.add_scalar('val/accuracy', val_log['accuracy'], global_step=epoch)  # Add accuracy

        my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)
        my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch)
        my_writer.add_scalar('val/best_accuracy_value', best_accuracy, global_step=epoch)

                # Log threshold metrics
        for thresh in thresholds:
            my_writer.add_scalar(f'val_thresholds/iou_{thresh}', val_log[f'iou_{thresh}'], global_step=epoch)
            my_writer.add_scalar(f'val_thresholds/dice_{thresh}', val_log[f'dice_{thresh}'], global_step=epoch)

        current_best_iou = max([val_log[f'iou_{thresh}'] for thresh in thresholds])


        if current_best_iou > best_iou:
            best_iou = current_best_iou
            best_metrics = val_log.copy()  # Save all metrics at this point
            best_threshold = [thresh for thresh in thresholds if val_log[f'iou_{thresh}'] == best_iou][0]

            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_iou': best_iou,
                'best_threshold': best_threshold,
                'best_metrics': best_metrics
            }
            torch.save(checkpoint, f'{output_dir}/{exp_name}/model.pth')
            print(f"=> Saved best model with IoU {best_iou:.4f} at threshold {best_threshold}")
            print_metrics(best_metrics)  # Display all metrics
            trigger = 0
        else:
            trigger += 1

        my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)
        if best_metrics:
            my_writer.add_scalar('val/best_dice_value', best_metrics['dice'], global_step=epoch)
            my_writer.add_scalar('val/best_accuracy_value', best_metrics['accuracy'], global_step=epoch)


        if config['early_stopping'] > 0 and trigger >= config['early_stopping']:
             print("=> early stopping")
             break

        # Thorough memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
if __name__ == '__main__':
    main()
