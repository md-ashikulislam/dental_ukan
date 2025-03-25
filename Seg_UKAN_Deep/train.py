import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from albumentations import Compose, RandomRotate90, Resize
from albumentations.augmentations import transforms
import albumentations as A
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import subprocess
from pdb import set_trace as st

import archs
import losses
from dataset import Dataset
from metrics import iou_score, indicators
from utils import AverageMeter, str2bool

ARCH_NAMES = ['UKAN']
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

def list_type(s):
    return [int(a) for a in s.split(',')]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('-b', '--batch_size', default=16, type=int)
    parser.add_argument('--dataseed', default=2981, type=int)
    
    # Model args
    parser.add_argument('--arch', default='UKAN')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_w', default=256, type=int)
    parser.add_argument('--input_h', default=256, type=int)
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])
    
    # Loss args
    parser.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES)
    
    # Dataset args
    parser.add_argument('--dataset', default='busi')
    parser.add_argument('--data_dir', default='inputs')
    parser.add_argument('--output_dir', default='outputs')
    
    # Optimizer args
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)
    parser.add_argument('--kan_lr', default=1e-2, type=float)
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float)
    
    # Scheduler args
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                      choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int)
    parser.add_argument('--cfg', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--no_kan', action='store_true')

    return vars(parser.parse_args())

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                 'iou': AverageMeter(),
                 'accuracy': AverageMeter()}
    
    model.train()
    
    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # Forward pass
        outputs = model(input)
        
        # Deep supervision handling
        if config['deep_supervision']:
            loss = 0
            for output in outputs:
                # Ensure output matches target size
                if output.shape[-2:] != target.shape[-2:]:
                    output = F.interpolate(output, size=target.shape[-2:], 
                                         mode='bilinear', align_corners=True)
                loss += criterion(output, target)
            loss /= len(outputs)
            # Metrics on final output
            output = outputs[-1]
        else:
            loss = criterion(outputs, target)
            output = outputs
        
        # Compute metrics
        iou, dice, _ = iou_score(output, target)
        iou_, dice_, hd_, hd95_, recall_, specificity_, precision_, accuracy_ = indicators(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['accuracy'].update(accuracy_, input.size(0))
        
        pbar.set_postfix(OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('accuracy', avg_meters['accuracy'].avg)
        ]))
        pbar.update(1)
    pbar.close()
    
    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('accuracy', avg_meters['accuracy'].avg)
    ])

def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                 'iou': AverageMeter(),
                 'dice': AverageMeter(),
                 'accuracy': AverageMeter()}
    
    model.eval()
    
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()
            
            outputs = model(input)
            
            if config['deep_supervision']:
                loss = 0
                for output in outputs:
                    if output.shape[-2:] != target.shape[-2:]:
                        output = F.interpolate(output, size=target.shape[-2:],
                                             mode='bilinear', align_corners=True)
                    loss += criterion(output, target)
                loss /= len(outputs)
                output = outputs[-1]
            else:
                loss = criterion(outputs, target)
                output = outputs
            
            # Compute metrics
            iou, dice, _ = iou_score(output, target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_, accuracy_ = indicators(output, target)
            
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['accuracy'].update(accuracy_, input.size(0))
            
            pbar.set_postfix(OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('accuracy', avg_meters['accuracy'].avg)
            ]))
            pbar.update(1)
        pbar.close()
    
    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg),
        ('accuracy', avg_meters['accuracy'].avg)
    ])

def main():
    config = parse_args()
    
    # Initialize
    torch.manual_seed(config['dataseed'])
    np.random.seed(config['dataseed'])
    random.seed(config['dataseed'])
    
    os.makedirs(f"{config['output_dir']}/{config['name']}", exist_ok=True)
    writer = SummaryWriter(f"{config['output_dir']}/{config['name']}")
    
    # Create model
    model = archs.__dict__[config['arch']](
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        deep_supervision=config['deep_supervision'],
        embed_dims=config['input_list'],
        no_kan=config['no_kan']
    )
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()
    
    # Loss function
    criterion = (nn.BCEWithLogitsLoss() if config['loss'] == 'BCEWithLogitsLoss' 
                else losses.__dict__[config['loss']]()).cuda()
    
    # Optimizer
    param_groups = []
    for name, param in model.named_parameters():
        if 'layer' in name.lower() and 'fc' in name.lower():
            param_groups.append({'params': param, 
                               'lr': config['kan_lr'],
                               'weight_decay': config['kan_weight_decay']})
        else:
            param_groups.append({'params': param,
                               'lr': config['lr'],
                               'weight_decay': config['weight_decay']})
    
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)
    else:
        optimizer = optim.SGD(param_groups, 
                            lr=config['lr'],
                            momentum=config['momentum'],
                            nesterov=config['nesterov'],
                            weight_decay=config['weight_decay'])
    
    # Scheduler
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 
                                                 T_max=config['epochs'],
                                                 eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 factor=config['factor'],
                                                 patience=config['patience'],
                                                 min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                           milestones=[int(e) for e in config['milestones'].split(',')],
                                           gamma=config['gamma'])
    else:
        scheduler = None
    
    # Data loading
    img_ext = '.JPG' if config['dataset'] in ['Teeth_Final', 'DentalLast'] else '.png'
    mask_ext = '.png'
    
    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.15, random_state=config['dataseed'])
    
    train_transform = Compose([
        RandomRotate90(),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        transforms.Normalize()
    ])
    
    val_transform = Compose([
        transforms.Normalize()
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
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)
    
    train_loader = DataLoader(train_dataset,
                            batch_size=config['batch_size'],
                            shuffle=True,
                            num_workers=config['num_workers'],
                            drop_last=True)
    
    val_loader = DataLoader(val_dataset,
                          batch_size=config['batch_size'],
                          shuffle=False,
                          num_workers=config['num_workers'],
                          drop_last=False)
    
    # Training loop
    best_iou = 0
    trigger = 0
    
    for epoch in range(config['epochs']):
        print(f'Epoch [{epoch+1}/{config["epochs"]}]')
        
        train_log = train(config, train_loader, model, criterion, optimizer)
        val_log = validate(config, val_loader, model, criterion)
        
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])
        
        # Save best model
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f"{config['output_dir']}/{config['name']}/model.pth")
            best_iou = val_log['iou']
            trigger = 0
            print(f"=> Saved best model (IoU: {best_iou:.4f})")
        
        trigger += 1
        
        # Early stopping
        if config['early_stopping'] > 0 and trigger >= config['early_stopping']:
            print("=> Early stopping")
            break
        
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()