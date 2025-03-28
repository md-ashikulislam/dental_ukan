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
import cv2

from albumentations.augmentations import transforms
from albumentations.augmentations import geometric
import albumentations as A
from albumentations import CLAHE

from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize
from albumentations import MedianBlur

import archs
from archs import U_Net, R2U_Net, AttU_Net, R2AttU_Net

import losses
from dataset import Dataset

from metrics import iou_score, indicators, dice_coef, accuracy_score

from utils import AverageMeter, str2bool

from tensorboardX import SummaryWriter
from prettytable import PrettyTable

import shutil
import os
import subprocess

from pdb import set_trace as st


ARCH_NAMES = ['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net']
# ARCH_NAMES = archs.__all__
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
    parser.add_argument('--arch', '-a', metavar='ARCH', default='U_Net',
                       choices=['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net'])    
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
    parser.add_argument('--t', default=2, type=int,
                    help='number of recurrent steps for R2U-Net and R2AttU-Net')

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
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'accuracy': AverageMeter(),
                  'recall': AverageMeter(),
                  'specificity': AverageMeter(),
                  'precision': AverageMeter()}

    # switch to evaluate mode
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

            # Get all metrics at once
            iou, dice, recall, specificity, precision, accuracy = indicators(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['accuracy'].update(accuracy, input.size(0))
            avg_meters['recall'].update(recall, input.size(0))
            avg_meters['specificity'].update(specificity, input.size(0))
            avg_meters['precision'].update(precision, input.size(0))

            postfix = OrderedDict([
                ('l', avg_meters['loss'].avg),
                ('i', avg_meters['iou'].avg),
                ('dc', avg_meters['dice'].avg),
                ('acc', avg_meters['accuracy'].avg)  # Add accuracy to postfix
            ])

            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()


    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('accuracy', avg_meters['accuracy'].avg),
                        ('recall', avg_meters['recall'].avg),
                        ('specificity', avg_meters['specificity'].avg),
                        ('precision', avg_meters['precision'].avg)])

def log_training_images(writer, train_loader, num_images=4, global_step=0):
    """
    Log training images and masks to TensorBoard.
    
    Args:
        writer (SummaryWriter): TensorBoard writer.
        train_loader (DataLoader): Training data loader.
        num_images (int): Number of images to log.
        global_step (int): Global step for TensorBoard logging.
    """
    # Get a batch of training data
    images, masks, _ = next(iter(train_loader))
    
    # Log only the first `num_images` images and masks
    images = images[:num_images]
    masks = masks[:num_images]
    
    # Log images
    writer.add_images('train/images', images, global_step)
    
    # Log masks (convert to grayscale for visualization)
    writer.add_images('train/masks', masks, global_step)

def log_validation_images(writer, val_loader, model, num_images=4, global_step=0):
    """
    Log validation images, masks, and predictions to TensorBoard.
    
    Args:
        writer (SummaryWriter): TensorBoard writer.
        val_loader (DataLoader): Validation data loader.
        model (nn.Module): Trained model.
        num_images (int): Number of images to log.
        global_step (int): Global step for TensorBoard logging.
    """
    # Get a batch of validation data
    images, masks, _ = next(iter(val_loader))
    images = images.cuda()
    masks = masks.cuda()
    
    # Run the model to get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(images)
        if isinstance(predictions, list):  # Handle deep supervision
            predictions = predictions[-1]
    
    # Log only the first `num_images` images, masks, and predictions
    images = images[:num_images].cpu()
    masks = masks[:num_images].cpu()
    predictions = predictions[:num_images].cpu()
    
    # Log validation images
    writer.add_images('val/images', images, global_step)
    
    # Log ground truth masks (convert to grayscale for visualization)
    writer.add_images('val/masks', masks, global_step)
    
    # Log predictions (apply sigmoid if necessary and threshold at 0.5)
    predictions = torch.sigmoid(predictions)  # Apply sigmoid for binary classification
    predictions = (predictions > 0.5).float()  # Threshold at 0.5
    writer.add_images('val/predictions', predictions, global_step)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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


    if config['arch'] in ['R2U_Net', 'R2AttU_Net']:
        model = archs.__dict__[config['arch']](
            img_ch=config['input_channels'],
            output_ch=config['num_classes'],
            t=config['t']
        )
        # Initialize weights for R2U_Net or R2AttU_Net
        archs.init_weights(model, init_type='kaiming')  # or your preferred init_type
    else:
        model = archs.__dict__[config['arch']](
            img_ch=config['input_channels'],
            output_ch=config['num_classes']
        )
        # Initialize weights for other architectures
        archs.init_weights(model, init_type='kaiming')  # or your preferred init_type

    # Count parameters and print PrettyTable
    total_params = count_parameters(model)
    config['total_params'] = total_params  # Store in config for yaml

    #FOR 1 GPUs
    # model = model.cuda()

    #FOR 2 GPUs
    # Move model to multiple GPUs
    if torch.cuda.device_count() > 1:
      print(f"Using {torch.cuda.device_count()} GPUs!")
      model = torch.nn.DataParallel(model)
    model = model.cuda()  # Move to CUDA

    # model.load_state_dict(torch.load('/kaggle/input/checkpoint60/model60.pth'))


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


    dataset_name = config['dataset']

    if dataset_name == 'Dental':
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
        # A.ToGray(always_apply=True),
        MedianBlur(blur_limit=3),  # Median filter (3x3 kernel)
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),  # Add CLAHE here
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        # A.ToGray(always_apply=True),
        MedianBlur(blur_limit=3),  # Median filter (3x3 kernel)
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),  # Add CLAHE here
        transforms.Normalize(mean=[0.5], std=[0.5]),
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
    best_recall = 0
    best_specificity = 0
    best_precision = 0
    trigger = 0

    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # Log training images at the start of training
        if epoch % 10 == 0:
            log_training_images(my_writer, train_loader, global_step=epoch)

        train_log = train(config, train_loader, model, criterion, optimizer)
        val_log = validate(config, val_loader, model, criterion)

        log_validation_images(my_writer, val_loader, model, global_step=epoch)

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

        trigger += 1

        if val_log['iou'] > best_iou:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_iou': best_iou,
                'best_dice': best_dice,
                'best_accuracy': best_accuracy
            }
            torch.save(checkpoint, f'{output_dir}/{exp_name}/model.pth')
            best_accuracy = val_log['accuracy']
            best_iou = val_log['iou']
            best_dice = val_log['dice']
            best_recall = val_log['recall']
            best_specificity = val_log['specificity']
            best_precision = val_log['precision']

            print("=> saved best model")
            print('Accuracy: %.4f' % best_accuracy)
            print('IoU: %.4f' % best_iou)
            print('Dice: %.4f' % best_dice)
            print('Recall: %.4f' % best_recall)
            print('specificity: %.4f' % best_specificity)
            print('precision: %.4f' % best_precision)
            trigger = 0


        if config['early_stopping'] > 0 and trigger >= config['early_stopping']:
             print("=> early stopping")
             break

        # Thorough memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
if __name__ == '__main__':
    main()
