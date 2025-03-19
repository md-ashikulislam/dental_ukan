import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations import Compose, RandomRotate90, Resize, Normalize
import albumentations as A
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import archs
import losses
from dataset import Dataset
from metrics import iou_score, indicators
from utils import AverageMeter, str2bool
from tensorboardX import SummaryWriter

import shutil
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.amp as amp  # For mixed precision

# Import the required model directly
from archs import UKAN

ARCH_NAMES = ['UKAN']
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=15, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--dataseed', default=2981, type=int, help='')

    # Model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UKAN')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=1, type=int, help='number of classes')
    parser.add_argument('--input_w', default=256, type=int, help='image width')
    parser.add_argument('--input_h', default=256, type=int, help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    # Loss
    parser.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES, help='loss: ' + ' | '.join(LOSS_NAMES) + ' (default: BCEDiceLoss)')

    # Dataset
    parser.add_argument('--dataset', default='busi', help='dataset name')
    parser.add_argument('--data_dir', default='inputs', help='dataset dir')
    parser.add_argument('--output_dir', default='outputs', help='output dir')

    # Optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], help='optimizer: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')
    parser.add_argument('--kan_lr', default=1e-2, type=float, metavar='LR', help='initial learning rate for KAN layers')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float, help='weight decay for KAN layers')

    # Scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR', choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--no_kan', action='store_true')

    config = parser.parse_args()
    return config

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'accuracy': AverageMeter()}
    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.to(xm.xla_device())
        target = target.to(xm.xla_device())

        # Mixed precision training
        with amp.autocast():
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice, _ = iou_score(outputs[-1], target)
                iou_, dice_, hd_, hd95_, recall_, specificity_, precision_, accuracy_ = indicators(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice, _ = iou_score(output, target)
                iou_, dice_, hd_, hd95_, recall_, specificity_, precision_, accuracy_ = indicators(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        xm.optimizer_step(optimizer)

        # Update metrics
        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['accuracy'].update(accuracy_, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('accuracy', avg_meters['accuracy'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg), ('accuracy', avg_meters['accuracy'].avg)])

def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter(), 'accuracy': AverageMeter()}
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(xm.xla_device())
            target = target.to(xm.xla_device())

            with amp.autocast():
                if config['deep_supervision']:
                    outputs = model(input)
                    loss = 0
                    for output in outputs:
                        loss += criterion(output, target)
                    loss /= len(outputs)
                    iou, dice, _ = iou_score(outputs[-1], target)
                    iou_, dice_, hd_, hd95_, recall_, specificity_, precision_, accuracy_ = indicators(outputs[-1], target)
                else:
                    output = model(input)
                    loss = criterion(output, target)
                    iou, dice, _ = iou_score(output, target)
                    iou_, dice_, hd_, hd95_, recall_, specificity_, precision_, accuracy_ = indicators(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['accuracy'].update(accuracy_, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('accuracy', avg_meters['accuracy'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg), ('dice', avg_meters['dice'].avg), ('accuracy', avg_meters['accuracy'].avg)])

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
    """
    Set random seed for reproducibility on CPU, GPU, and TPU.
    """
    # Set seed for Python's random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch CPU
    torch.manual_seed(seed)
    
    # Set seed for PyTorch GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    # Set seed for PyTorch TPU
    if xm.xla_device().type == 'xla':
        xm.set_rng_state(seed)

def main():

    # Set random seed for reproducibility
    seed_torch(seed=1029)

    # Parse arguments
    config = vars(parse_args())
    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')
    
    # Set up experiment name and output directory
    exp_name = config.get('name')
    output_dir = config.get('output_dir')

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    
    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    # Print configuration
    xm.master_print('-' * 20)
    for key in config:
        xm.master_print(f'{key}: {config[key]}')
    xm.master_print('-' * 20)

    # Save configuration to file
    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # Define loss function
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(xm.xla_device())
    else:
        criterion = losses.__dict__[config['loss']]().to(xm.xla_device())

    # Create model and move to TPU
    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'], embed_dims=config['input_list'], no_kan=config['no_kan'])
    model = model.to(xm.xla_device())

    # Optimizer
    param_groups = []
    for name, param in model.named_parameters():
        if 'layer' in name.lower() and 'fc' in name.lower():  # Higher LR for KAN layers
            param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']})
        else:
            param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(param_groups, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    # Scheduler
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading
    dataset_name = config['dataset']
    img_ext = '.png'  # Default for most datasets
    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
    elif dataset_name in ['Dental', 'Enhanced_Dental', 'Resized_Teeth']:
        img_ext = '.JPG'
        mask_ext = '.jpg'
    elif dataset_name == 'ph2':
        img_ext = '.bmp'
        mask_ext = '.bmp'
    elif dataset_name == 'HAM':
        img_ext = '.jpg'
        mask_ext = '_segmentation.png'

    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'and_images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.15, random_state=config['dataseed'])

    train_transform = Compose([
        RandomRotate90(),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'and_images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform)

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'and_images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)

    # Wrap DataLoader with MpDeviceLoader for TPU
    train_loader = pl.MpDeviceLoader(
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            drop_last=True),
        device=xm.xla_device())

    val_loader = pl.MpDeviceLoader(
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=False),
        device=xm.xla_device())

    # Logging
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('accuracy', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('val_accuracy', []),
    ])

    best_iou = 0
    best_dice = 0
    best_accuracy = 0
    trigger = 0

    # Training loop
    for epoch in range(config['epochs']):
        xm.master_print(f'Epoch [{epoch}/{config["epochs"]}]')

        # Log training images at the start of training
        if epoch == 0:
            log_training_images(my_writer, train_loader, global_step=epoch)
        # Train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)

        # Evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)
        # Log validation images and predictions
        log_validation_images(my_writer, val_loader, model, global_step=epoch)

        # Update scheduler
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        # Log metrics
        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['accuracy'].append(train_log['accuracy'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_accuracy'].append(val_log['accuracy'])

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

        # Save best model
        if val_log['iou'] > best_iou:
            xm.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')
            best_iou = val_log['iou']
            best_dice = val_log['dice']
            best_accuracy = val_log['accuracy']
            xm.master_print(f'=> saved best model with IoU: {best_iou:.4f}, Dice: {best_dice:.4f}, Accuracy: {best_accuracy:.4f}')
            trigger = 0

        # Early stopping
        trigger += 1
        if config['early_stopping'] > 0 and trigger >= config['early_stopping']:
            xm.master_print("=> early stopping")
            break

        # Synchronize TPU cores
        xm.rendezvous('end_of_epoch')


# Launch the training process
if __name__ == '__main__':
    # xmp.spawn(main, nprocs=8)  # Use 8 TPU 
    main()  # No need for xmp.spawn()
