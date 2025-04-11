
import argparse
import os
from glob import glob
import random
import numpy as np
from albumentations import MedianBlur
from albumentations import CLAHE
import matplotlib.pyplot as plt


import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import OrderedDict

import archs

from dataset import Dataset
from metrics import iou_score, dice_coef
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time

from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--output_dir', default='outputs', help='output dir')
    parser.add_argument('--checkpoint_path')
            
    args = parser.parse_args()

    return args

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    args = parse_args()

    # with open(f'{args.output_dir}/{args.name}/config.yml', 'r') as f:
    with open('/kaggle/input/checkukan/UKAN150/config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'], embed_dims=config['input_list'])
    # model = model.cuda()  # Move to CUDA

    model = model.cuda()

    checkpoint = torch.load(args.checkpoint_path, weights_only=True)

    # Remove 'module.' prefix from checkpoint keys if present
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Load the state dictionary into the model
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("Attempting to load with strict=False")
        model.load_state_dict(state_dict, strict=False)

    model.eval()


    dataset_name = config['dataset']
    if dataset_name == 'Dental' or dataset_name == 'new_dataset':
       img_ext = '.JPG'  # Update for teeth dataset
       mask_ext = '.jpg'

    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.15, random_state=config['dataseed'])

  


    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        MedianBlur(blur_limit=3),  # Median filter (3x3 kernel)
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),  # Add CLAHE here
        transforms.Normalize(mean=[0.5], std=[0.5]),    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=True)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    # hd95_avg_meter = AverageMeter()

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)

            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output)

            output[output>=0.5]=1
            output[output<0.5]=0

            os.makedirs(os.path.join(args.output_dir, config['name'], 'out_val'), exist_ok=True)
            for pred, img_id in zip(output, meta['img_id']):
                pred_np = pred[0].astype(np.uint8)
                pred_np = pred_np * 255
                img = Image.fromarray(pred_np, 'L')
                img.save(os.path.join(args.output_dir, config['name'], 'out_val/{}.jpg'.format(img_id)))

    print(config['name'])
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)



if __name__ == '__main__':
    main()

# import os
# import argparse
# import yaml
# import numpy as np
# import torch
# import torch.backends.cudnn as cudnn
# from glob import glob
# from tqdm import tqdm
# from PIL import Image
# import cv2
# from albumentations import Compose, Resize, MedianBlur, CLAHE
# from albumentations.augmentations import transforms
# import archs
# from dataset import Dataset

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--name', default='Dental_UKAN_woDS', help='model name (as used in training)')
#     parser.add_argument('--output_dir', default='/kaggle/working/predictions', help='directory to save predictions')
#     parser.add_argument('--checkpoint_path', default='/kaggle/working/outputs/Dental_UKAN_woDS/model.pth', help='path to model checkpoint')
#     parser.add_argument('--data_dir', default='inputs', help='dataset directory')
#     parser.add_argument('--dataset', default='Dental', help='dataset name')
#     parser.add_argument('--config_path')
#     return parser.parse_args()

# def main():
#     args = parse_args()

#     # Load config from training
#     config_path = f'{args.config_path}'
#     with open(config_path, 'r') as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)

#     print('-' * 20)
#     for key in config.keys():
#         print(f'{key}: {config[key]}')
#     print('-' * 20)

#     # Enable cudnn benchmark
#     cudnn.benchmark = True

#     # Create model
#     model = archs.__dict__[config['arch']](
#         config['num_classes'],
#         config['input_channels'],
#         config['deep_supervision'],
#         embed_dims=config['input_list'],
#         no_kan=config.get('no_kan', False)
#     )
#     model = model.cuda()

#     checkpoint = torch.load(args.checkpoint_path, weights_only=True)

#     # Remove 'module.' prefix from checkpoint keys if present
#     state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
#     if any(key.startswith('module.') for key in state_dict.keys()):
#         state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

#     # Load the state dictionary into the model
#     try:
#         model.load_state_dict(state_dict, strict=True)
#     except RuntimeError as e:
#         print(f"Error loading state dict: {e}")
#         print("Attempting to load with strict=False")
#         model.load_state_dict(state_dict, strict=False)

#     model.eval()

#     # Define dataset extensions
#     img_ext = '.JPG' if config['dataset'] in ['Dental', 'new_dataset'] else '.png'
#     mask_ext = '.jpg'

#     # Get all image IDs
#     img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
#     img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

#     # Define validation transform (same as training)
#     val_transform = Compose([
#         Resize(config['input_h'], config['input_w']),
#         MedianBlur(blur_limit=3),
#         CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
#         transforms.Normalize(mean=[0.5], std=[0.5]),
#     ])

#     # Create dataset
#     dataset = Dataset(
#         img_ids=img_ids,
#         img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
#         mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
#         img_ext=img_ext,
#         mask_ext=mask_ext,
#         num_classes=config['num_classes'],
#         transform=val_transform
#     )

#     # Create data loader
#     data_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=config['batch_size'],
#         shuffle=False,
#         num_workers=config['num_workers'],
#         drop_last=False
#     )

#     # Create output directory
#     os.makedirs(args.output_dir, exist_ok=True)

#     # Predict and save masks
#     with torch.no_grad():
#         for input, _, meta in tqdm(data_loader, total=len(data_loader)):
#             input = input.cuda()
#             # Compute output
#             if config['deep_supervision']:
#                 outputs = model(input)
#                 output = outputs[-1]  # Use final output
#             else:
#                 output = model(input)

#             output = torch.sigmoid(output).cpu().numpy()
#             output = (output >= 0.5).astype(np.uint8)  # Threshold at 0.5

#             # Save predictions
#             for pred, img_id in zip(output, meta['img_id']):
#                 pred_np = pred[0] * 255  # Convert to 0-255 for saving as image
#                 img = Image.fromarray(pred_np, 'L')
#                 img.save(os.path.join(args.output_dir, f'{img_id}.jpg'))

#     print(f'Predictions saved to {args.output_dir}')

# if __name__ == '__main__':
#     main()