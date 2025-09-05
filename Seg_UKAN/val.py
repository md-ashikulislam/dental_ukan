# import argparse
# import os
# from glob import glob
# import random
# import numpy as np
# from albumentations import MedianBlur, CLAHE
# import matplotlib.pyplot as plt
# import cv2
# import torch
# import yaml
# from albumentations.augmentations import transforms
# from albumentations.core.composition import Compose
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm
# from collections import OrderedDict
# import archs
# from dataset import Dataset
# from metrics import iou_score, dice_coef
# from utils import AverageMeter
# from albumentations import RandomRotate90, Resize
# import time
# from PIL import Image

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--name', default=None, help='model name')
#     parser.add_argument('--output_dir', default='outputs', help='output dir')
#     parser.add_argument('--model_path')
#     parser.add_argument('--config_path')
#     parser.add_argument('--threshold', type=float)

#     args = parser.parse_args()
#     return args

# def seed_torch(seed=1029):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

# def main():
#     seed_torch()
#     args = parse_args()

#     # Load configuration
#     with open(args.config_path) as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)

#     print('-'*20)
#     for key in config.keys():
#         print('%s: %s' % (key, str(config[key])))
#     print('-'*20)

#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Initialize model
#     model = archs.__dict__[config['arch']](
#         config['num_classes'],
#         config['input_channels'],
#         config['deep_supervision'],
#         embed_dims=config['input_list']
#     )
#     model = model.to(device)

#     # Load checkpoint
#     checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)

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

#     # Dataset configuration
#     dataset_name = config['dataset']
#     if dataset_name in ['Dental', 'new_dataset']:
#         img_ext = '.JPG'
#         mask_ext = '.jpg'

#     img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
#     img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

#     _, val_img_ids = train_test_split(img_ids, test_size=0.15, random_state=config['dataseed'])

#     # Validation transform
#     val_transform = Compose([
#         Resize(config['input_h'], config['input_w']),
#         MedianBlur(blur_limit=3),
#         CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
#         transforms.Normalize(mean=[0.5], std=[0.5]),
#     ])

#     # Validation dataset and loader
#     val_dataset = Dataset(
#         img_ids=val_img_ids,
#         img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
#         mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
#         img_ext=img_ext,
#         mask_ext=mask_ext,
#         num_classes=config['num_classes'],
#         transform=val_transform
#     )
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset,
#         batch_size=config['batch_size'],
#         shuffle=False,
#         num_workers=config['num_workers'],
#         drop_last=True
#     )

#     # Metrics
#     iou_avg_meter = AverageMeter()
#     dice_avg_meter = AverageMeter()

#     # Evaluation loop
#     with torch.no_grad():
#         for input, target, meta in tqdm(val_loader, total=len(val_loader)):
#             input = input.to(device)
#             target = target.to(device)

#             output = model(input)

#             iou = iou_score(output, target)
#             dice = dice_coef(output, target)
#             iou_avg_meter.update(iou, input.size(0))
#             dice_avg_meter.update(dice, input.size(0))

#             output = torch.sigmoid(output).cpu().numpy()
#             output = (output >= args.threshold).astype(np.uint8)

#             # Save predictions
#             os.makedirs(os.path.join(args.output_dir, config['name'], 'out_val'), exist_ok=True)
#             for pred, img_id in zip(output, meta['img_id']):
#                 pred_np = pred[0] * 255
#                 img = Image.fromarray(pred_np, 'L')
#                 img.save(os.path.join(args.output_dir, config['name'], 'out_val/{}.jpg'.format(img_id)))

#     # Print results
#     print(config['name'])
#     print('IoU: %.4f' % iou_avg_meter.avg)
#     print('Dice: %.4f' % dice_avg_meter.avg)

# if __name__ == '__main__':
#     main()











#### Single image Prediction

import argparse
import os
import random
import numpy as np
import albumentations as A

from albumentations import MedianBlur, CLAHE
import cv2
import torch
import yaml
from albumentations.core.composition import Compose
from tqdm import tqdm
import archs
from dataset import Dataset
from metrics import iou_score, dice_coef
from utils import AverageMeter
from albumentations import Resize
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--output_dir', default='outputs', help='output dir')
    parser.add_argument('--model_path')
    parser.add_argument('--config_path')
    parser.add_argument('--image_id', required=True, help='ID of the image to predict')
    args = parser.parse_args()
    return args

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    seed_torch()
    args = parse_args()

    # Load configuration
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list']
    )
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)

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

    # Dataset configuration
    dataset_name = config['dataset']
    if dataset_name in ['Dental', 'new_dataset']:
        img_ext = '.JPG'
        mask_ext = '.jpg'

    # Validation transform
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        MedianBlur(blur_limit=3),
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
        A.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Single image dataset
    val_dataset = Dataset(
        img_ids=[args.image_id],
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # Force batch size to 1 for single image
        shuffle=False,
        num_workers=config['num_workers']
    )

    # Metrics
    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()

    # Prediction for single image
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)

            output = model(input)

            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output = (output >= 0.5).astype(np.uint8)

            # Save prediction
            os.makedirs(os.path.join(args.output_dir, config['name'], 'out_val'), exist_ok=True)
            pred = output[0]  # Single image
            pred_np = pred[0] * 255
            img = Image.fromarray(pred_np, 'L')
            img.save(os.path.join(args.output_dir, config['name'], 'out_val/{}.jpg'.format(args.image_id)))

    # Print results
    print(config['name'])
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

if __name__ == '__main__':
    main()