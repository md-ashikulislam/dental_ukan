
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

    dataset_name = config['dataset']
    if dataset_name == 'Dental' or dataset_name == 'new_dataset':
       img_ext = '.JPG'  # Update for teeth dataset
       mask_ext = '.jpg'


    # Data loading code
    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    # img_ids.sort()
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.15, random_state=config['dataseed'])

    # ckpt = torch.load(f'{args.output_dir}/{args.name}/model.pth')
    # ckpt = torch.load(f'{args.output_dir}/{args.name}/model.pth', weights_only=True)
    ckpt = torch.load('/kaggle/input/checkukan/UKAN150/model.pth')


    try:        
        model.load_state_dict(ckpt['state_dict'])
    except:
        # print("Pretrained model keys:", ckpt.keys())
        # print("Current model keys:", model.state_dict().keys())

        pretrained_dict = {k: v for k, v in ckpt.items() if k in model.state_dict()}
        current_dict = model.state_dict()
        diff_keys = set(current_dict.keys()) - set(pretrained_dict.keys())

        # print("Difference in model keys:")
        # for key in diff_keys:
        #     print(f"Key: {key}")

        model.load_state_dict(ckpt, strict=False)
 
    model.eval()

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
        for i, (input, target, meta) in enumerate(tqdm(val_loader, total=len(val_loader))): 
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)

            if i == 0:
                plt.figure(figsize=(15,5))
                plt.subplot(1,3,1)
                plt.imshow(input[0].cpu().numpy().transpose(1,2,0))
                plt.title('Input')
                plt.subplot(1,3,2)
                plt.imshow(target[0,0].cpu().numpy(), cmap='gray')
                plt.title('Ground Truth')
                plt.subplot(1,3,3)
                plt.imshow(torch.sigmoid(output[0,0]).cpu().numpy(), cmap='gray')
                plt.title('Prediction')
                plt.close()

            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            print("Output values before sigmoid:", output.cpu().numpy())

            output = torch.sigmoid(output).cpu().numpy()

            print("Output values before sigmoid:", output.cpu().numpy())


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
    # print('HD95: %.4f' % hd95_avg_meter.avg)



if __name__ == '__main__':
    main()
