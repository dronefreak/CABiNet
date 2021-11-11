
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import datetime
from model_final_mv3 import DABLNet
import statistics as stat
from visualize import get_color_pallete
import os
parse = argparse.ArgumentParser()
parse.add_argument(
        '--ckpt',
        dest='ckpt',
        type=str,
        default='./res/model_non_local.pth',)

parse.add_argument(
        '--data_dir',
        dest='data_dir',
        type=str,
        default='./val/munster',)

parse.add_argument(
        '--save_dir',
        dest='save_dir',
        type=str,
        default='./outputs',)

args = parse.parse_args()
os.makedirs(args.save_dir, exist_ok=True)

# define model
net = DABLNet(n_classes=19)
net.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
net.eval()
net.cuda()

# prepare data
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

image_list = os.listdir(args.data_dir)
for image in image_list:
    im = to_tensor(Image.open(os.path.join(args.data_dir, image)).convert('RGB')).unsqueeze(0).cuda()
    out = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
    mask = get_color_pallete(out, 'citys')
    save_path = os.path.join(args.save_dir, image)
    #cv2.imwrite(save_path, out)
    mask.save(save_path)