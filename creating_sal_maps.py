#this file creates saliency maps for sal.txt images using the DINO trained model.

import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO
import wandb
from numba import jit, cuda

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import PIL
from matplotlib import pyplot as plt

import visualisation_utils as utils
import vision_transformer as vits

def create_sal_map(img, fname, args, og_img):

    print("The size of image ", img.size())
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size #480,480
    img = img[:, :w, :h].unsqueeze(0) #1,3,480,480


    w_featmap = img.shape[-2] // args.patch_size #60
    h_featmap = img.shape[-1] // args.patch_size #60

    attentions = model.get_last_selfattention(img.to(device)) #(1, 6, 3601, 3601)

    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1) #(num_heads x num_patches ) Taking only the class attention from all the head. (6, 3600)
			
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy() #(num_heads x image_size x image_size) (6, 480, 480)
    
    # save attentions heatmaps
    main_merge = np.zeros(attentions[0].shape) #480,480
    full_merge = np.zeros(attentions[0].shape)
    
    maxx = 0
    for j in range(nh):
        att_fname = os.path.join(args.temp_output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=att_fname, arr=attentions[j], format='png')
        norm_attention = (attentions[j] - np.min(attentions[j])) / (np.max(attentions[j]) - np.min(attentions[j]))

        new_image = cv2.imread(att_fname)
        os.remove(att_fname) #uncomment if you wish to save the attention maps from each head for every image

        grey_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

        (T, thresh) = cv2.threshold(grey_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if np.sum(thresh)>np.sum(maxx):
          maxx = thresh
          max_head = j
        if j==0:
         merge = thresh 
        elif j==1:
         merge = np.concatenate((np.expand_dims(merge, axis=0),np.expand_dims(thresh, axis=0)), axis=0)
        else:
         merge = np.concatenate((merge,np.expand_dims(thresh, axis=0)), axis=0)


    for i in range(merge.shape[0]):
      if i==max_head:
        continue
      else:
        main_merge = (np.logical_or(main_merge, merge[i]))*1
    for i in range(main_merge.shape[0]):
      for j in range(main_merge.shape[1]):
        if main_merge[i][j] == 1:
          main_merge[i][j] = 255
        else:
          main_merge[i][j] = 0
    main_merge = np.array(main_merge, dtype=np.uint8)
    Image.fromarray(main_merge).save(os.path.join(args.output_dir, fname+'.png'))
    print("Saved at ", os.path.join(args.output_dir, fname+'.png'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser('Creating saliency maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument('--log_name', type=str, default="log", help="log name for wandb")
    parser.add_argument('--log_results', action = 'store_true', help="To log results in wandb")
    parser.add_argument("--temp_output_dir", default='temp_data', help = 'temp output directory to save the attentions')
    parser.add_argument("--root_image_dir", help = 'Path to the root images')
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    if args.log_results:
        wandb.init(project="incre-seg",entity="incremental_seg", name=args.log_name)
        wandb.config.update(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir)
    # build model
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    print("New images are being stored at", args.output_dir)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    img_dir = os.path.join(args.root_image_dir, 'test_images_sal_map')
    if not os.path.isdir(args.root_image_dir):
      raise RuntimeError(f'The dataset from the path {img_dir} not found. It might be corrupted')
    
    file_names = open(os.path.join(args.root_image_dir, 'ImageSets/Segmentation', 'sal_list.txt'), 'r')
        
    file_names = file_names.read().splitlines() #a list with names of the files

    t = 0 
    for f in file_names:
      t+=1
      f_name = os.path.join(img_dir, f + ".jpg")
      img = Image.open(f_name) 
      img = img.convert('RGB')
      og_img = skimage.io.imread(f_name)
      og_img = skimage.transform.resize(og_img, (480, 480), anti_aliasing=True)
      transform = pth_transforms.Compose([
          pth_transforms.Resize(args.image_size),
          pth_transforms.ToTensor(),
          pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
      ])
      img = transform(img)
      create_sal_map(img,f, args, og_img)
    print(f"Created total {t} saliency maps")

    
