import cv2
import os
import numpy as np
import torch
from spatial_transform import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)

import imgaug.augmenters as iaa
from config import get_cfg_defaults
opt = get_cfg_defaults()

video_path = "my_transform/UCF-test/v_ApplyEyeMakeup_g01_c01.avi"
output_dir = "my_transform/UCF-frames"

def avi2frames(video_path = "UCF-test/v_ApplyEyeMakeup_g01_c01.avi",output_dir = "UCF-frames"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if ret:
            output_filename = os.path.join(output_dir, f"frame_{frame_idx:04}.jpg")
            cv2.imwrite(output_filename, frame)
        else:
            break
    cap.release()

def get_normalize_method(data,mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            m = [0, 0, 0]
            s = [1, 1, 1]
        else:
            m = [0, 0, 0]
            s =std
    else:
        if no_std_norm:
            m = mean
            s = [1, 1, 1]
        else:
            m = mean
            s = std
    for d, m, s in zip(data, m, s):
        d.sub_(m).div_(s) 
    return data

# Setting for train

def get_train_utils(img,opt):
    assert opt.train_crop in ['random', 'corner', 'center']
    spatial_transform = []
    if opt.train_crop == 'random':
        spatial_transform.append(
            RandomResizedCrop(
                opt.sample_size, (opt.train_crop_min_scale, 1.0),
                (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio)))
    elif opt.train_crop == 'corner':
        scales = [1.0]
        scale_step = 1 / (2**(1 / 4))
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        spatial_transform.append(MultiScaleCornerCrop(opt.sample_size, scales))
    elif opt.train_crop == 'center':
        spatial_transform.append(Resize(opt.sample_size))
        spatial_transform.append(CenterCrop(opt.sample_size))

    if not opt.no_hflip:
        spatial_transform.append(RandomHorizontalFlip())
    if opt.colorjitter:
        spatial_transform.append(ColorJitter())
    spatial_transform.append(ToTensor())
   
    spatial_transform.append(ScaleValue(opt.value_scale))
    spatial_transform = Compose(spatial_transform)

    new_img = spatial_transform (img)
    new_img = get_normalize_method(new_img,opt.mean, opt.std, opt.no_mean_norm,opt.no_std_norm)
    return new_img

def transform_multi_frames_train(data,opt):
    tensor_list = []
    for img in data:
        new_img = get_train_utils(img,opt)
        tensor_list.append(new_img.permute(2,0,1))
    tensor_data = torch.stack(tensor_list, dim=0)
    return tensor_data

def transform_multi_frames_val(data,opt):
    tensor_list = []
    for img in data:
        new_img = get_val_utils(img,opt)
        tensor_list.append(new_img)
    tensor_data = torch.stack(tensor_list, dim=0)
    return tensor_data

def get_val_utils(img,opt):
    spatial_transform1 = [
        Resize(opt.sample_size),
        CenterCrop(opt.sample_size),
    ]
    spatial_transform1 = Compose(spatial_transform1)

    spatial_transform2=[]
    spatial_transform2.append(ToTensor())
    spatial_transform2.append(ScaleValue(opt.value_scale))
    spatial_transform2 = Compose(spatial_transform2)

    new_img = spatial_transform1 (img)
    new_img = new_img.numpy().transpose(1,2,0)
    new_img = spatial_transform2(new_img)
    new_img = get_normalize_method(new_img,opt.mean, opt.std, opt.no_mean_norm,opt.no_std_norm)
    return new_img


#Convert frames in folder to torch.tensor
frames_dir = "my_transform/UCF-frames"
tensor_list = []
for filename in os.listdir(frames_dir):
    img = cv2.imread(os.path.join(frames_dir,filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    clip = torch.from_numpy(np.asarray(img).transpose(2,0,1))
    tensor_list.append(clip)
tensor_data = torch.stack(tensor_list, dim=0)

#Transform multi frames
dat = transform_multi_frames_train(tensor_data,opt)
# dat = transform_multi_frames_val(tensor_data,opt)
print(dat.shape)
