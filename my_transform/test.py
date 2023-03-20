import cv2
import os
import numpy as np
import torch
from spatial_transform import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from spatial_transform import transform_multi_frames_train,transform_multi_frames_val

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
