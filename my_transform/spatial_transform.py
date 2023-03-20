import random
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from PIL import Image
import torchvision 
import torch

class Compose(transforms.Compose):

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(transforms.ToTensor):

    def randomize_parameters(self):
        pass


class Normalize(transforms.Normalize):
    def randomize_parameters(self):
        pass


class ScaleValue(object):

    def __init__(self, s):
        self.s = s

    def __call__(self, tensor):
        tensor *= self.s
        return tensor

    def randomize_parameters(self):
        pass


class Resize(transforms.Resize):

    def randomize_parameters(self):
        pass


class Scale(transforms.Resize):

    def randomize_parameters(self):
        pass


class CenterCrop(transforms.CenterCrop):

    def randomize_parameters(self):
        pass


class CornerCrop(object):

    def __init__(self,
                 size,
                 crop_position=None,
                 crop_positions=['c', 'tl', 'tr', 'bl', 'br']):
        self.size = size
        self.crop_position = crop_position
        self.crop_positions = crop_positions

        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.randomize_parameters()

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]

        h, w = (self.size, self.size)
        if self.crop_position == 'c':
            i = int(round((image_height - h) / 2.))
            j = int(round((image_width - w) / 2.))
        elif self.crop_position == 'tl':
            i = 0
            j = 0
        elif self.crop_position == 'tr':
            i = 0
            j = image_width - self.size
        elif self.crop_position == 'bl':
            i = image_height - self.size
            j = 0
        elif self.crop_position == 'br':
            i = image_height - self.size
            j = image_width - self.size

        img = F.crop(img, i, j, h, w)

        return img

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, crop_position={1}, randomize={2})'.format(
            self.size, self.crop_position, self.randomize)


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __init__(self, p=0.5):
        super().__init__(p)
        self.randomize_parameters()

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.random_p < self.p:
            return F.hflip(img)
        return img

    def randomize_parameters(self):
        self.random_p = random.random()


class MultiScaleCornerCrop(object):

    def __init__(self,
                 size,
                 scales,
                 crop_positions=['c', 'tl', 'tr', 'bl', 'br'],
                 interpolation=Image.BILINEAR):
        self.size = size
        self.scales = scales
        self.interpolation = interpolation
        self.crop_positions = crop_positions

        self.randomize_parameters()

    def __call__(self, img):
        short_side = min(img.size[0], img.size[1])
        crop_size = int(short_side * self.scale)
        self.corner_crop.size = crop_size

        img = self.corner_crop(img)
        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        crop_position = self.crop_positions[random.randint(
            0,
            len(self.crop_positions) - 1)]

        self.corner_crop = CornerCrop(None, crop_position)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, scales={1}, interpolation={2})'.format(
            self.size, self.scales, self.interpolation)


class RandomResizedCrop(transforms.RandomResizedCrop):

    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation=Image.BILINEAR):
        super().__init__(size, scale, ratio, interpolation)
        self.randomize_parameters()

    # Take tensor img input
    def __call__(self, img):
        if self.randomize:
            self.random_crop = self.get_params(img, self.scale, self.ratio)
            self.randomize = False

        i, j, h, w = self.random_crop
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def randomize_parameters(self):
        self.randomize = True


class ColorJitter(transforms.ColorJitter):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.randomize_parameters()

    def __call__(self, img):
        if self.randomize:
            
            info = transforms.ColorJitter.get_params(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.1, 0.1))

            brightness = info[1]
            contrast = info[2]
            saturation = info[3]
            hue = info[4]

            self.randomize = False
            img_transforms=[]
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            jittered_img = img
            for func in img_transforms:
                jittered_img = func(jittered_img)
            # convert to numpy 
            return jittered_img.numpy().transpose(2,0,1)

    def randomize_parameters(self):
        self.randomize = True


class PickFirstChannels(object):

    def __init__(self, n):
        self.n = n

    def __call__(self, tensor):
        return tensor[:self.n, :, :]

    def randomize_parameters(self):
        pass


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
