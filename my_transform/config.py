from yacs.config  import CfgNode as CN
_C = CN()

_C.train_crop = 'random'
_C.sample_size = 224
_C.train_crop_min_scale = 0.25
_C.mean = [0.485, 0.456, 0.406]
_C.std = [0.229, 0.224, 0.225]
_C.no_mean_norm = False
_C.no_std_norm = False
_C.no_hflip = False
_C.colorjitter = True
_C.input_type = 'rgb'
_C.value_scale = 0.5
_C.train_t_crop = 'random'
_C.sample_t_stride = 1
_C.sample_duration = 16
_C.train_crop_min_ratio = 0.75


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()