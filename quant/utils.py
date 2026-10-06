"""Small shared quantization utilities; optional config dependencies load lazily."""
import pickle
from pathlib import Path
import torch

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x): return x.round()
    @staticmethod
    def backward(ctx,grad): return grad

Round = RoundSTE.apply
LINEAR_SHIFT_NUM = 2 ** 32
MATMUL_SHIFT_NUM = 2 ** 20

def load_config(path):
    import yaml
    with open(path,encoding="utf-8") as f: return yaml.safe_load(f)

def save_scales(scales,scale_dir,layer_name):
    root=Path(scale_dir);root.mkdir(parents=True,exist_ok=True)
    for name,value in scales.items():
        with (root/f"{layer_name}_{name}.p").open("wb") as f: pickle.dump(value,f)

def load_scales(scale_dir,layer_name,scale_types):
    result={}
    for name in scale_types:
        with (Path(scale_dir)/f"{layer_name}_{name}.p").open("rb") as f: result[name]=pickle.load(f)
    return result
