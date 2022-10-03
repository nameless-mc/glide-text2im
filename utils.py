import os
from PIL import Image
import torch as th
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def show_images_in_dir(batch: th.Tensor, dir: str, fname: str):
    scaled = ((batch + 1)*127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    os.makedirs(dir, exist_ok=True)
    Image.fromarray(reshaped.numpy()).save(f'./results/img/{fname}')


def show_images(batch: th.Tensor, fname: str):
    show_images_in_dir(batch, './results/img', fname)
