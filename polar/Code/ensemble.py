from utils import *

import os
import argparse

import torchvision.models
from PIL import Image
from datetime import datetime, timezone
import pytz

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torchmetrics import F1Score

import numpy as np
import pandas as pd
from Datasets import EnsembleDataSet, CustomDataSet
from torch.utils.data import DataLoader
from Models import CustomModel

from sklearn.model_selection import train_test_split

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize, CenterCrop


def main(args):
    pass


if __name__ == '__main__':
    args = args_getter()
    non_list = ['name', 'mode', 'save']
    hypers = {k: v for k, v in vars(args).items() if k not in non_list}
    print(f"Hyper Parameter : {hypers}")
    print(f"Model Save name : {args.name}")

    print(f"Current Using GPU : {torch.cuda.get_device_name(0)}")

    main(args)
