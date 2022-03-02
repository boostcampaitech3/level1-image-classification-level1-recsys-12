import os
import argparse
import random
import glob
import re
from distutils.util import strtobool

import numpy as np
import torch

from dotenv import load_dotenv
from pprint import pprint
from pathlib import Path


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            key, value = kv.split("=")

            for t in (int, float):
                try:
                    value = t(value)
                except ValueError:
                    pass
                else:
                    break
            if isinstance(value, str):
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    print(value.lower())
                    value = False
            my_dict[key] = value
        setattr(namespace, self.dest, my_dict)


def args_getter():
    parser = argparse.ArgumentParser()

    load_dotenv(verbose=True)

    # - Data and Model
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--model', type=str, default='ResNet', help='model type (default: ResNet)')
    parser.add_argument('--version', type=int, default=18)
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset',
                        help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--scheduler', type=str, default='StepLR', help='Learning rate scheduler (default: StepLR)')
    parser.add_argument('--patience', type=int, default=10)

    # - Hyperparameter
    # -- Basic hp
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='size for test(validation) (default: 0.2)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default64)')
    parser.add_argument('--valid_batch_size', type=int, default=32,
                        help='input batch size for validating (default: 1000)')
    parser.add_argument('--drop_last', type=lambda x: bool(strtobool(x)), default=True,
                        help='set the mini-batch drop_last parameter (default: True)')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--n_split', type=int, default=5, help='K-Fold split value')

    # -- augmentation hp
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation',
                        help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--flip_ratio', type=float, default=0.5,
                        help='HorizontalFlip or VerticalFlip probability (default: 0.5)')
    parser.add_argument('--resize', nargs='+', type=int, default=[512, 384],
                        help='resize size for image when training (default: 512 384)')
    parser.add_argument('--beta', type=float, default=1.0, help='CutMix beta value (default: 1.0)')

    # -- scheduler hp
    parser.add_argument('--sch_params', dest='sch_params', default="Step_size=20,gamma=1.0",
                        action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2",
                        help='Your scheduler parameters', required=True)

    # - Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test_data_dir', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--record_dir', type=str, default=os.environ.get('SM_RECORD_DIR'))

    # - wandb and etc
    parser.add_argument('--wandb', type=lambda x: bool(strtobool(x)), default=False,
                        help='WandB use setting (default: False)')
    parser.add_argument('--record', type=lambda x: bool(strtobool(x)), default=False,
                        help='Record in README or record_dir (default: False)')
    parser.add_argument('--name', required=True, help='model save at {SM_MODEL_DIR}/{name}')

    args = parser.parse_args()

    pprint(vars(args))

    return args


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def increment_path(path: str, args, exist_ok: bool = False) -> str:
    """
    Automatically increment path, i.e. runs/exp --> runs/exp0, runs/epx1 etc.

    Args:
        args: arguments
        path (str or pathlib.Path): f"{model_dir}/{args.name}"
        exist_ok (bool): whether increment path (increment if False)

    Returns:
        str: increment path name
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}_{args.criterion}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}_{args.criterion}_{n}"
