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


def train(args, data_loaders):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gender_model = torchvision.models.vgg13(pretrained=False)
    age_model = torchvision.models.vgg16(pretrained=False)
    mask_model = torchvision.models.vgg16(pretrained=False)

    models = [gender_model, age_model, mask_model]

    stdv = 1/np.sqrt(4096)

    # model tuning
    for idx, model in enumerate(models):
        if idx > 0:
            model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=3, bias=True)
        else:
            model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2, bias=True)
        torch.nn.init.xavier_uniform_(model.classifier[6].weight)
        model.classifier[6].bias.data.uniform_(-stdv, stdv)
        model.to(device)

    # optimizer, criterion
    gender_optimizer = optim.Adam(params=models[0].parameters(), lr=args.learning_rate)
    age_optimizer = optim.Adam(params=models[1].parameters(), lr=args.learning_rate)
    mask_optimizer = optim.Adam(params=models[2].parameters(), lr=args.learning_rate)
    optimizers = [gender_optimizer, age_optimizer, mask_optimizer]

    criterion = nn.CrossEntropyLoss().to(device)
    f1_score = F1Score(num_classes=18, average="macro").to(device)

    best_models = [None, None, None]
    f1_best = 0.0

    # training
    print("Start Training...")
    for epoch in range(1, args.epochs+1):
        for model in models:
            model.train()

        for idx, data in enumerate(data_loaders['train']):
            img, classes = data[0].to(device), data[4].to(device)
            gender, age, mask = data[1].to(device), data[2].to(device), data[3].to(device)
            labels = [gender, age, mask]

            for optimizer in optimizers:
                optimizer.zero_grad()

            predictions = [None, None, None]
            losses = [None, None, None]

            for i, model in enumerate(models):
                output = model(img)
                loss = criterion(output, labels[i])
                losses[i] = loss
                pred = torch.argmax(output, dim=-1)
                predictions[i] = pred
                loss.backward()
                optimizers[i].step()

            classes_pred = predictions[2] * 6 + predictions[0] * 3 + predictions[1]
            f1 = f1_score(classes_pred, classes)

            if idx % 100 == 99:
                print(f"Epoch {epoch} Batch {idx+1} - F1 score: {f1:.4g} | age loss: {losses[0]:.4g} | gender loss: "
                      f"{losses[1]:.4g} | mask loss: {losses[2]:.4g}")

        # validation check
        val_loss = [0.0, 0.0, 0.0]
        val_f1 = 0.0
        with torch.no_grad():
            for model in models:
                model.eval()

            count = 0
            for k, data in enumerate(data_loaders['valid']):
                img, classes = data[0].to(device), data[4].to(device)
                gender, age, mask = data[1].to(device), data[2].to(device), data[3].to(device)
                labels = [gender, age, mask]

                predictions = [None, None, None]

                for i, model in enumerate(models):
                    output = model(img)
                    loss = criterion(output, labels[i])
                    val_loss[i] += loss
                    pred = torch.argmax(output, dim=-1)
                    predictions[i] = pred

                classes_pred = predictions[2] * 6 + predictions[0] * 3 + predictions[1]
                val_f1 += f1_score(classes_pred, classes)

                count += 1

            val_loss = [val/count for val in val_loss]
            val_f1 = val_f1/count

            print(f"Epoch {epoch} - Avg Models Validation loss : {sum(val_loss):.6g} | Avg F1 Score : {val_f1:.4g}")

            if f1_best < val_f1:
                best_models = models
                f1_best = val_f1
                print("Best Model Save!")

    if args.mode == "test":
        test(args, best_models, device)


def test(args, models, device):
    current_time = datetime.now(pytz.timezone('Asia/Seoul'))

    base_dir = '/opt/ml'
    test_img_dir = base_dir + '/input/data/eval/images'
    test_df_dir = base_dir + '/input/data/eval/info.csv'
    output_path = base_dir + f'/outputs/submissions/submission_{current_time.year}_{current_time.month}' \
                             f'_{current_time.day}_{current_time.hour}{current_time.minute}.csv'

    submission = pd.read_csv(test_df_dir)
    test_img_list = [test_img_dir + '/' + img for img in submission.ImageID.values]

    transform = transforms.Compose([
        Resize((512, 284)),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
    ])

    pre_transform = transforms.Compose([
        Resize((512 // 3, 284 // 3)),
        CenterCrop((64, 64))
    ])

    test_dataset = CustomDataSet(test_img_list, train=False, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, num_workers=4)

    predictions = []

    print("Start Test...")
    with torch.no_grad():
        answers = []

        for idx, img in enumerate(test_loader):
            if idx % 1000 == 999:
                print(f"Now I'm testing {idx+1}the data start...")

            img = img.to(device)

            preds = [None, None, None]

            for i, model in enumerate(models):
                output = model(img)
                pred = torch.argmax(output, dim=-1)
                preds[i] = pred

            classes = preds[2] * 6 + preds[0] * 3 + preds[1]
            classes = classes.cpu().numpy()[0]
            answers.append(classes)

        submission['ans'] = answers
        submission.to_csv(output_path, index=False)


def main(args):
    base_url = '/opt/ml'
    output_url = base_url + '/outputs/train_info.csv'
    img_path_df = pd.read_csv(base_url + '/outputs/img_path.csv')

    if "train_info.csv" not in os.listdir(base_url + "/outputs"):
        print("Make train info dataframe")
        make_info_df(img_path_df, output_url)

    train_info = pd.read_csv(output_url)

    train_data, valid_data = train_test_split(train_info, test_size=0.3, random_state=1234)

    transform = transforms.Compose([
        Resize((512, 284), Image.BILINEAR),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
    ])

    pre_transform = transforms.Compose([
        Resize((512 // 3, 284 // 3)),
        CenterCrop((64, 64))
    ])

    train_dataset = EnsembleDataSet(train_data, pre_transforms=pre_transform, transforms=transform)
    valid_dataset = EnsembleDataSet(valid_data, pre_transforms=pre_transform, transforms=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    train(args, {"train":train_loader, "valid":valid_loader})


if __name__ == '__main__':
    args = args_getter()
    non_list = ['name', 'mode', 'save']
    hypers = {k: v for k, v in vars(args).items() if k not in non_list}
    print(f"Hyper Parameter : {hypers}")
    print(f"Model Save name : {args.name}")

    print(f"Current Using GPU : {torch.cuda.get_device_name(0)}")

    main(args)
