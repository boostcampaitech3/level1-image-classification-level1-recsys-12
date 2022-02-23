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
from torchvision.transforms import Resize, ToTensor, Normalize


def train(args, data_loaders):
    train_loader = data_loaders[0]
    val_loader = data_loaders[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Use Device : {device}')

    # model load
    model_age = torchvision.models.resnet18(pretrained=False)
    model_gender = torchvision.models.resnet18(pretrained=False)
    model_mask = torchvision.models.resnet18(pretrained=False)
    models = [model_age, model_gender, model_age]

    # per-trained model settings
    stdv = 1 / np.sqrt(512)

    for model in models:
        model.fc = torch.nn.Linear(in_features=512, out_features=3, bias=True)
        torch.nn.init.xavier_uniform_(model.fc.weight)
        model.fc.bias.data.uniform_(-stdv, stdv)
        model.to(device)

    # optimizer, loss setting
    age_optimizer = optim.Adam(params=model_age.parameters(), lr=args.learning_rate)
    gender_optimizer = optim.Adam(params=model_gender.parameters(), lr=args.learning_rate)
    mask_optimizer = optim.Adam(params=model_mask.parameters(), lr=args.learning_rate)
    optimizers = [age_optimizer, gender_optimizer, mask_optimizer]

    criterion = nn.CrossEntropyLoss().to(device)
    f1_score = F1Score(num_classes=18).to(device)

    best_models = [None, None, None]
    prev_val_losses = [1000.0, 1000.0, 1000.0]
    f1_bests = 1000.0

    print("Train Start!")

    # training
    for epoch in range(1, args.epochs + 1):
        for model in models:
            model.train()

        for idx, data in enumerate(train_loader):
            img, gender, age, mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
            classes = data[4].to(device)

            labels = [age, gender, mask]
            preds = [None, None, None]

            for optimizer in optimizers:
                optimizer.zero_grad()

            for idx, model in enumerate(models):
                output = model(img)
                pred = torch.argmax(output, dim=-1)
                preds[idx] = pred
                loss = criterion(output, labels[idx])
                loss.backward()
                optimizers[idx].step()

            pred_classes = preds[2] * 6 + preds[1] * 3 + preds[0]
            f1 = f1_score(pred_classes, classes)

            if idx % 100 == 99:
                print(f"Epoch {epoch} Batch {idx+1} - F1 score : {f1:.4g} | age loss : {age_loss:.4g} | gender loss : {gender_loss:.4g} | mask loss : {mask_loss:.4g}")

        val_loss = [0.0, 0.0, 0.0]
        avg_val_loss = 0.0
        val_f1 = 0.0

        # validation check
        with torch.no_grad():
            for k, data in enumerate(val_loader):
                img, gender, age, mask = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
                classes = data[4].to(device)

                age_output = model_age(img)
                age_pred = torch.argmax(age_output, dim=-1)
                gender_output = model_gender(img)
                gender_pred = torch.argmax(gender_output, dim=-1)
                mask_output = model_mask(img)
                mask_pred = torch.argmax(mask_output, dim=-1)

                age_loss = criterion(age_output, age)
                gender_loss = criterion(gender_output, gender)
                mask_loss = criterion(mask_output, mask)

                val_loss[0] = val_loss[0] + age_loss
                val_loss[1] = val_loss[1] + gender_loss
                val_loss[2] = val_loss[2] + mask_loss

                pred_classes = mask_pred * 6 + gender_pred * 3 + age_pred
                val_f1 += f1_score(pred_classes, classes)

            val_loss = [val/len(val_loader.dataset) for val in val_loss]
            val_f1 = val_f1/len(val_loader.dataset)

            print(f"Epoch {epoch} - Avg Models Validation loss : {val_loss.sum():.6g} | Avg F1 Score : {val_f1:.4g}")

            # age model
            if prev_val_losses[0] > val_loss[0]:
                best_models[0] = model_age
                prev_val_losses[0] = val_loss[0]

            # gender model
            if prev_val_losses[1] > val_loss[1]:
                best_models[1] = model_gender
                prev_val_losses[1] = val_loss[0]

            # mask model
            if prev_val_losses[2] > val_loss[2]:
                best_models[2] = model_age
                prev_val_losses[2] = val_loss[2]

    if args.mode == "test":
        test(best_models, device)


def test(models, device):
    current_time = datetime.now(pytz.timezone('Asia/Seoul'))

    base_dir = '/opt/ml'
    test_img_dir = base_dir + '/input/data/eval/images'
    test_df_dir = base_dir + '/input/data/eval/info.csv'
    output_path = base_dir + f'/outputs/submissions/submission_\{current_time.month}{current_time.day}{current_time.hour}{current_time.minute}.csv'

    submission = pd.read_csv(test_df_dir)
    test_img_list = [test_img_dir + '/' + img for img in submission.ImageID.values]

    transform = transforms.Compose([
        Resize((512, 284), Image.BILINEAR),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
    ])

    test_dataset = CustomDataSet(test_img_list, train=False, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False)

    preds = []

    print("Start Test")
    with torch.no_grad():
        for img in test_loader:
            img = img.to(device)

            age_output = models[0](img)
            age_pred = torch.argmax(age_output, dim=-1)
            gender_output = models[1](img)
            gender_pred = torch.argmax(gender_output, dim=-1)
            mask_output = models[2](img)
            mask_pred = torch.argmax(mask_output, dim=-1)

            pred = mask_pred * 6 + gender_pred * 3 + age_pred
            pred = pred.cpu().numpy()[0]
            preds.append(pred)

    submission['ans'] = preds

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

    train_dataset = EnsembleDataSet(train_data, transforms=transform)
    valid_dataset = EnsembleDataSet(valid_data, transforms=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

    train(args, [train_loader, valid_loader])


if __name__ == '__main__':
    args = args_getter()
    non_list = ['name', 'mode', 'save']
    hypers = {k: v for k, v in vars(args).items() if k not in non_list}
    print(f"Hyper Parameter : {hypers}")
    print(f"Model Save name : {args.name}")

    main(args)
