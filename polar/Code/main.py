from utils import *

import os

import torchvision.models
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import F1Score

import numpy as np
import pandas as pd
from Datasets import CustomDataSet, CustomDataSet2
from torch.utils.data import DataLoader
from Models import CustomModel

from sklearn.model_selection import train_test_split

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize


def main(args):
    # directories
    base_dir = '/opt/ml'
    train_dir = base_dir+'/input/data/train/images'
    output_dir = base_dir+'/outputs/img_path.csv'
    save_model_path = base_dir+'/outputs/model_save'

    # basic directory and dataframe check
    if 'outputs' not in os.listdir('/opt/ml'):
        print("output directory is not find...")
        os.mkdir(base_dir+'/outputs')
        print("Make directory success")

    if 'img_path.csv' not in os.listdir(base_dir+'/outputs'):
        print("path dataframe is not find...")
        make_img_path_df(train_dir, output_dir)
        print("Make dataframe success")

    # train image path load
    train_path_df = pd.read_csv(output_dir)
    train_img_paths = list(train_path_df['path'].values)

    # train valid split
    y = train_path_df['class']
    X = train_path_df.copy().drop(['class'], axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=1234, stratify=y)

    # DataSet and Loader Setting
    pre_transform = transforms.Compose([
        Resize((512//3, 284//3)),
        transforms.CenterCrop((64, 64)),
    ])

    transform = transforms.Compose([
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
    ])

    train_dataset = CustomDataSet2(list(X_train['path'].values), list(y_train.values), pre_transform=pre_transform, transform=transform)
    val_dataset = CustomDataSet2(list(X_valid['path'].values), list(y_valid.values), pre_transform=pre_transform, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model part
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Now use device : {device}")

    # custom model code
    # model = CustomModel(num_classes=18).to(device)

    # ResNet 18 load and transfer learning
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(in_features=512, out_features=18, bias=True)
    torch.nn.init.xavier_uniform_(model.fc.weight)
    stdv = 1/np.sqrt(512)
    model.fc.bias.data.uniform_(-stdv, stdv)
    model.to(device)

    # optimizer, loss setting
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)
    f1_score = F1Score(num_classes=18, average="macro").to(device)

    train_loss_history = []
    val_loss_history = []

    prev_loss = 1000.0
    prev_val_loss = 1000.0
    best_model = None
    f1_best = 0.0
    # training
    print("Start Training...")
    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        for idx, (imgs, classes) in enumerate(train_loader):
            imgs, classes = imgs.to(device), classes.to(device)

            optimizer.zero_grad()

            output = model(imgs)
            loss = criterion(output, classes)
            pred = torch.argmax(output, dim=-1)
            f1 = f1_score(pred, classes)

            loss.backward()
            optimizer.step()

            if prev_loss > loss:
                prev_loss = loss

            if idx % 100 == 99:
                print(f"Epoch {epoch} Batch {idx+1} - Train loss : {loss:.6g} | F1 score : {f1:.4g}")

        model.eval()
        val_loss = 0.0
        val_f1 = 0.0

        # validation check
        with torch.no_grad():
            count = 0
            for k, (img, label) in enumerate(val_loader):
                img, label = img.to(device), label.to(device)

                output = model(img)
                loss = criterion(output, label)
                pred = torch.argmax(output, dim=-1)

                val_loss += loss
                val_f1 += f1_score(pred, label)
                count += 1

            val_loss = val_loss/count
            val_f1 = val_f1/count
            print(f"Epoch {epoch} - Avg Validation loss : {val_loss:.6g} | Avg F1 Score : {val_f1:.4g}")

            # check best validation model
            if prev_val_loss > val_loss:
                prev_val_loss = val_loss
                best_model = model
                f1_best = val_f1

    if args.save.lower() == "true":
        record_expr(best_model, args.name, prev_loss, val_loss, val_f1, f1_best, args)

    if args.mode == "test":
        test(best_model, device)


def test(best_model, device):
    current_time = datetime.now(pytz.timezone('Asia/Seoul'))

    base_dir = '/opt/ml'
    test_img_dir = base_dir + '/input/data/eval/images'
    test_df_dir = base_dir + '/input/data/eval/info.csv'
    output_path = base_dir + f'/outputs/submissions/submission_{current_time.year}_{current_time.month}' \
                             f'_{current_time.day}_{current_time.hour}{current_time.minute}.csv'

    submission = pd.read_csv(test_df_dir)
    test_img_list = [test_img_dir+'/'+img for img in submission.ImageID.values]

    pre_transform = transforms.Compose([
        Resize((512 // 3, 284 // 3)),
        transforms.CenterCrop((64, 64)),
    ])

    transform = transforms.Compose([
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
    ])

    test_dataset = CustomDataSet2(test_img_list, train=False, pre_transform=pre_transform, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False)

    preds = []

    print("Start Test")
    with torch.no_grad():
        for img in test_loader:
            img = img.to(device)

            output = best_model(img)
            pred = torch.argmax(output, dim=-1)
            # print(pred)
            pred = pred.cpu().numpy()[0]
            preds.append(pred)

    submission['ans'] = preds

    submission.to_csv(output_path, index=False)


if __name__ == '__main__':
    args = args_getter()
    non_list = ['name', 'mode', 'save']
    hypers = {k: v for k, v in vars(args).items() if k not in non_list}
    print(f"Hyper Parameter : {hypers}")
    print(f"Model Save name : {args.name}")
    main(args)

    print("============================ Run End ============================")
