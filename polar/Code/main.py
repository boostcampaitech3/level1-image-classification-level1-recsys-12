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
from Datasets import CustomDataSet
from torch.utils.data import DataLoader
from Models import CustomModel

from sklearn.model_selection import train_test_split

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize


def args_getter():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", required=False, default=10, type=int, help='Num train epochs (default=10)')
    parser.add_argument("--batch_size", required=False, default=32, type=int, help="Num of batch size (default=32)")
    parser.add_argument("--d", required=False, default=0.5, type=float, help="dropout ratio (default=0.5)")
    parser.add_argument("--lr", required=False, default=0.01, type=float, help="Learning rate (default=0.01)")
    parser.add_argument("--name", required=False, default="ResNet18", help="Model name (if you use pre-trained, \
                                                                            write pre-trained model name)")
    parser.add_argument("save", default=str, help="Save the experiments")
    #parser.add_argument("--m", required=False, default=None, type=float, help="momentum (default=0.9)")

    args = parser.parse_args()

    return args


def make_img_path_df(path, output_dir=None, train=True):
    data_path = '/opt/ml/input/data'

    if train:
        df = pd.DataFrame(None, columns=['path', 'class'])
        data_path += '/train/train.csv'
        train_df = pd.read_csv(data_path)

        for idx, row in enumerate(train_df.iloc):
            for file in list(os.listdir(os.path.join(path, row['path']))):
                if file[0] == '.':
                    continue
                else:
                    file_name = file.split('.')[0]

                    if file_name == "normal":
                        mask = 2
                    elif file_name == "incorrect_mask":
                        mask = 1
                    else:
                        mask = 0
                gender = 0 if row['gender'] == "male" else 1
                data = {
                    'path': os.path.join(path, row['path'], file),
                    'class':mask*6 + gender*3 + min(2, row['age']//30)
                }
                df = df.append(data, ignore_index=True)

        df.to_csv(output_dir, index=False)


def record_expr(model, model_name, best_train_loss, best_train_score, avg_val_loss, avg_val_score, args):
    # | Date | model_name | best_loss | f1 score | avg val loss | avg val f1 score | Hyperparameters |
    base_url = '/opt/ml'
    model_state_save_path = base_url+'/level1-image-classification-level1-recsys-12/polar/model_state'
    markdown_path = base_url+'/level1-image-classification-level1-recsys-12/polar/README.md'
    current_time = datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S")

    os.system(f"echo '|{current_time}|{model_name}|{best_train_loss:.3g}|{best_train_score:.3g}|{avg_val_loss:.3g}|{avg_val_score:.3g}|{str(vars(args))}|' >> {markdown_path}")
    print("Save the experiment complete!")


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
    transform = transforms.Compose([
        Resize((512, 284), Image.BILINEAR),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
    ])

    train_dataset = CustomDataSet(list(X_train['path'].values), list(y_train.values), transform)
    valid_dataset = CustomDataSet(list(X_valid['path'].values), list(y_valid.values), transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model part
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # custom model code
    # model = CustomModel(num_classes=18).to(device)

    # ResNet 18 load and transfer learning
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(in_features=512, out_features=18, bias=True)
    torch.nn.init.xavier_uniform_(model.fc.weight)
    stdv = 1/np.sqrt(512)
    model.fc.bias.data.uniform_(-stdv, stdv)
    model.to(device)

    # optimizer, loss setting
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss().to(device)
    f1_score = F1Score(num_classes=18, average="macro").to(device)

    train_loss_history = []
    val_loss_history = []

    prev_loss = 1000.0
    f1_best = 0
    best_model = None
    # training
    for epoch in range(1, args.epochs+1):

        running_loss = 0.0
        for idx, (imgs, classes) in enumerate(train_loader):
            imgs, classes = imgs.to(device), classes.to(device)

            optimizer.zero_grad()

            pred = model(imgs)
            loss = criterion(pred, classes)

            train_loss_history.append(loss.data)

            loss.backward()
            optimizer.step()

            if idx % 50 == 0:
                print(f"epoch {epoch} batch {idx} - loss: {loss.data}")

            if loss.data < prev_loss:
                prev_loss = loss.data
                torch.save(model, os.path.join(save_model_path, 'custom_best_model.pt'))
                best_model = model

                outputs = torch.argmax(pred, dim=-1)
                f1_best = f1_score(outputs, classes)

                print(f"---------------------- Best Model Save! epoch {epoch} - batch {idx} : loss {loss.data:.5g} / F1 {f1_best:.4g} ----------------------")

        if args.epochs < 100:
            if epoch % 10 == 0:
                print(f"epoch {epoch} batch {idx} - loss: {loss.data}")
        else:
            if epoch % 100 == 0:
                print(f"epoch {epoch} batch {idx} - loss: {loss.data}")

    # validation/test
    best_model.eval()
    f1 = 0
    avg_score = 0
    test_loss = 0.0
    counts = 0

    print("Evaluation Start")

    with torch.no_grad():
        corrects = 0

        for k, (img, label) in enumerate(val_loader):
            img, label = img.to(device), label.to(device)

            output = best_model(img)
            loss = criterion(output, label)
            pred = torch.argmax(output, dim=-1)
            corrects += pred.eq(label.view_as(pred)).sum().item()
            test_loss += loss
            f1 += f1_score(pred, label)
            counts += 1

        acc = corrects / len(valid_dataset)
        f1 = f1 / counts
        test_loss = test_loss/counts

        print(f"Avg Accuracy : {acc :.3g} | Avg F1 Score {f1 : .3g} - Avg loss : {test_loss:.5g}")

    if args.save.lower() == "true":
        record_expr(best_model, args.name, prev_loss, f1_best, test_loss, f1, args)


if __name__ == '__main__':
    args = args_getter()
    main(args)

    print("============================ Run End ============================")
