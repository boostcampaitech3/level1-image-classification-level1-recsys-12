# Doritos's Final Code

import pandas as pd
import numpy as np
import os
from PIL import Image
import argparse

import torch
import torchvision.models
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor, Normalize, CenterCrop
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

from sklearn.model_selection import train_test_split

from tqdm import tqdm
import shutil
from torchmetrics import F1Score
import gc

gc.collect()
torch.cuda.empty_cache()

def change_age(x):
    x = int(x)
    if x >= 60:
        return "2"
    elif 30 <= x < 60:
        return "1"
    else:
        return "0"

def args_getter():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", required=False, default=100, type=int, help='Num train epochs (default=10)')
    parser.add_argument("--batch_size", required=False, default=32, type=int, help="Num of batch size (default=32)")
    parser.add_argument("--d", required=False, default=0.5, type=float, help="dropout ratio (default=0.5)")
    parser.add_argument("--lr", required=False, default=0.01, type=float, help="Learning rate (default=0.01)")
    parser.add_argument("--m", required=False, default=0.9, type=float, help="momentum (default=0.9)")

    args = parser.parse_args()

    return args


def main(args):
    base_dir = '/opt/ml'
    data_dir = os.path.join(base_dir, 'input','data','train')

    df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    
    # images's directories---------------------------
    images_dir = os.path.join(data_dir, 'images')
    category = list(df['path'])
    #------------------------------------------------
    
    # Create a directory for classes-----------------
    code_dir = os.path.join(base_dir, 'doritos_code')
    classes_dir = os.path.join(code_dir, 'classes')
    
    if not os.path.exists(classes_dir):
        os.mkdir(classes_dir)
        print("Create a directory for classes : Done!")
        

       # Create a directories for each class--------------------------------------
        num = 0
        num_dict = {}
        for m in ['wear', 'incorrect', 'notwear']:
            for gender in ['male', 'female']:    
                for age in ['0','1','2']:
                    title = "{}_{}_{}_{}".format(str(num).zfill(2), m, gender, age)
                    os.mkdir(os.path.join(classes_dir, title))
                    num_dict[title[3:]] = str(num).zfill(2)
                    num += 1

        print("Create a directories for each class : Done!")

        # Classifying images into each directory for each class.------------------
        cnt = 1

        for cate in tqdm(category):
            _,gen,_,age = cate.split("_")
            age = change_age(age)
            pictures = os.listdir(os.path.join(images_dir, cate))

            for pic in pictures:
                if pic.startswith('incorrect'):
                    folder = "{}_{}_{}".format('incorrect', gen, age)

                elif pic.startswith('mask'):
                    folder = "{}_{}_{}".format('wear', gen, age)

                elif pic.startswith('normal'):
                    folder = "{}_{}_{}".format('notwear', gen, age)

                else:
                    continue

                folder = f"{num_dict[folder]}_" + folder        
                file = os.path.join(os.path.join(images_dir, cate), pic)
                root = os.path.join(classes_dir, folder, str(cnt)+'.jpg')

                shutil.copyfile(file, root)
                cnt += 1
        print("Classifying images into each directory for each class. : Done")
    else:
        print("'classes' directory already exists.")
        #--------------------------------------------------------------------------
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device :", device)

    transform_base = transforms.Compose([Resize((512 // 3, 384 // 3)), CenterCrop((100, 100)), ToTensor()])
    train_data = ImageFolder(classes_dir, transform = transform_base)
    
    print("train_test_split : Start")
    train_set, valid_set = train_test_split(train_data, test_size = 0.2, random_state=960217)
    print("train_test_split : Done")
    print("train_set : {}, valid_set : {}".format(len(train_set), len(valid_set)))
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = args.batch_size, shuffle = True)
    
    
    # ResNet model----------------------------------------------------------
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(in_features=512, out_features=18, bias=True)

    torch.nn.init.xavier_uniform_(model.fc.weight)
    stdv = 1/np.sqrt(512)
    model.fc.bias.data.uniform_(-stdv, stdv)

    model.to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss().to(device)
    
    
    # train-----------------------------------------------------------------
    loss_ = []
    epoch = 100

    n = len(train_loader)

    model.train()
    for epoch in tqdm(range(1, args.epochs+1)):
        running_loss = 0.0
        prev_loss = 1000.0
        best_model = None

        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if i % 50 == 0:
                print(f"epoch {epoch} batch {i} - loss: {loss.data}")

            if loss.data < prev_loss:
                prev_loss = loss.data
                torch.save(model, 'custom_best_model.pt')
                best_model = model
                print(f"---------------------- Best Model Save! epoch {epoch} - batch {i} : loss {loss.data} ----------------------")

        loss_.append(running_loss / n)

        if epoch % 5 == 0:
            print("[{}] loss : {:.3f}".format(epoch, running_loss / n))
    #--------------------------------------------------------------------------
    
    # eval---------------------------------------------------------------------
    f1 = F1Score(num_classes = 18, average = 'macro').to(device)

    correct = 0
    total = 0
    f1_score = 0.0
    val_loss = 0.0

    with torch.no_grad():
        model.eval()       
        for data in tqdm(valid_loader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = best_model(images)

            pred = torch.max(outputs, 1)[1]

            total += labels.size(0) # 개수 누적(총 개수)
            correct += (pred == labels).sum().item()

            f1_score += f1(pred, labels)

    print("\nValid Accuracy : {:.4f}%, loss : {:.4f}, F1 Score : {:.4f}".format(100*correct/total, val_loss/total, f1_score/len(valid_loader)))
    #----------------------------------------------------------------------------
    
if __name__ == '__main__':
    args = args_getter()
    main(args)
    print("============================ Run End ============================")