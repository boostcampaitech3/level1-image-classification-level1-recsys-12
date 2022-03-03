import pandas as pd
import numpy as np
import os
from PIL import Image
import argparse
import wandb

import torch
import torchvision.models
import torchvision.transforms as transforms
from torchvision.transforms import *
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from tqdm import tqdm
import shutil
from torchmetrics import F1Score
import time
from dataset import *

def change_age(x):
    x = int(x)
    if x >= 58:
        return "2"
    elif 30 <= x < 58:
        return "1"
    else:
        return "0"
    
def modeltype(model):
    if model == 'resnet18':
        return ResNet([2, 2, 2, 2])
    elif model == 'resnet34':
        return ResNet([3, 4, 6, 3])
    
base_dir = '/opt/ml'
data_dir = os.path.join(base_dir, 'input','data','train')

df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

# Create a directory for classes-----------------
code_dir = os.path.join(base_dir, 'doritos_code')
classes_dir = os.path.join(code_dir, 'classes')

# images's directories
images_dir = os.path.join(data_dir, 'images')
category = list(df['path'])

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

mean=(0.548, 0.504, 0.479)
std=(0.237, 0.247, 0.246)
transform_base = transforms.Compose([
#                                      Resize((400, 200)), 
                                     CenterCrop((350 , 250)),
#                                      ColorJitter(0.1, 0.1, 0.1, 0.1),
#                                      RandomHorizontalFlip(),
#                                      RandomRotation(10),
#                                      RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                     ToTensor(),
                                     Normalize(mean=mean, std=std),
                                    ])

train_data = ImageFolder(classes_dir, transform = transform_base)


print("train_test_split : Start")
time1 = time.time()
train_set, valid_set = train_test_split(train_data, test_size = 0.2)
time2 = time.time()
print(f"train_test_split : Done, time : {time2-time1:.2f} sec")
print("train_set : {}, valid_set : {}".format(len(train_set), len(valid_set)))

lr = 0.00001
batch_size = 64

train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size = batch_size, shuffle = True)

resnet = modeltype('resnet18').to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr = lr)

# train
f1 = F1Score(num_classes = 18, average = 'macro').to(device)

# loss_ = []
n = len(train_loader)
best_val_loss = np.inf
best_f1 = 0.0
counter = 0
patience = 30
epochs = 1000

# -- wandb initialize with configuration
wandb.init(config={"batch_size": batch_size,
                "lr"        : lr,
                "epochs"    : epochs,
                "criterion_name" : 'cross_entropy'})

for epoch in tqdm(range(epochs)):
    resnet.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = resnet(inputs)
        pred = torch.max(outputs, 1)[1]
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_total += labels.size(0)
        train_correct += (pred == labels).sum().item()
    
#     loss_.append(running_loss / n)
    print("\n[{}] Train Accuracy : {:.4f}, Train Loss : {:.6f}".format(epoch+1,100*train_correct/train_total, running_loss / n))
    
    # wandb 학습 단계에서 Loss, Accuracy 로그 저장
    wandb.log({
        "Train loss": running_loss / n,
        "Train acc" : 100*train_correct/train_total})
    
    
    with torch.no_grad():
        print("Calculating validation results...")
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        f1_score = 0.0

        resnet.eval()
        for data in valid_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = resnet(images)

            pred = torch.max(outputs, 1)[1]
            loss1 = criterion(outputs, labels).item()

            val_total += labels.size(0) # 개수 누적(총 개수)
            val_correct += (pred == labels).sum().item()
            val_loss += loss1

            f1_score += f1(pred, labels)
            
        valid_loss = val_loss/len(valid_loader)
        valid_acc = 100*val_correct/val_total
        valid_f1 = f1_score/len(valid_loader)
        
        if (valid_loss < best_val_loss) or (valid_f1 > best_f1):
            print(f"New best model for val loss : {valid_loss:.6f}, val f1 : {valid_f1:.4f} saving the best model..")
            torch.save(resnet, f"{epoch+1:03}_acc_{valid_acc:.4f}%_loss_{valid_loss:.6f}_f1_{valid_f1:.4f}.pt")
            
            best_f1 = max(valid_f1, best_f1)
            best_val_loss = min(valid_loss, best_val_loss)
            counter = 0
        else:
            counter += 1

        if counter > patience:
            print("Early Stopping...")
            break
            
    print("\nval acc : {:.4f}%, val loss : {:.6f}, val F1 : {:.6f}".format(valid_acc, valid_loss, valid_f1))
    print(f"Best loss : {best_val_loss:.6f}, Best f1 : {best_f1:.4f}")
    print("Counter : {} / {}".format(counter, patience))
    
    # wandb 검증 단계에서 Loss, Accuracy 로그 저장
    wandb.log({
        "Valid loss": valid_loss,
        "Valid acc" : valid_acc,
        "Valid f1" : valid_f1
    })

    print("="*100)
    #----------------------------------------------------------------------------