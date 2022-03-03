import os
import pandas as pd
from dataset import *
from torch.utils.data import DataLoader, Dataset
import torchvision.models
import torch
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device :", device)

base_dir = '/opt/ml'
eval_dir = os.path.join(base_dir, 'input', 'data','eval')

info_df = pd.read_csv(os.path.join(eval_dir, 'info.csv'))
info_id = list(info_df['ImageID'])

info_img = os.path.join(eval_dir, 'images')
img_paths = [os.path.join(info_img, img_id) for img_id in info_id]

dataset = TestDataset(img_paths, ((400, 200)))


test_loader = DataLoader(dataset, batch_size = 64, shuffle = False)

# best_model = torchvision.models.resnet18(pretrained=False)
# best_model.fc = torch.nn.Linear(in_features=512, out_features=18, bias=True)

best_model = torch.load('062_acc_90.0529%_loss_0.393161_f1_0.8278.pt')
# best_model.load_state_dict(torch.load('021_accuracy_97.83%.ckpt'))

best_model.to(device)

preds = []
with torch.no_grad():
    for idx, images in tqdm(enumerate(test_loader)):
        images = images.to(device)
        pred = best_model(images)
        pred = pred.argmax(dim=-1)
        preds.extend(pred.cpu().numpy())
        
info_df['ans'] = preds
info_df.to_csv(f'output.csv', index=False)
print(f'Inference Done!')