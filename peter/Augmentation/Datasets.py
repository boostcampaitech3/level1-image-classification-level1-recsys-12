import os
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
import cv2

# 고려해볼만한 것?
# Shearing, Contrasting, Horizontal Flip, Zoom, Bfur, Resize, Grey Scale

# albumentation >>  --epochs 10 --batch_size 32 --lr 0.001 tensor(0.7695, device='cuda:0')

# 1. 나이 
# 흰머리, 주름으로 구별되므로 색상변경이나 흐릿해지는건 안써야할듯
# 나이의 핵심은 50대 후반과 60대초를 구별하는것같다

# 2. 성별
# 마찬가지로 좌우반전만 고려해봐야할거같다

# 3. 마스크 착용 여부
# 색상 변경 정도는 괜찮을거같다

class AugDataSet(Dataset):
    def __init__(self,image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform:
            transformed = self.transform(image=image)
            transformed_image = transformed["image"]
        return transformed_image, label

    def __len__(self):
        return len(self.image_paths)