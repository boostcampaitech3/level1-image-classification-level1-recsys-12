import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List, Any

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchsampler import ImbalancedDatasetSampler
from torchvision import transforms
from torchvision.transforms import *

from pandas_streaming.df import train_test_apart_stratify


IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        print(len(profiles))
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile,
                                        file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        print("outset")
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.sample(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile,
                                            file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        print("intset")
        return [Subset(self, indices) for phase, indices in self.indices.items()]


class TenAgeLabels(int, Enum):
    ZERO, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

    @classmethod
    def from_age(cls, value: str) -> int:
        try:
            value = int(value)
        except ValueError:
            raise ValueError(f"Age value should be numeric, {value}")

        if 0 < value <= 20:
            return cls.ZERO
        elif 21 <= value < 25:
            return cls.ONE
        elif 25 <= value < 30:
            return cls.TWO
        elif 30 <= value < 35:
            return cls.THREE
        elif 35 <= value < 40:
            return cls.FOUR
        elif 40 <= value < 45:
            return cls.FIVE
        elif 45 <= value < 50:
            return cls.SIX
        elif 50 <= value < 55:
            return cls.SEVEN
        elif 55 <= value < 60:
            return cls.EIGHT
        elif 60 <= value:
            return cls.NINE


class AgeBaseDataset(MaskBaseDataset):
    num_classes = 10
    origin_age_labels = []
    indices = []
    class_labels = []
    groups = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        super(AgeBaseDataset, self).__init__(data_dir, mean, std, val_ratio)

    def setup(self):
        cnt = 0
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                origin_age_label = AgeLabels.from_number(age)
                age_label = TenAgeLabels.from_age(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.origin_age_labels.append(origin_age_label)
                self.age_labels.append(age_label)
                self.class_labels.append(self.encode_multi_class(mask_label, gender_label, origin_age_label))

                self.indices.append(cnt)
                self.groups.append(id)
                cnt += 1

    def get_age_label(self, index):
        return self.origin_age_labels[index], self.age_labels[index]

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        oring_age_label, age_label = self.get_age_label(index)

        image_transform = self.transform(image)
        return image_transform, oring_age_label, age_label

    def split_dataset(self) -> List[Subset[Any]]:
        df = pd.DataFrame({"indices": self.indices, "group": self.groups, "labels":self.age_labels})
        train, valid = train_test_apart_stratify(df, group='group', stratify='labels', test_size=self.val_ratio)
        train_index = train['indices'].tolist()
        valid_index = valid['indices'].tolist()

        return [Subset(self, train_index), Subset(self, valid_index)]

    @staticmethod
    def encode_original_age(age_label) -> int:
        original_age_label = torch.zeros_like(age_label)
        for ind, age in enumerate(age_label):
            if age <= 2:
                original_age_label[ind] = 0
            elif age <= 8:
                original_age_label[ind] = 1
            else:
                original_age_label[ind] = 2
        return original_age_label

    def k_fold_split(self) -> List[List[Subset[Any]]]:
        df = pd.DataFrame({"indices": self.indices, "group": self.groups, "labels": self.age_labels})

        # k-fold 5
        train, valid1 = train_test_apart_stratify(df, group='group', stratify='labels', test_size=self.val_ratio)
        train, valid2 = train_test_apart_stratify(train, group='group', stratify='labels', test_size=0.25)
        train, valid3 = train_test_apart_stratify(train, group='group', stratify='labels', test_size=1/3)
        valid4, valid5 = train_test_apart_stratify(train, group='group', stratify='labels', test_size=1/2)

        val1_idx = valid1['indices'].tolist()
        val2_idx = valid2['indices'].tolist()
        val3_idx = valid3['indices'].tolist()
        val4_idx = valid4['indices'].tolist()
        val5_idx = valid5['indices'].tolist()

        train1_idx = val2_idx + val3_idx + val4_idx + val5_idx
        train2_idx = val1_idx + val3_idx + val4_idx + val5_idx
        train3_idx = val1_idx + val2_idx + val4_idx + val5_idx
        train4_idx = val1_idx + val2_idx + val3_idx + val5_idx
        train5_idx = val1_idx + val2_idx + val3_idx + val4_idx

        return [
            [Subset(self, train1_idx), Subset(self, val1_idx)], [Subset(self, train2_idx), Subset(self, val2_idx)],
            [Subset(self, train3_idx), Subset(self, val3_idx)], [Subset(self, train4_idx), Subset(self, val4_idx)],
            [Subset(self, train5_idx), Subset(self, val5_idx)]
        ]


class MaskOnlyBaseDataset(MaskBaseDataset):
    num_classes = 3
    indices = []
    class_labels = []
    groups = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        super(MaskOnlyBaseDataset, self).__init__(data_dir, mean, std, val_ratio)

    def setup(self):
        cnt = 0
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                origin_age_label = AgeLabels.from_number(age)
                age_label = TenAgeLabels.from_age(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
                self.class_labels.append(self.encode_multi_class(mask_label, gender_label, origin_age_label))

                self.indices.append(cnt)
                self.groups.append(id)
                cnt += 1

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)

        image_transform = self.transform(image)
        return image_transform, self.mask_labels[index]

    def split_dataset(self) -> List[Subset[Any]]:
        df = pd.DataFrame({"indices": self.indices, "group": self.groups, "labels":self.mask_labels})
        train, valid = train_test_apart_stratify(df, group='group', stratify='labels', test_size=self.val_ratio)
        train_index = train['indices'].tolist()
        valid_index = valid['indices'].tolist()

        return [Subset(self, train_index), Subset(self, valid_index)]

    def k_fold_split(self) -> List[List[Subset[Any]]]:
        df = pd.DataFrame({"indices": self.indices, "group": self.groups, "labels": self.mask_labels})

        # k-fold 5
        train, valid1 = train_test_apart_stratify(df, group='group', stratify='labels', test_size=self.val_ratio)
        train, valid2 = train_test_apart_stratify(train, group='group', stratify='labels', test_size=0.25)
        train, valid3 = train_test_apart_stratify(train, group='group', stratify='labels', test_size=1/3)
        valid4, valid5 = train_test_apart_stratify(train, group='group', stratify='labels', test_size=1/2)

        val1_idx = valid1['indices'].tolist()
        val2_idx = valid2['indices'].tolist()
        val3_idx = valid3['indices'].tolist()
        val4_idx = valid4['indices'].tolist()
        val5_idx = valid5['indices'].tolist()

        train1_idx = val2_idx + val3_idx + val4_idx + val5_idx
        train2_idx = val1_idx + val3_idx + val4_idx + val5_idx
        train3_idx = val1_idx + val2_idx + val4_idx + val5_idx
        train4_idx = val1_idx + val2_idx + val3_idx + val5_idx
        train5_idx = val1_idx + val2_idx + val3_idx + val4_idx

        return [
            [Subset(self, train1_idx), Subset(self, val1_idx)], [Subset(self, train2_idx), Subset(self, val2_idx)],
            [Subset(self, train3_idx), Subset(self, val3_idx)], [Subset(self, train4_idx), Subset(self, val4_idx)],
            [Subset(self, train5_idx), Subset(self, val5_idx)]
        ]


class GenderBaseDataset(MaskBaseDataset):
    num_classes = 2
    indices = []
    class_labels = []
    groups = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        super(GenderBaseDataset, self).__init__(data_dir, mean, std, val_ratio)

    def setup(self):
        cnt = 0
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                origin_age_label = AgeLabels.from_number(age)
                age_label = TenAgeLabels.from_age(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
                self.class_labels.append(self.encode_multi_class(mask_label, gender_label, origin_age_label))

                self.indices.append(cnt)
                self.groups.append(id)
                cnt += 1

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)

        image_transform = self.transform(image)
        return image_transform, self.gender_labels[index]

    def split_dataset(self) -> List[Subset[Any]]:
        df = pd.DataFrame({"indices": self.indices, "group": self.groups, "labels":self.gender_labels})
        train, valid = train_test_apart_stratify(df, group='group', stratify='labels', test_size=self.val_ratio)
        train_index = train['indices'].tolist()
        valid_index = valid['indices'].tolist()

        return [Subset(self, train_index), Subset(self, valid_index)]

    def k_fold_split(self) -> List[List[Subset[Any]]]:
        df = pd.DataFrame({"indices": self.indices, "group": self.groups, "labels": self.gender_labels})

        # k-fold 5
        train, valid1 = train_test_apart_stratify(df, group='group', stratify='labels', test_size=self.val_ratio)
        train, valid2 = train_test_apart_stratify(train, group='group', stratify='labels', test_size=0.25)
        train, valid3 = train_test_apart_stratify(train, group='group', stratify='labels', test_size=1/3)
        valid4, valid5 = train_test_apart_stratify(train, group='group', stratify='labels', test_size=1/2)

        val1_idx = valid1['indices'].tolist()
        val2_idx = valid2['indices'].tolist()
        val3_idx = valid3['indices'].tolist()
        val4_idx = valid4['indices'].tolist()
        val5_idx = valid5['indices'].tolist()

        train1_idx = val2_idx + val3_idx + val4_idx + val5_idx
        train2_idx = val1_idx + val3_idx + val4_idx + val5_idx
        train3_idx = val1_idx + val2_idx + val4_idx + val5_idx
        train4_idx = val1_idx + val2_idx + val3_idx + val5_idx
        train5_idx = val1_idx + val2_idx + val3_idx + val4_idx

        return [
            [Subset(self, train1_idx), Subset(self, val1_idx)], [Subset(self, train2_idx), Subset(self, val2_idx)],
            [Subset(self, train3_idx), Subset(self, val3_idx)], [Subset(self, train4_idx), Subset(self, val4_idx)],
            [Subset(self, train5_idx), Subset(self, val5_idx)]
        ]


class ClassKFoldDataset(MaskBaseDataset):
    num_classes = 18
    indexes = []
    class_labels = []
    groups = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        super(ClassKFoldDataset, self).__init__(data_dir, mean, std, val_ratio)

    def setup(self):
        cnt = 0
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                origin_age_label = AgeLabels.from_number(age)
                age_label = TenAgeLabels.from_age(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
                self.class_labels.append(self.encode_multi_class(mask_label, gender_label, origin_age_label))

                self.indexes.append(cnt)
                self.groups.append(id)
                cnt += 1
        self.class_labels = torch.tensor(self.class_labels)

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)

        image_transform = self.transform(image)
        return image_transform, self.class_labels[index]

    def split_dataset(self) -> List[Subset[Any]]:
        df = pd.DataFrame({"indices": self.indexes, "group": self.groups, "labels":self.class_labels})
        train, valid = train_test_apart_stratify(df, group='group', stratify='labels', test_size=self.val_ratio)
        train_index = train['indices'].tolist()
        valid_index = valid['indices'].tolist()

        return [Subset(self, train_index), Subset(self, valid_index)]

    def k_fold_split(self) -> List[List[Subset[Any]]]:
        df = pd.DataFrame({"indices": self.indexes, "group": self.groups, "labels": self.class_labels})

        # k-fold 5
        train, valid1 = train_test_apart_stratify(df, group='group', stratify='labels', test_size=self.val_ratio)
        train, valid2 = train_test_apart_stratify(train, group='group', stratify='labels', test_size=0.25)
        train, valid3 = train_test_apart_stratify(train, group='group', stratify='labels', test_size=1/3)
        valid4, valid5 = train_test_apart_stratify(train, group='group', stratify='labels', test_size=1/2)

        val1_idx = valid1['indices'].tolist()
        val2_idx = valid2['indices'].tolist()
        val3_idx = valid3['indices'].tolist()
        val4_idx = valid4['indices'].tolist()
        val5_idx = valid5['indices'].tolist()

        train1_idx = val2_idx + val3_idx + val4_idx + val5_idx
        train2_idx = val1_idx + val3_idx + val4_idx + val5_idx
        train3_idx = val1_idx + val2_idx + val4_idx + val5_idx
        train4_idx = val1_idx + val2_idx + val3_idx + val5_idx
        train5_idx = val1_idx + val2_idx + val3_idx + val4_idx

        return [
            [Subset(self, train1_idx), Subset(self, val1_idx)], [Subset(self, train2_idx), Subset(self, val2_idx)],
            [Subset(self, train3_idx), Subset(self, val3_idx)], [Subset(self, train4_idx), Subset(self, val4_idx)],
            [Subset(self, train5_idx), Subset(self, val5_idx)]
        ]


class Sampler(ImbalancedDatasetSampler):
    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            # print(dataset.indices)
            return dataset.dataset.class_labels[dataset.indices]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.classes
        else:
            raise NotImplementedError



class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        div = 512/resize[0]
        self.img_paths = img_paths
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            CenterCrop((400/div, 200/div)),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

