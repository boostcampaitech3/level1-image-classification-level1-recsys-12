from PIL import Image
from torchvision.transforms import transforms, Resize, ToTensor, Normalize, RandomHorizontalFlip, CenterCrop


class BaseAugmentation:
    def __init__(self, resize, mean, std, **kwargs):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class FlipAugmentation(BaseAugmentation):
    def __init__(self, resize, mean, std, **kwargs):
        super().__init__(resize, mean, std, **kwargs)
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            RandomHorizontalFlip(p=kwargs['p']),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])


class CenterCropFlip(BaseAugmentation):
    def __init__(self, resize, mean, std, **kwargs):
        super().__init__(resize, mean, std, **kwargs)
        div = 512//resize[0]
        print(f"crop {255//div}")
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            CenterCrop((256//div, 256//div)),
            RandomHorizontalFlip(p=kwargs['p']),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])


class Crop(BaseAugmentation):
    def __init__(self, resize, mean, std, **kwargs):
        super(Crop, self).__init__(resize, mean, std, **kwargs)
        div = 512/resize[0]
        print(f'Crop augmentation : {div}')
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            CenterCrop((400/div, 200/div)),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])


class CropFlip(BaseAugmentation):
    def __init__(self, resize, mean, std, **kwargs):
        super(CropFlip, self).__init__(resize, mean, std, **kwargs)
        div = 512/resize[0]
        print(f'Crop augmentation : {div}')
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            CenterCrop((400/div, 200/div)),
            RandomHorizontalFlip(p=kwargs['p']),
            ToTensor(),
            Normalize(mean=mean, std=std)
        ])
