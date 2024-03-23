import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, paths, targets, size, num_channels, target_lengths=None):
        self.paths = paths
        self.targets = targets
        self.size = size
        if self.size is not None:
            self.width = size[0]
            self.height = size[1]
        self.num_channels = num_channels
        self.target_lengths = target_lengths

        if self.num_channels == 3:
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        elif self.num_channels == 1:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.num_channels == 3:
            image = Image.open(self.paths[index]).convert("RGB")
        elif self.num_channels == 1:
            image = Image.open(self.paths[index]).convert("L")
        target = self.targets[index]

        # for images with different sizes apply black padding to image that helps image to not be stretched images
        # with long words like "alternative" will remain with their natural proportions, but words like "to" will
        # be unnatural stretched
        '''
        if self.size is not None:
            if np.array(image).shape[1] > self.size[1]:
                image = image.resize((self.size[1], self.size[0]), resample=Image.BILINEAR)
            else:
                image = image.resize((int(image.width * self.size[0] / image.height), self.size[0]), Image.LANCZOS)
                padding = (self.size[1] - image.width) // 2
                image = ImageOps.expand(image, (padding, 0, padding, 0), fill='black')
                image = image.resize((self.size[1], self.size[0]), resample=Image.BILINEAR)
        '''

        if self.size is not None:
            image = image.resize((self.size[1], self.size[0]), resample=Image.BILINEAR)

        if self.num_channels == 3:
            image = np.array(image)
            # image = np.transpose(image, (2,0,1)).astype(np.float32)
            print(image.shape)
            image = self.transform(image)
        elif self.num_channels == 1:
            image = np.array(image)
            image = np.expand_dims(image, axis=0)
            image = self.transform(image)  # is the transform needed?
            image = image.permute(1, 2, 0)

        if self.target_lengths is None:
            return image, torch.tensor(target, dtype=torch.long)
        else:
            target_length = self.target_lengths[index]
            return image, torch.tensor(target, dtype=torch.long), target_length
