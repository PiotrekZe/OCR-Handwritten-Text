from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, images, masks, size):
        self.images = images
        self.masks = masks
        self.size = size
        self.transform = transforms.Compose(
            [transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image_shape_x, image_shape_y = np.array(image).shape[0], np.array(image).shape[1]
        image = image.resize((self.size[1], self.size[0]))
        image = self.transform(image)

        mask = Image.open(self.masks[index])
        mask_shape_x, mask_shape_y = np.array(mask).shape[0], np.array(mask).shape[1]
        mask = mask.resize((self.size[1], self.size[0]))
        mask = self.transform(mask)
        return image, mask, image_shape_x, image_shape_y, mask_shape_x, mask_shape_y
