from sklearn import model_selection
import os


class Dataset:
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path

    def read_dataset(self):
        image_paths = []
        mask_paths = []

        images = os.listdir(self.images_path)
        masks = os.listdir(self.masks_path)

        for image in images:
            temp_img = os.path.join(self.images_path, image)
            image_paths.append(temp_img)

        for mask in masks:
            temp_mask = os.path.join(self.masks_path, mask)
            mask_paths.append(temp_mask)
        train_images, test_images, train_masks, test_masks = model_selection.train_test_split(image_paths, mask_paths,
                                                                                              test_size=0.2,
                                                                                              random_state=42)
        return train_images, test_images, train_masks, test_masks
