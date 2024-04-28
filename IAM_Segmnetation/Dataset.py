from sklearn import model_selection
import os


class Dataset:
    def __init__(self, images_path, masks_path):
        # self.images_path = images_path
        # self.masks_path = masks_path

        self.images_path = "D:/Databases/06/forms_obciete"
        self.masks_path = "D:/Databases/06/forms_masks"

        self.images_path1 = "D:/Databases/06/forms_obciete1"
        self.masks_path1 = "D:/Databases/06/forms_masks1"

        self.images_path2 = "D:/Databases/06/forms_obciete2"
        self.masks_path2 = "D:/Databases/06/forms_masks2"

        self.images_path3 = "D:/Databases/06/forms_obciete3"
        self.masks_path3 = "D:/Databases/06/forms_masks3"

    def read_dataset(self):
        image_paths = []
        mask_paths = []

        # first folder
        images = os.listdir(self.images_path)
        masks = os.listdir(self.masks_path)

        for image in images:
            temp_img = os.path.join(self.images_path, image)
            image_paths.append(temp_img)

        for mask in masks:
            temp_mask = os.path.join(self.masks_path, mask)
            mask_paths.append(temp_mask)

        print(len(image_paths), len(mask_paths))
        # second folder
        images = os.listdir(self.images_path1)
        masks = os.listdir(self.masks_path1)

        for image in images:
            temp_img = os.path.join(self.images_path1, image)
            image_paths.append(temp_img)

        for mask in masks:
            temp_mask = os.path.join(self.masks_path1, mask)
            mask_paths.append(temp_mask)

        print(len(image_paths), len(mask_paths))
        # third folder
        images = os.listdir(self.images_path2)
        masks = os.listdir(self.masks_path2)

        for image in images:
            temp_img = os.path.join(self.images_path2, image)
            image_paths.append(temp_img)

        for mask in masks:
            temp_mask = os.path.join(self.masks_path2, mask)
            mask_paths.append(temp_mask)

        print(len(image_paths), len(mask_paths))
        # fourth folder
        images = os.listdir(self.images_path3)
        masks = os.listdir(self.masks_path3)

        for image in images:
            temp_img = os.path.join(self.images_path3, image)
            image_paths.append(temp_img)

        for mask in masks:
            temp_mask = os.path.join(self.masks_path3, mask)
            mask_paths.append(temp_mask)

        print(len(image_paths), len(mask_paths))
        train_images, test_images, train_masks, test_masks = model_selection.train_test_split(image_paths, mask_paths,
                                                                                              test_size=0.2,
                                                                                              random_state=42)
        return train_images, test_images, train_masks, test_masks
