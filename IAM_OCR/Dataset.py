import os
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from PIL import Image
import pandas as pd


class ImageOpenError(Exception):
    pass


class Dataset:
    def __init__(self, images_path, file_path):
        self.images_path = images_path
        self.file_path = file_path

    def read_dataset(self):
        image_paths = []
        wrong_image = []  # to store broken files

        for root, dirs, files in os.walk(self.images_path):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_path = os.path.join(root, file)
                    try:
                        self.__check_image(image_path)
                        image_paths.append(image_path)
                    except Exception:
                        wrong_image.append(file[:-4])

        largest_column_count = 0
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                column_count = len(line.split(" "))
                largest_column_count = column_count if largest_column_count < column_count else largest_column_count

        column_names = [i for i in range(0, largest_column_count)]
        df = pd.read_csv(self.file_path, header=None, sep=" ", names=column_names, low_memory=False, quoting=3,
                         dtype=str, keep_default_na=False)
        df = df[~df[0].isin(wrong_image)]

        for i in range(df.shape[0]):
            temp_text = ''
            for j in range(8, 11):
                if df.iloc[i][j] == '':
                    break
                temp_text += df.iloc[i][j]
                temp_text += '|'
            temp_text = temp_text[:-1]
            df.loc[i, 8] = temp_text

        targets = np.array(df[8])
        Y = targets
        print(len(targets), len(image_paths), wrong_image)

        targets, targets_lengths, target_classes = self.__encode_labels(targets)

        # to test faster
        # num = 10000
        # targets = targets[:num]
        # targets_lengths = targets_lengths[:num]
        # image_paths = image_paths[:num]
        # Y = Y[:num]

        (train_paths, test_paths, train_targets, test_targets, train_targets_lengths, test_targets_lengths,
         train_original_targets, test_original_targets) = model_selection.train_test_split(image_paths, targets,
                                                                                           targets_lengths, Y,
                                                                                           test_size=0.2,
                                                                                           random_state=42)

        return (train_paths, test_paths, train_targets, test_targets, train_targets_lengths, test_targets_lengths,
                train_original_targets, test_original_targets, target_classes)

    def __check_image(self, path):
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            raise ImageOpenError()

    def __encode_labels(self, targets):
        labels = [[y for y in x] for x in targets]
        labels_flat = [i for label_list in labels for i in label_list]
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(labels_flat)
        labels_encoded = [label_encoder.transform(x) for x in labels]

        labels_lengths = [len(x) for x in labels_encoded]
        padding_symbol = -1
        max_length = max(len(seq) for seq in labels_encoded)
        labels_encoded = [list(label) + [padding_symbol] * (max_length - len(label)) for label in labels_encoded]
        labels_encoded = np.array(labels_encoded) + 1

        # Use for datasets with different size of targets - they require padding, and the original size of target for
        # CTC Loss in case of CAPTCHA dataset all targets are the same size
        '''
        labels_lengths = [len(x) for x in labels_encoded]
        padding_symbol = -1
        max_length = max(len(seq) for seq in labels_encoded)
        labels_encoded = [list(label) + [padding_symbol] * (max_length - len(label)) for label in labels_encoded]
        labels_encoded = np.array(labels_encoded) + 1
        '''
        return labels_encoded, labels_lengths, label_encoder.classes_
