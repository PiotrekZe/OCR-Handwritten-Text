import os
from sklearn import preprocessing
from sklearn import model_selection


class Dataset:
    def __init__(self, path):
        self.path = path

    def read_dataset(self):
        image_paths = []
        targets = []

        files = os.listdir(self.path)
        for file in files:
            file_path = os.path.join(self.path, file)
            image_paths.append(file_path)
            targets.append(file[:-4])

        targets, target_classes = self.__encode_labels(targets)
        train_paths, test_paths, train_targets, test_targets = model_selection.train_test_split(image_paths, targets,
                                                                                                test_size=0.2,
                                                                                                random_state=42)

        return train_paths, test_paths, train_targets, test_targets, target_classes

    def __encode_labels(self, targets):
        labels = [[y for y in x] for x in targets]
        labels_flat = [i for label_list in labels for i in label_list]
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(labels_flat)
        labels_encoded = [label_encoder.transform(x) for x in labels]

        # Use for datasets with different size of targets - they require padding, and the original size of target for
        # CTC Loss in case of CAPTCHA dataset all targets are the same size
        '''
        labels_lengths = [len(x) for x in labels_encoded]
        padding_symbol = -1
        max_length = max(len(seq) for seq in labels_encoded)
        labels_encoded = [list(label) + [padding_symbol] * (max_length - len(label)) for label in labels_encoded]
        labels_encoded = np.array(labels_encoded) + 1
        '''
        return labels_encoded, label_encoder.classes_
