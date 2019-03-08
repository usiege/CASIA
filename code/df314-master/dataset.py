import os
from random import sample

import keras
from keras.utils import to_categorical
from sklearn.model_selection import RepeatedKFold

from utils import *


class DF314(keras.utils.Sequence):
    def __init__(self, folder_name, mode='train', batch_size=10, num_classes=2, fold=False):
        assert mode in ['train', 'predict', 'test']
        self.mode = mode
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.folder_name = folder_name
        self.file_names = next(os.walk('{}/binary/merged'.format(folder_name)))[2]
        self.fold = fold

        if fold:
            kf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
            self.folds = list(kf.split(self.file_names))

            self.stage = 0
            self.current = 0
            self.count = 0
            self.folds_len = len(self.folds)
            self.current_fold = self.folds[self.current][self.stage]
            length = len(self.current_fold)
            self.current_fold_size = int(np.ceil(length) / float(self.batch_size))

    def __len__(self):
        if self.fold:
            length = self.current_fold_size
        else:
            length = len(self.file_names)
            length = int(np.ceil(length) / float(self.batch_size))
        return length

    def reset_fold(self):
        self.count += 1

        if self.count == self.current_fold_size:
            if self.stage == 1:
                self.current += 1
                if self.current == self.folds_len:
                    self.current = 0
                self.stage = 0
            else:
                self.stage = 1
            self.count = 0
            self.current_fold = self.folds[self.current][self.stage]
            length = len(self.current_fold)
            self.current_fold_size = int(np.ceil(length) / float(self.batch_size))

    def __getitem__(self, idx):
        images = []
        masks = []

        ridx = [i + self.batch_size * idx for i in range(self.batch_size)]
        if self.fold:
            ridx = [self.current_fold[i] for i in ridx]
        image_names = [self.file_names[i] for i in ridx]

        for image_name in image_names:
            image, mask = load_data(image_name, self.folder_name, mode=self.mode)
            if self.mode == 'train':
                mask = to_categorical(mask, self.num_classes)

            images.append(image)
            masks.append(mask)

        if self.fold:
            self.reset_fold()

        if self.mode == 'test':
            return np.array(images), np.array(image_names)
        elif self.mode == 'predict':
            return np.array(images), np.array(masks), np.array(image_names)
        else:
            return np.array(images), np.array(masks)

    def get_validation_step(self):
        length = len(self.folds[0][1])
        return int(np.ceil(length) / float(self.batch_size))


def load_data(filename, path, mode='train'):
    image = np.load('{}/binary/merged/{}'.format(path, filename))
    if mode == 'test':
        mask = image[:, :, 1]
    elif mode == 'predict':
        mask = image[:, :, 1]
        mask = mask.reshape((64, 4000, 1)).astype(np.uint8)
    else:
        mask = image[:, :, 3]
        mask = mask.reshape((64, 4000, 1)).astype(np.uint8)

    image = image[:, :, 1]
    image = image / 255.0

    image = image.reshape((64, 4000, 1))

    return image, mask
