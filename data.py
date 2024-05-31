from torch.utils.data import Dataset
import numpy as np
import cv2
from albumentations import *
import torch
import pandas as pd
import random
import numpy as np


class CAPTCHALoader(Dataset):

    def __init__(self, data_path, mode, train_size=0.8, labels_path=None):
        super().__init__()
        assert mode in ['train', 'val', 'test']

        self.data = np.load(data_path)
        if mode in ['train', 'val']:
            assert labels_path is not None
            self.labels = np.load(labels_path)
            self.n_classes = len(np.unique(self.labels))
            # разделяем валидацию и тест с фиксированным сидом
            np.random.seed(100500)
            rand_idx = np.random.permutation(len(self.labels))
            split_idx = round(len(self.labels) * train_size)
            rand_idx = rand_idx[:split_idx] if mode == 'train' else rand_idx[split_idx:]
            self.data, self.labels = self.data[rand_idx], self.labels[rand_idx]

        self.mode = mode
        if mode == 'train':
            # инициализируем аугментации
            img_size = self.data.shape[1]
            min_crop_size = round(0.7 * img_size)
            self.augmentation = Compose([
                RandomSizedCrop(min_max_height=(min_crop_size, img_size), size=(img_size, img_size), p=0.5),
                RGBShift(r_shift_limit=(-60, 60), g_shift_limit=(-60, 60), b_shift_limit=(-60, 60), p=0.5),
                Rotate(border_mode=cv2.BORDER_CONSTANT, limit=30, interpolation=4, p=.3, value=0),
                ChannelShuffle(p=0.25),
                ChannelDropout(p=0.25),
                OneOf([
                    GaussianBlur(p=1),
                    MotionBlur(p=1),
                    MedianBlur(p=1),
                ], p=0.25)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = np.copy(self.data[index])
        if self.mode == 'train':
            img = self.augmentation(image=img)['image']
        img = torch.from_numpy(img).float()
        if self.mode == 'test':
            return img
        return img, self.labels[index]


if __name__ == '__main__':
    dataset = CAPTCHALoader(data_path='./images_sub.npy', mode='test')
    dataset = CAPTCHALoader(data_path='./images.npy', mode='val', labels_path='./labels.npy')
    from matplotlib import pyplot as plt
    for index in range(len(dataset)):
        img, label = dataset[index]
        plt.imshow(img.numpy().astype(np.uint8))
        plt.title(str(label))
        plt.show()

        if index == 10:
            break
