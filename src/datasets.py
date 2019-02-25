from glob import glob
import random

import os
import torch
from torch.utils.data import Dataset
import numpy as np

from scipy.misc import imresize, imread
from sklearn.model_selection import train_test_split

from new_alignment_utils import img2tensor


class AgeGenderRaceDataset(Dataset):
    def __init__(self, image_dir, test_size=None, resize=None):
        self.resize = resize
        self.image_dir = image_dir
        self.all_images = np.array(glob(os.path.join(image_dir, '*.jpg')))
        if test_size is not None:
            inds = np.random.permutation(len(self.all_images)).astype(int)
            self.train_ind, self.val_ind = train_test_split(inds, test_size=test_size)
            self.set_mode('train')
        else:
            self.images = self.all_images
        print('Find {} images'.format(len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        while True:
            try:
                image_path = self.images[idx]
                image = imread(image_path)
                if self.resize and (image.shape[0] != self.resize):
                    image = imresize(image, (self.resize, self.resize))
                image = img2tensor(image)

                age_ = int(os.path.basename(image_path).split('_')[0])
                age = np.array([age_], dtype=np.uint8)
                age = np.clip(age, 0, 99)
                age = torch.from_numpy(age).long()
                gender = torch.from_numpy(np.array([os.path.basename(image_path).split('_')[1]], dtype=np.uint8)).long()
                race = torch.from_numpy(np.array([os.path.basename(image_path).split('_')[2]], dtype=np.uint8)).long()

                sample = {'image': image, 'age': age, 'race': race, 'gender': gender}
                return sample
            except Exception as e:
                print(image_path, e)
                idx = random.randint(0, len(self.images) - 2)



    def set_mode(self, mode):
        # mode train or val
        if mode == 'train':
            self.images = self.all_images[self.train_ind]
        elif mode == 'val':
            self.images = self.all_images[self.val_ind]
        else:
            raise ValueError

