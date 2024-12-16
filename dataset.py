import torch
import torchvision
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import random
import pickle as pkl
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class HDF5Dataset(Dataset):
    def __init__(self, h5_file, indices, preload_fraction=0.5, transform=None):
        self.h5_file = h5_file
        self.indices = indices
        self.transform = transform
        self.h5data = h5py.File(self.h5_file, 'r')

        # Determine the number of preloaded items
        self.num_preload = int(len(indices) * preload_fraction)
        self.preload_data()

    def preload_data(self):
        self.preloaded_images = []
        self.preloaded_masks = []

        for idx in range(self.num_preload):
            true_idx = self.indices[idx]
            image = self.h5data['images'][true_idx]
            gt = self.h5data['ground_truth_masks'][true_idx]

            image = Image.fromarray(image)  # Assuming the image data is in [0, 1]
            gt = Image.fromarray(gt)  # Assuming the mask data is in [0, 1]

            if self.transform:
                image = self.transform(image)

            self.preloaded_images.append(image)
            self.preloaded_masks.append(gt)

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if idx < self.num_preload:
            image = self.preloaded_images[idx]
            gt = self.preloaded_masks[idx]
        else:
            true_idx = self.indices[idx]
            image = self.h5data['images'][true_idx]
            gt = self.h5data['ground_truth_masks'][true_idx]

            image = Image.fromarray(image)  # Assuming the image data is in [0, 1]
            gt = Image.fromarray(gt)  # Assuming the mask data is in [0, 1]

            if self.transform:
                image = self.transform(image)

        if random.random() > 0.5:
            image = TF.hflip(image)
            gt = TF.hflip(gt)
        if random.random() > 0.5:
            image = TF.vflip(image)
            gt = TF.vflip(gt)

        image = TF.to_tensor(image)
        gt = TF.to_tensor(gt)

        return image, gt

    def __del__(self):
        self.h5data.close()