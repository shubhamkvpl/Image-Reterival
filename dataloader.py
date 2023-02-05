import random
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils


class TripletData(Dataset):
    def __init__(self, path, transforms, split="train"):
        self.path = path
        self.split = split  # train or valid
        self.cats = 102  # number of categories
        self.transforms = transforms

    def __getitem__(self, idx):
        # our positive class for the triplet
        idx = str(idx % self.cats + 1)

        # choosing our pair of positive images (im1, im2)
        positives = os.listdir(os.path.join(self.path, idx))
        im1, im2 = random.sample(positives, 2)

        # choosing a negative class and negative image (im3)
        negative_cats = [str(x + 1) for x in range(self.cats)]
        negative_cats.remove(idx)
        negative_cat = str(random.choice(negative_cats))
        negatives = os.listdir(os.path.join(self.path, negative_cat))
        im3 = random.choice(negatives)

        im1, im2, im3 = os.path.join(self.path, idx, im1), os.path.join(self.path, idx, im2), os.path.join(self.path,
                                                                                                           negative_cat,
                                                                                                           im3)

        im1 = self.transforms(Image.open(im1))
        im2 = self.transforms(Image.open(im2))
        im3 = self.transforms(Image.open(im3))

        return [im1, im2, im3]

    # we'll put some value that we want since there can be far too many triplets possible
    # multiples of the number of images/ number of categories is a good choice
    def __len__(self):
        return self.cats * 8