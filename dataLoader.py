import h5py
import json
import torch
from torch.utils.data import Dataset
import os

# This class to load images, caption and their lengths is adapted from the codebase of the original study (Ramos et al., 2024).
# Link to their GitHub repository: https://github.com/Leo-Thomas/ConvNeXt-for-Image-Captioning/tree/main
# The original study (Ramos et al., 2024) seem to have adapted their code from another repository (Vinodababu, 2019) 
# which is a popular open source implementation of the 'Show, Attend and Tell' paper (Xu et al., 2015).
# Link to the (Vinodababu, 2019) repository: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
# The original class is modified to support multiple workers, lazy loading of images to avoid OOM issues and faster loading 
# which is a contribution of this study.

class CaptionDataset(Dataset):
    def __init__(self, dataFolder, dataName, split, transform=None):
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        self.dataFolder = dataFolder
        self.dataName = dataName
        # We store path instead of opening hdf5 here
        self.h5_path = os.path.join(dataFolder, self.split + '_IMAGES_' + dataName + '.hdf5')
        self.h = None  # lazy open
        # Load captions fully into memory (lightweight)
        with open(os.path.join(dataFolder, self.split + '_CAPTIONS_' + dataName + '.json'), 'r') as j:
            self.captions = json.load(j)
        with open(os.path.join(dataFolder, self.split + '_CAPLENS_' + dataName + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Load captions_per_image from file attribute
        with h5py.File(self.h5_path, 'r') as h:
            self.cpi = h.attrs['captions_per_image']
            self.dataset_len = len(h['images'])

        self.transform = transform
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        if self.h is None:
            self.h = h5py.File(self.h5_path, 'r')
            self.imgs = self.h['images']

        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)
        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])
        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size

