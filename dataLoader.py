import h5py
import json
import torch
from torch.utils.data import Dataset
import os

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
