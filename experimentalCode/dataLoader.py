import torch
from torch.utils.data import Dataset
import h5py
import json
import os


# class CaptionDataset(Dataset):
#     """
#     A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
#     """
#     def __init__(self, data_folder, data_name, split, transform=None):
#         """
#         :param data_folder: folder where data files are stored
#         :param data_name: base name of processed datasets
#         :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
#         :param transform: image transform pipeline
#         """
#         self.split = split
#         assert self.split in {'TRAIN', 'VAL', 'TEST'}
#         # Open hdf5 file where images are stored
#         print('check1')
#         self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
#         self.imgs = self.h['images']
        
#         # with h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r') as h:
#         #     self.imgs = h['images'][:] 
#         #     self.cpi = h.attrs['captions_per_image']
            
#         print('check2')
#         print(len(self.imgs))
#         # Captions per image
#         self.cpi = self.h.attrs['captions_per_image']
#         # Load encoded captions (completely into memory)
#         with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
#             self.captions = json.load(j)
#         # Load caption lengths (completely into memory)
#         with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
#             self.caplens = json.load(j)

#         # PyTorch transformation pipeline for the image (normalizing, etc.)
#         self.transform = transform
#         # Total number of datapoints
#         self.dataset_size = len(self.captions)

#     def __getitem__(self, i):
#         # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
#         img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
#         if self.transform is not None:
#             img = self.transform(img)

#         caption = torch.LongTensor(self.captions[i])
#         caplen = torch.LongTensor([self.caplens[i]])

#         if self.split == 'TRAIN':
#             return img, caption, caplen
#         else:
#             # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
#             all_captions = torch.LongTensor(
#                 self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
#             return img, caption, caplen, all_captions

#     def __len__(self):
#         return self.dataset_size


import h5py
import json
import torch
from torch.utils.data import Dataset
import os

class CaptionDataset(Dataset):
    def __init__(self, data_folder, data_name, split, transform=None):
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
        self.data_folder = data_folder
        self.data_name = data_name
        # We store path instead of opening hdf5 here
        self.h5_path = os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5')
        self.h = None  # lazy open
        # Load captions fully into memory (lightweight)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
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







# import tensorflow as tf
# import torch
# from torch.utils.data import Dataset
# import json
# import os
# from PIL import Image
# import numpy as np


# class CaptionDataset(Dataset):
#     """
#     A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
#     """
#     def __init__(self, data_folder, data_name, split, transform=None):
#         """
#         :param data_folder: folder where data files are stored
#         :param data_name: base name of processed datasets
#         :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
#         :param transform: image transform pipeline
#         """
#         self.split = split
#         assert self.split in {'TRAIN', 'VAL', 'TEST'}

#         # Load encoded captions and their lengths (completely into memory)
#         with open(os.path.join(data_folder, f"{self.split}_CAPTIONS_{data_name}.json"), 'r') as j:
#             self.captions = json.load(j)
#         with open(os.path.join(data_folder, f"{self.split}_CAPLENS_{data_name}.json"), 'r') as j:
#             self.caplens = json.load(j)

#         # Image paths are in the TFRecord file, so we don't need to load them directly
#         self.tfrecord_file = os.path.join(data_folder, f"{self.split}_IMAGES_{data_name}.tfrecord")
#         # TFRecord reader setup
#         self.raw_dataset = tf.data.TFRecordDataset(self.tfrecord_file)
#         self.dataset_size = len(self.captions)
#         # Apply the image transformation pipeline (normalization, resizing, etc.)
#         self.transform = transform

#     def _parse_function(self, proto):
#         # Define the feature structure
#         keys_to_features = {
#             'image': tf.io.FixedLenFeature([], tf.string),  # The image is stored as a byte array
#         }
#         # Parse the example
#         parsed_features = tf.io.parse_single_example(proto, keys_to_features)
#         # Decode the image
#         img_bytes = parsed_features['image']
#         img = tf.io.decode_jpeg(img_bytes, channels=3)
#         img = np.array(img)  # Convert to numpy array
#         img = img.transpose(2, 0, 1)

#         return img

#     def __getitem__(self, index):
#         # Get the image from the TFRecord file
#         image = next(iter(self.raw_dataset.skip(index//5).take(1)))  # Get the image from the TFRecord
#         image = self._parse_function(image)
#         if self.transform:
#             image = self.transform(image)
#         caption = torch.LongTensor(self.captions[index])
#         caplen = torch.LongTensor([self.caplens[index]])
#         return image, caption, caplen

#     def __len__(self):
#         return self.dataset_size



# class CaptionDataset(Dataset):
#     def __init__(self, data_folder, data_name, split, transform=None):
#         self.split = split
#         assert self.split in {'TRAIN', 'VAL', 'TEST'}

#         with open(os.path.join(data_folder, f"{self.split}_CAPTIONS_{data_name}.json"), 'r') as j:
#             self.captions = json.load(j)
#         with open(os.path.join(data_folder, f"{self.split}_CAPLENS_{data_name}.json"), 'r') as j:
#             self.caplens = json.load(j)
#         self.tfrecord_file = os.path.join(data_folder, f"{self.split}_IMAGES_{data_name}.tfrecord")
#         self.raw_dataset = tf.data.TFRecordDataset(self.tfrecord_file)

#         # Pre-load all images into memory:
#         self.images = []
#         for example in self.raw_dataset:
#             img = self._parse_function(example)
#             self.images.append(img)

#         self.images = np.stack(self.images)  # shape: (dataset_size, 3, 256, 256)
#         self.dataset_size = len(self.captions)
#         self.transform = transform

#     def _parse_function(self, proto):
#         keys_to_features = {
#             'image': tf.io.FixedLenFeature([], tf.string),
#         }
#         parsed_features = tf.io.parse_single_example(proto, keys_to_features)
#         img_bytes = parsed_features['image']
#         img = tf.io.decode_jpeg(img_bytes, channels=3)
#         img = np.array(img)
#         img = img.transpose(2, 0, 1)
#         # img = img.astype(np.float32) / 255.0  # Normalize (optional)
#         return img

#     def __getitem__(self, index):
#         img = self.images[index//5]
#         if self.transform:
#             img = self.transform(img)
#         img = torch.from_numpy(img)
#         caption = torch.LongTensor(self.captions[index])
#         caplen = torch.LongTensor([self.caplens[index]])
#         return img, caption, caplen

#     def __len__(self):
#         return self.dataset_size


# class CaptionDataset(Dataset):
#     def __init__(self, images, captions, caplens, transform=None):
#         self.images = images
#         self.captions = captions
#         self.caplens = caplens
#         self.transform = transform
#         self.dataset_size = len(captions)

#     def __getitem__(self, index):
#         img = self.images[index // 5]
#         if self.transform:
#             img = self.transform(img)
#         img = torch.from_numpy(img)
#         caption = torch.LongTensor(self.captions[index])
#         caplen = torch.LongTensor([self.caplens[index]])
#         return img, caption, caplen

#     def __len__(self):
#         return self.dataset_size