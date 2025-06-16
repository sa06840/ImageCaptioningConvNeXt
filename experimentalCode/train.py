import torch
from torch.utils.data import DataLoader
from models.encoder import Encoder  # Assuming Encoder is saved in models/encoder.py
from dataLoader import CaptionDataset
import torchvision.transforms as transforms
import json
import time
import tensorflow as tf
import numpy as np
import os

if __name__ == '__main__':

    # Set device to GPU (if available) or CPU
    device = torch.device("mps")
    print(device)

    # Load word map
    # with open(word_map_file, 'r') as j:
    #     word_map = json.load(j)

    # tfrecord_file = 'cocoDataset/tfrecInputFiles/TRAIN_IMAGES_coco_5_cap_per_img_5_min_word_freq.tfrecord'
    # def parse_function(proto):
    #     keys_to_features = {
    #         'image': tf.io.FixedLenFeature([], tf.string),  # The image is stored as a byte array
    #     }
    #     parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    #     img_bytes = parsed_features['image']
    #     img = tf.io.decode_jpeg(img_bytes, channels=3)
    #     img = np.array(img)  # Convert to numpy array
    #     img = img.transpose(2, 0, 1)
    #     return img

    # images = []
    # for example in tf.data.TFRecordDataset(tfrecord_file):
    #     img = parse_function(example)
    #     images.append(img)
    # images = np.stack(images)

    # data_folder='cocoDataset/tfrecInputFiles'
    # data_name='coco_5_cap_per_img_5_min_word_freq'
    # split = 'TRAIN' 
    # with open(os.path.join(data_folder, f"{split}_CAPTIONS_{data_name}.json"), 'r') as j:
    #         captions = json.load(j)
    # with open(os.path.join(data_folder, f"{split}_CAPLENS_{data_name}.json"), 'r') as j:
    #         caplens = json.load(j)

    # images = torch.from_numpy(images).float().share_memory_()
    # dataset = CaptionDataset(images, captions, caplens, transform=None)
    # data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = CaptionDataset(data_folder='cocoDataset/inputFiles', data_name='coco_5_cap_per_img_5_min_word_freq', split='TRAIN')
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=False)

    encoder = Encoder(encoded_image_size=7).to(device)

    # Grab a batch of data from the DataLoader
    for epoch in range(2):  # Loop over epochs
        print(f"Epoch {epoch + 1}/{1}")
        start_time = time.time()
        for i, (images, captions, caplen) in enumerate(data_loader):
            images = images.to(device)  # Move images to the device
            print(f"Processing batch {i + 1}...")  # Print batch number (i + 1 for human-friendly index)
            # Pass the sample images through the encoder
            # encoded_images = encoder(images)
            # # Print the shape of the encoded images
            # print(f'Encoded images shape: {encoded_images.shape}')
            # if i == 10:
            #     break

        print("--- %s seconds ---" % (time.time() - start_time))