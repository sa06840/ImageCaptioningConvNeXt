import torch
from torch.utils.data import DataLoader
from models.encoder import Encoder 
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