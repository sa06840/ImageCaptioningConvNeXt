import torch
from torch.utils.data import DataLoader
from models.encoder import Encoder 
from models.decoder import DecoderWithAttention
from dataLoader import CaptionDataset
import torchvision.transforms as transforms
import json
import time
import tensorflow as tf
import numpy as np
import os
from torch import nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence

if __name__ == '__main__':

    # Set device to GPU (if available) or CPU
    device = torch.device("mps")
    print(device)

    word_map_file = 'cocoDataset/inputFiles/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
    # Load word map
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset = CaptionDataset(data_folder='cocoDataset/inputFiles', data_name='coco_5_cap_per_img_5_min_word_freq', split='TRAIN')
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=False)

    vocab_size = len(word_map)
    fine_tune_encoder = True

    encoder = Encoder()
    encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                            lr=1e-4) if fine_tune_encoder else None
    
    attention_dim = 512
    embed_dim = 512
    decoder_dim = 512
    dropout = 0.5
    decoder = DecoderWithAttention(attention_dim, embed_dim, decoder_dim, vocab_size)
    decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=4e-4)

    decoder = decoder.to(device)
    encoder = encoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)


    # Grab a batch of data from the DataLoader
    for epoch in range(2):  # Loop over epochs
        print(f"Epoch {epoch + 1}/{1}")
        start_time = time.time()
        for i, (images, captions, caplen) in enumerate(data_loader):
            print(f"Processing batch {i + 1}...") 

            images = images.to(device)
            captions = captions.to(device)
            caplen = caplen.to(device)
            
            encoded_images = encoder(images)

            predictions, encoded_captions, decode_lengths, alphas, sort_ind = decoder(encoded_images, captions, caplen)

            # Remove timesteps that we didn't decode at, pack before loss
            targets = encoded_captions[:, 1:]

            # Pack predictions and targets for loss calculation
            predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Compute loss
            loss = criterion(predictions, targets)

            # Backprop
            decoder_optimizer.zero_grad()
            encoder_optimizer.zero_grad()
            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)

            decoder_optimizer.step()
            encoder_optimizer.step()
           