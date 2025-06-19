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
from nltk.translate.bleu_score import corpus_bleu

# Set device to GPU (if available) or CPU
device = torch.device("mps")

# Data parameters
dataFolder = 'flickr8kDataset/inputFiles'
dataName = 'flickr8k_5_cap_per_img_5_min_word_freq'

# Model parameters
embDim = 512  # dimension of word embeddings
attentionDim = 512  # dimension of attention linear layers
decoderDim = 512  # dimension of decoder RNN
dropout = 0.5

# Training parameters
startEpoch = 0
epochs = 20  # number of epochs to train for (if early stopping is not triggered)
batchSize = 32
encoderLr = 1e-4  # learning rate for encoder if fine-tuning
decoderLr = 4e-4  # learning rate for decoder
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fineTuneEncoder = False  # fine-tune encoder


def main():

    global best_bleu4, start_epoch, fine_tune_encoder, dataName, wordMap

    # Load word map
    word_map_file = os.path.join(dataFolder, 'WORDMAP_' + dataName + '.json')
    with open(word_map_file, 'r') as j:
        wordMap = json.load(j)

    decoder = DecoderWithAttention(attention_dim=attentionDim, embed_dim=embDim, decoder_dim=decoderDim, vocab_size=len(wordMap), dropout=dropout)
    decoderOptimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoderLr)

    encoder = Encoder()
    encoder.fine_tune(fineTuneEncoder)
    if fineTuneEncoder:
        encoderOptimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=encoderLr)
    else:
        encoderOptimizer = None
    

    decoder = decoder.to(device)
    encoder = encoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    trainDataset = CaptionDataset(dataFolder, dataName, 'TRAIN', transform=transforms.Compose([normalize]))
    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=4, pin_memory=True)
    valDataset = CaptionDataset(dataFolder, dataName, 'VAL', transform=transforms.Compose([normalize]))
    valDataLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=4, pin_memory=True)

    for epoch in range(startEpoch, epochs):

        train(trainDataLoader=trainDataLoader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            encoderOptimizer=encoderOptimizer,
            decoderOptimizer=decoderOptimizer,
            epoch=epoch)
        
        recentBleu4 = validate(valDataLoader=valDataLoader,
                            encoder=encoder,
                            decoder=decoder,
                            criterion=criterion)


def train(trainDataLoader, encoder, decoder, criterion, encoderOptimizer, decoderOptimizer, epoch):

    encoder.train()
    decoder.train()

    metrics = {
        'lossSum': 0.0,
        'lossCount': 0,
        'batchTimes': [],
        'dataTimes': [],
    }

    start = time.time()
    for i, (imgs, caps, caplens) in enumerate(trainDataLoader):
        if (i == 1):
            break

        dataTime = time.time() - start

        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        imgs = encoder(imgs)
        scores, capsSorted, decodeLengths, alphas, sortInd = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = capsSorted[:, 1:]  # still in the form of indices

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decodeLengths, batch_first=True).data  # scores are logits
        targets = pack_padded_sequence(targets, decodeLengths, batch_first=True).data

        loss = criterion(scores, targets)

        if encoderOptimizer is not None:
            encoderOptimizer.zero_grad()
        decoderOptimizer.zero_grad()
        loss.backward()

        if encoderOptimizer is not None:
            encoderOptimizer.step()
        decoderOptimizer.step()

        metrics['dataTimes'].append(dataTime)
        numWords = sum(decodeLengths)
        metrics['lossSum'] += loss.item() * numWords
        metrics['lossCount'] += numWords
        metrics['batchTimes'].append(time.time() - start)
        start = time.time()
    
    avgLoss = metrics['lossSum'] / metrics['lossCount']
    avgBatchTime = sum(metrics['batchTimes']) / len(metrics['batchTimes'])
    avgDataTime = sum(metrics['dataTimes']) / len(metrics['dataTimes'])

    print(f"Epoch {epoch}: Training Loss = {avgLoss:.4f}")


def validate(valDataLoader, encoder, decoder, criterion):

    decoder.eval()  
    if encoder is not None:
        encoder.eval()

    metrics = {
        'lossSum': 0.0,
        'lossCount': 0,
        'batchTimes': [],
    }

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(valDataLoader):
            if (i == 1):
                break

            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            if encoder is not None:
                imgs = encoder(imgs)
            scores, capsSorted, decodeLengths, alphas, sortInd = decoder(imgs, caps, caplens)

            targets = capsSorted[:, 1:]

            scoresCopy = scores.clone()
            scores = pack_padded_sequence(scores, decodeLengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decodeLengths, batch_first=True).data

            loss = criterion(scores, targets)

            numWords = sum(decodeLengths)
            metrics['lossSum'] += loss.item() * numWords
            metrics['lossCount'] += numWords
            metrics['batchTimes'].append(time.time() - start)
            start = time.time()

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            sortInd = sortInd.to(device)
            allcaps = allcaps.to(device)
            allcaps = allcaps[sortInd]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                imgCaps = allcaps[j].tolist()
                imgCaptions = list(
                    map(lambda c: [w for w in c if w not in {wordMap['<start>'], wordMap['<pad>']}],
                        imgCaps))  # remove <start> and pads
                references.append(imgCaptions)

            # Hypotheses
            _, preds = torch.max(scoresCopy, dim=2)
            preds = preds.tolist()
            tempPreds = list()
            for j, p in enumerate(preds):
                tempPreds.append(preds[j][:decodeLengths[j]])  # remove pads
            preds = tempPreds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)
        
        bleu4 = corpus_bleu(references, hypotheses)

        avgLoss = metrics['lossSum'] / metrics['lossCount']
        avgBatchTime = sum(metrics['batchTimes']) / len(metrics['batchTimes'])

        print(f"Validation Loss = {avgLoss:.4f}, BLEU-4 = {bleu4:.4f}")
    
    return bleu4


    

if __name__ == '__main__':
    main()
           