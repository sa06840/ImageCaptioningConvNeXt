import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import random
import numpy as np

# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.use_deterministic_algorithms(True)
#     os.environ["PYTHONHASHSEED"] = str(seed)

# set_seed(42)

# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

# g = torch.Generator()
# g.manual_seed(42)

from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import json
import time
from torch import nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
from models.encoder import Encoder 
from models.decoder import DecoderWithAttention
# from modelsFile import Encoder, DecoderWithAttention
from models.transformerDecoder import TransformerDecoder
from dataLoader import CaptionDataset
from utils.utils import *

# Set device to GPU (if available) or CPU
device = torch.device("cuda")

# Data parameters
# dataFolder = 'flickr8kDataset/inputFiles'
# dataName = 'flickr8k_5_cap_per_img_5_min_word_freq'
dataFolder = 'cocoDataset/inputFiles'
dataName = 'coco_5_cap_per_img_5_min_word_freq'

# Model parameters
embDim = 512  # dimension of word embeddings
attentionDim = 512  # dimension of attention linear layers
decoderDim = 512  # dimension of decoder RNN
dropout = 0.5
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
# cudnn.deterministic = True # for reproducibility
maxLen = 52 # maximum length of captions (in words), used for padding

# Training parameters
startEpoch = 0
epochs = 2  # number of epochs to train for (if early stopping is not triggered)
epochsSinceImprovement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batchSize = 128   #32
workers = 6
encoderLr = 1e-4  # learning rate for encoder if fine-tuning
decoderLr = 1e-4  # learning rate for decoder
gradClip = 5.  # clip gradients at an absolute value of
alphaC = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
bestBleu4 = 0.  # BLEU-4 score right now
printFreq = 100  # print training/validation stats every __ batches
fineTuneEncoder = False  # fine-tune encoder
checkpoint = None  # path to checkpoint, None if none
lstmDecoder = True  # use LSTM decoder instead of Transformer decoder



def main():

    global bestBleu4, epochsSinceImprovement, checkpoint, startEpoch, fineTuneEncoder, dataName, wordMap

    # Load word map
    wordMapFile = os.path.join(dataFolder, 'WORDMAP_' + dataName + '.json')
    with open(wordMapFile, 'r') as j:
        wordMap = json.load(j)

    if checkpoint is None:
        if lstmDecoder is True:
            decoder = DecoderWithAttention(attention_dim=attentionDim, embed_dim=embDim, decoder_dim=decoderDim, vocab_size=len(wordMap), dropout=dropout)
        else:
            decoder = TransformerDecoder(embed_dim=embDim, decoder_dim=decoderDim, vocab_size=len(wordMap), maxLen=maxLen, dropout=dropout)
        decoderOptimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoderLr)
        encoder = Encoder()
        encoder.fine_tune(fineTuneEncoder)
        if fineTuneEncoder is True:
            encoderOptimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=encoderLr)
        else:
            encoderOptimizer = None
        results = []
    else:
        checkpoint = torch.load(checkpoint)
        startEpoch = checkpoint['epoch'] + 1
        epochsSinceImprovement = checkpoint['epochsSinceImprovement']
        bestBleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoderOptimizer = checkpoint['decoderOptimizer']
        encoder = checkpoint['encoder']
        encoderOptimizer = checkpoint['encoderOptimizer']
        if fineTuneEncoder is True and encoderOptimizer is None:
            encoder.fine_tune(fineTuneEncoder)
            encoderOptimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=encoderLr)
        results = checkpoint['results']
        
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    trainDataset = CaptionDataset(dataFolder, dataName, 'TRAIN', transform=transforms.Compose([normalize]))
    # trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=workers, persistent_workers=True, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=workers, persistent_workers=True, pin_memory=True)
    # trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=workers, pin_memory=True)
    valDataset = CaptionDataset(dataFolder, dataName, 'VAL', transform=transforms.Compose([normalize]))
    valDataLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=True, num_workers=workers, persistent_workers=True, pin_memory=True)
    # valDataLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=True, num_workers=workers, pin_memory=True)

    for epoch in range(startEpoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochsSinceImprovement == 20:
            break
        if epochsSinceImprovement > 0 and epochsSinceImprovement % 8 == 0:
            adjust_learning_rate(decoderOptimizer, 0.8)
            if fineTuneEncoder:
                adjust_learning_rate(encoderOptimizer, 0.8)

        trainLoss, trainTop5Acc, trainBatchTime, trainDataTime = train(trainDataLoader=trainDataLoader,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            encoderOptimizer=encoderOptimizer,
            decoderOptimizer=decoderOptimizer,
            epoch=epoch)
        
        valLoss, valTop5Acc, bleu1, bleu2, bleu3, recentBleu4 = validate(valDataLoader=valDataLoader,
                            encoder=encoder,
                            decoder=decoder,
                            criterion=criterion)
        
        results.append({
            'epoch': epoch,
            'trainLoss': trainLoss,
            'trainTop5Acc': trainTop5Acc,
            'trainBatchTime': trainBatchTime,
            'trainDataTime': trainDataTime,
            'valLoss': valLoss,
            'valTop5Acc': valTop5Acc,
            'bleu1': bleu1,
            'bleu2': bleu2,
            'bleu3': bleu3,
            'bleu4': recentBleu4
        })

        # Check if there was an improvement
        isBest = recentBleu4 > bestBleu4
        bestBleu4 = max(recentBleu4, bestBleu4)
        if not isBest:
            epochsSinceImprovement += 1
            print("\nEpochs since last improvement: %d\n" % (epochsSinceImprovement,))
        else:
            epochsSinceImprovement = 0

        #  Save checkpoint
        save_checkpoint(dataName, epoch, epochsSinceImprovement, encoder, decoder, encoderOptimizer,
                        decoderOptimizer, recentBleu4, isBest, results)

    resultsDF = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    resultsDF.to_csv('results/metrics-lstmDecoder(6workers-45gbRAM-noReproducibility-128bs).csv', index=False)



def train(trainDataLoader, encoder, decoder, criterion, encoderOptimizer, decoderOptimizer, epoch):

    encoder.train()
    decoder.train()

    batchTime = AverageMeter()  # forward prop. + back prop. time
    dataTime = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    start = time.time()

    for i, (imgs, caps, caplens) in enumerate(trainDataLoader):
        dataTime.update(time.time() - start)

        if (i % 100 == 0):
            print(f"Epoch {epoch}, Batch {i + 1}/{len(trainDataLoader)}")

        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        imgs = encoder(imgs)
        if lstmDecoder is True:
            scores, capsSorted, decodeLengths, alphas, sortInd = decoder(imgs, caps, caplens)
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = capsSorted[:, 1:]  # still in the form of indices
            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decodeLengths, batch_first=True).data  # scores are logits
            targets = pack_padded_sequence(targets, decodeLengths, batch_first=True).data
            loss = criterion(scores, targets)
            # Add doubly stochastic attention regularization
            loss += alphaC * ((1. - alphas.sum(dim=1)) ** 2).mean()
        else: 
            tgt_key_padding_mask = (caps == wordMap['<pad>'])
            scores, capsSorted, decodeLengths = decoder(imgs, caps, caplens, tgt_key_padding_mask)
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = capsSorted[:, 1:]  # still in the form of indices
            scores = pack_padded_sequence(scores, decodeLengths, batch_first=True, enforce_sorted=False).data  # scores are logits
            targets = pack_padded_sequence(targets, decodeLengths, batch_first=True, enforce_sorted=False).data
            loss = criterion(scores, targets)

        if encoderOptimizer is not None:
            encoderOptimizer.zero_grad()
        decoderOptimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if gradClip is not None:
            clip_gradient(decoderOptimizer, gradClip)
            if encoderOptimizer is not None:
                clip_gradient(encoderOptimizer, gradClip)

        if encoderOptimizer is not None:
            encoderOptimizer.step()
        decoderOptimizer.step()

        top5 = accuracy(scores, targets, 5)

        # Keep track of metrics
        losses.update(loss.item(), sum(decodeLengths))
        top5accs.update(top5, sum(decodeLengths))
        batchTime.update(time.time() - start)

        start = time.time()

        # Print status
        # if i % printFreq == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(trainDataLoader),
        #                                                                   batch_time=batchTime,
        #                                                                   data_time=dataTime, loss=losses,
        #                                                                   top5=top5accs))

    print(f"Epoch {epoch}: Training Loss = {losses.avg:.4f}, Top-5 Accuracy = {top5accs.avg:.4f}")
    return losses.avg, top5accs.avg, batchTime.avg, dataTime.avg


def validate(valDataLoader, encoder, decoder, criterion):

    decoder.eval()  
    if encoder is not None:
        encoder.eval()

    batchTime = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(valDataLoader):

            if (i % 100 == 0):
                print(f"Validation Batch {i + 1}/{len(valDataLoader)}")

            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            if encoder is not None:
                imgs = encoder(imgs)

            if lstmDecoder is True:
                scores, capsSorted, decodeLengths, alphas, sortInd = decoder(imgs, caps, caplens)
                targets = capsSorted[:, 1:]
                scoresCopy = scores.clone()
                scores = pack_padded_sequence(scores, decodeLengths, batch_first=True).data
                targets = pack_padded_sequence(targets, decodeLengths, batch_first=True).data
                loss = criterion(scores, targets)
                # Add doubly stochastic attention regularization
                loss += alphaC * ((1. - alphas.sum(dim=1)) ** 2).mean()
            else:     
                tgt_key_padding_mask = (caps == wordMap['<pad>'])
                scores, capsSorted, decodeLengths = decoder(imgs, caps, caplens, tgt_key_padding_mask)
                targets = capsSorted[:, 1:]
                scoresCopy = scores.clone()
                scores = pack_padded_sequence(scores, decodeLengths, batch_first=True, enforce_sorted=False).data
                targets = pack_padded_sequence(targets, decodeLengths, batch_first=True, enforce_sorted=False).data
                loss = criterion(scores, targets)

            top5 = accuracy(scores, targets, 5)

            losses.update(loss.item(), sum(decodeLengths))
            top5accs.update(top5, sum(decodeLengths))
            batchTime.update(time.time() - start)

            start = time.time()

            # if i % printFreq == 0:
            #     print('Validation: [{0}/{1}]\t'
            #           'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(valDataLoader), batch_time=batchTime,
            #                                                                     loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            if lstmDecoder is True:
                sortInd = sortInd.to(torch.device('cuda'))
                allcaps = allcaps.to(torch.device('cuda'))
                allcaps = allcaps[sortInd]  # because images were sorted in the decoder
            else:
                allcaps = allcaps.to(torch.device('cuda'))

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
        
        # bleu4 = corpus_bleu(references, hypotheses)
        bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0.0, 0.0, 0.0))
        bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0.0, 0.0))
        bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0.0))
        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

        print(f"Validation Loss = {losses.avg:.4f}, Top-5 Accuracy = {top5accs.avg:.4f}, Bleu-1 = {bleu1:.4f}, Bleu-2 = {bleu2:.4f}, Bleu-3 = {bleu3:.4f}, Bleu-4 = {bleu4:.4f}")
    
    return losses.avg, top5accs.avg, bleu1, bleu2, bleu3, bleu4


    
if __name__ == '__main__':
    main()