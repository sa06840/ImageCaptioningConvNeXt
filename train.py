import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.use_deterministic_algorithms(True)

# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

set_seed(42)
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
from models.transformerDecoder import TransformerDecoder
from models.transformerDecoderHF import HFTransformerDecoder
from dataLoader import CaptionDataset
from utils.utils import *
import argparse

# Set device to GPU (if available) or CPU
device = torch.device("mps")

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
batchSize = 32  #32
workers = 6
encoderLr = 1e-4  # learning rate for encoder if fine-tuning
decoderLr = 1e-4  # learning rate for decoder
gradClip = 5.  # clip gradients at an absolute value of
alphaC = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
bestBleu4 = 0.  # BLEU-4 score right now
printFreq = 100  # print training/validation stats every __ batches
fineTuneEncoder = False  # fine-tune encoder
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
parser.add_argument('--lstmDecoder', action='store_true', help='Use LSTM decoder instead of Transformer')
parser.add_argument('--teacherForcing', action='store_true', help='Use teacher forcing training strategy')
args = parser.parse_args()
# checkpoint = args.checkpoint
# lstmDecoder = args.lstmDecoder
# teacherForcing = args.teacherForcing
checkpoint = None
lstmDecoder = False
teacherForcing = True

def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def main():

    global bestBleu4, epochsSinceImprovement, checkpoint, startEpoch, fineTuneEncoder, dataName, wordMap

    # Load word map
    wordMapFile = os.path.join(dataFolder, 'WORDMAP_' + dataName + '.json')
    with open(wordMapFile, 'r') as j:
        wordMap = json.load(j)

    if checkpoint is None:
        if lstmDecoder is True:
            decoder = DecoderWithAttention(attention_dim=attentionDim, embed_dim=embDim, decoder_dim=decoderDim, vocab_size=len(wordMap), dropout=dropout, device=device)
        else:
            # decoder = TransformerDecoder(embed_dim=embDim, decoder_dim=decoderDim, vocab_size=len(wordMap), maxLen=maxLen, dropout=dropout, device=device)
            decoder = HFTransformerDecoder(vocab_size=len(wordMap), device=device, wordMap=wordMap)
        decoderOptimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoderLr)
        encoder = Encoder()
        encoder.fine_tune(fineTuneEncoder)
        if fineTuneEncoder is True:
            encoderOptimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=encoderLr)
        else:
            encoderOptimizer = None
        results = []
    else:
        if lstmDecoder is True:
            decoder = DecoderWithAttention(attention_dim=attentionDim, embed_dim=embDim, decoder_dim=decoderDim, vocab_size=len(wordMap), dropout=dropout, device=device)
        else:
            # decoder = TransformerDecoder(embed_dim=embDim, decoder_dim=decoderDim, vocab_size=len(wordMap), maxLen=maxLen, dropout=dropout, device=device)
            decoder = HFTransformerDecoder(vocab_size=len(wordMap), device=device, wordMap=wordMap)
        decoderOptimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoderLr)
        encoder = Encoder()
        checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)
        encoder.load_state_dict(checkpoint['encoder'])
        encoder.fine_tune(fineTuneEncoder)
        decoder.load_state_dict(checkpoint['decoder'])
        decoderOptimizer.load_state_dict(checkpoint['decoderOptimizer'])
        optimizer_to_device(decoderOptimizer, device)
        if fineTuneEncoder is True:
            encoderOptimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=encoderLr)
            if checkpoint['encoderOptimizer'] is not None:
                encoderOptimizer.load_state_dict(checkpoint['encoderOptimizer'])
                optimizer_to_device(encoderOptimizer, device)
        else:
            encoderOptimizer = None
        startEpoch = checkpoint['epoch'] + 1
        epochsSinceImprovement = checkpoint['epochsSinceImprovement']
        bestBleu4 = checkpoint['bleu-4']
        results = checkpoint['results']
        
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    trainDataset = CaptionDataset(dataFolder, dataName, 'TRAIN', transform=transforms.Compose([normalize]))
    # trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=workers, persistent_workers=True, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=workers, persistent_workers=True, pin_memory=True)
    valDataset = CaptionDataset(dataFolder, dataName, 'VAL', transform=transforms.Compose([normalize]))
    # valDataLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=True, num_workers=workers, persistent_workers=True, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    valDataLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=True, num_workers=workers, persistent_workers=True, pin_memory=True)

    for epoch in range(startEpoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochsSinceImprovement == 20:
            break
        if epochsSinceImprovement > 0 and epochsSinceImprovement % 8 == 0:
            adjust_learning_rate(decoderOptimizer, 0.8)
            if fineTuneEncoder:
                adjust_learning_rate(encoderOptimizer, 0.8)

        if teacherForcing is True:
            trainLoss, trainTop5Acc, trainBatchTime, trainDataTime = trainWithTeacherForcing(trainDataLoader=trainDataLoader,
                encoder=encoder,
                decoder=decoder,
                criterion=criterion,
                encoderOptimizer=encoderOptimizer,
                decoderOptimizer=decoderOptimizer,
                epoch=epoch,
                device=device)
        else:
            trainLoss, trainTop5Acc, trainBatchTime, trainDataTime = trainWithoutTeacherForcing(trainDataLoader=trainDataLoader,
                encoder=encoder,
                decoder=decoder,
                criterion=criterion,
                encoderOptimizer=encoderOptimizer,
                decoderOptimizer=decoderOptimizer,
                epoch=epoch,
                device=device)

        valLoss, valTop5Acc, bleu1, bleu2, bleu3, recentBleu4 = validate(valDataLoader=valDataLoader,
                            encoder=encoder,
                            decoder=decoder,
                            criterion=criterion,
                            device=device)
        
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
        encoderSaved =  encoder.state_dict()
        decoderSaved = decoder.state_dict()
        save_checkpoint(dataName, epoch, epochsSinceImprovement, encoderSaved, decoderSaved, encoderOptimizer,
                        decoderOptimizer, recentBleu4, isBest, results, lstmDecoder)

    resultsDF = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    if lstmDecoder is True:
        resultsDF.to_csv('results/metrics-LSTM(trainingTF-inferenceNoTF).csv', index=False)
    else: 
        resultsDF.to_csv('results/metrics-HFTransformerDecoder(trainingTF-inferenceNoTF).csv', index=False)



def trainWithTeacherForcing(trainDataLoader, encoder, decoder, criterion, encoderOptimizer, decoderOptimizer, epoch, device):

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
            print(f"TF, Epoch {epoch}, Batch {i + 1}/{len(trainDataLoader)}", flush=True)

        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        imgs = encoder(imgs)
        if lstmDecoder is True:
            scores, capsSorted, decodeLengths, alphas, sortInd = decoder(teacherForcing=True, encoder_out=imgs, encoded_captions=caps, caption_lengths=caplens)
            targets = capsSorted[:, 1:]  # still in the form of indices
            scores = pack_padded_sequence(scores, decodeLengths, batch_first=True).data  # scores are logits
            targets = pack_padded_sequence(targets, decodeLengths, batch_first=True).data
            loss = criterion(scores, targets)
            loss += alphaC * ((1. - alphas.sum(dim=1)) ** 2).mean()
        else: 
            # tgt_key_padding_mask = (caps == wordMap['<pad>'])
            # scores, capsSorted, decodeLengths = decoder(imgs, caps, caplens, tgt_key_padding_mask)
            # scores, capsSorted, decodeLengths = decoder(teacherForcing=True, encoder_out=imgs, encoded_captions=caps, caption_lengths=caplens, tgt_key_padding_mask=tgt_key_padding_mask)
            scores, remapped_encoded_captions, decodeLengths = decoder(teacherForcing=True, encoder_out=imgs, encoded_captions=caps, caption_lengths=caplens)
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = remapped_encoded_captions[:, 1:]  # still in the form of indices
            # targets = capsSorted[:, 1:]
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

        top5 = accuracy(scores, targets, 5, 'single')
        # Keep track of metrics
        losses.update(loss.item(), sum(decodeLengths))
        top5accs.update(top5, sum(decodeLengths))
        batchTime.update(time.time() - start)

        start = time.time()

    print(f"TF, Epoch {epoch}: Training Loss = {losses.avg:.4f}, Top-5 Accuracy = {top5accs.avg:.4f}", flush=True)
    return losses.avg, top5accs.avg, batchTime.avg, dataTime.avg


def trainWithoutTeacherForcing(trainDataLoader, encoder, decoder, criterion, encoderOptimizer, decoderOptimizer, epoch, device):
        encoder.train()
        decoder.train()

        batchTime = AverageMeter()
        dataTime = AverageMeter() 
        losses = AverageMeter()  
        top5accs = AverageMeter() 
        start = time.time()

        for i, (imgs, caps, caplens) in enumerate(trainDataLoader):
            dataTime.update(time.time() - start)


            if (i % 100 == 0):
                print(f"No TF, Epoch {epoch}, Batch {i + 1}/{len(trainDataLoader)}", flush=True)

            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            imgs = encoder(imgs)
            if lstmDecoder is True:
                scores, alphas, sequences = decoder(teacherForcing=False, encoder_out=imgs, wordMap=wordMap, maxDecodeLen=50)
                scoresUpdated, targetsUpdated, totalTokensEvaluated, actualDecodeLengths = preprocessDecoderOutputForMetrics(scores, sequences, caps, wordMap['<end>'], wordMap['<pad>'], 50)
                loss = criterion(scoresUpdated, targetsUpdated)
                loss += alphaC * ((1. - alphas.sum(dim=1)) ** 2).mean()
            else: 
                # tgt_key_padding_mask = (caps == wordMap['<pad>'])
                # scores, capsSorted, decodeLengths = decoder(imgs, caps, caplens, tgt_key_padding_mask)
                # targets = capsSorted[:, 1:]
                # scores = pack_padded_sequence(scores, decodeLengths, batch_first=True, enforce_sorted=False).data  # scores are logits
                # targets = pack_padded_sequence(targets, decodeLengths, batch_first=True, enforce_sorted=False).data
                # loss = criterion(scores, targets)
                # scores, sequences = decoder(teacherForcing=False, encoder_out=imgs, wordMap=wordMap, maxDecodeLen=50)
                # scoresUpdated, targetsUpdated, totalTokensEvaluated, actualDecodeLengths = preprocessDecoderOutputForMetrics(scores, sequences, caps, wordMap['<end>'], wordMap['<pad>'], 50)
                scores, sequences = decoder(teacherForcing=False, encoder_out=imgs, state='train', maxDecodeLen=50)
                remapped_encoded_captions = decoder.customToT5[caps]
                scoresUpdated, targetsUpdated, totalTokensEvaluated, actualDecodeLengths = preprocessDecoderOutputForMetrics(scores, sequences, remapped_encoded_captions, wordMap['<end>'], decoder.t5_pad_token_id, 50)
                loss = criterion(scoresUpdated, targetsUpdated)

            if encoderOptimizer is not None:
                encoderOptimizer.zero_grad()
            decoderOptimizer.zero_grad()
            loss.backward()

            if gradClip is not None:
                clip_gradient(decoderOptimizer, gradClip)
                if encoderOptimizer is not None:
                    clip_gradient(encoderOptimizer, gradClip)

            if encoderOptimizer is not None:
                encoderOptimizer.step()
            decoderOptimizer.step()

            top5 = accuracy(scoresUpdated, targetsUpdated, 5, 'single')
            losses.update(loss.item(), totalTokensEvaluated)
            top5accs.update(top5, totalTokensEvaluated)
            batchTime.update(time.time() - start)

            start = time.time()

        print(f"No TF, Epoch {epoch}: Training Loss = {losses.avg:.4f}, Top-5 Accuracy = {top5accs.avg:.4f}", flush=True)
        return losses.avg, top5accs.avg, batchTime.avg, dataTime.avg


def validate(valDataLoader, encoder, decoder, criterion, device):

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
                print(f"No TF, Validation Batch {i + 1}/{len(valDataLoader)}", flush=True)

            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            if encoder is not None:
                imgs = encoder(imgs)

            if lstmDecoder is True:
                scores, alphas, sequences = decoder(teacherForcing=False, encoder_out=imgs, wordMap=wordMap, maxDecodeLen=50)
                scoresUpdated, targetsUpdated, totalTokensEvaluated, actualDecodeLengths = preprocessDecoderOutputForMetrics(scores, sequences, caps, wordMap['<end>'], wordMap['<pad>'], 50)
                loss = criterion(scoresUpdated, targetsUpdated)
                # Add doubly stochastic attention regularization
                loss += alphaC * ((1. - alphas.sum(dim=1)) ** 2).mean()
            else:     
                # tgt_key_padding_mask = (caps == wordMap['<pad>'])
                # scores, capsSorted, decodeLengths = decoder(imgs, caps, caplens, tgt_key_padding_mask)
                # targets = capsSorted[:, 1:]
                # scoresCopy = scores.clone()
                # scores = pack_padded_sequence(scores, decodeLengths, batch_first=True, enforce_sorted=False).data
                # targets = pack_padded_sequence(targets, decodeLengths, batch_first=True, enforce_sorted=False).data
                # loss = criterion(scores, targets)
                # scores, sequences = decoder(teacherForcing=False, encoder_out=imgs, wordMap=wordMap, maxDecodeLen=50)
                # scoresUpdated, targetsUpdated, totalTokensEvaluated, actualDecodeLengths = preprocessDecoderOutputForMetrics(scores, sequences, caps, wordMap['<end>'], wordMap['<pad>'], 50)
                scores, sequences = decoder(teacherForcing=False, encoder_out=imgs, state='inference', maxDecodeLen=50)
                remapped_encoded_captions = decoder.customToT5[caps]
                scoresUpdated, targetsUpdated, totalTokensEvaluated, actualDecodeLengths = preprocessDecoderOutputForMetrics(scores, sequences, remapped_encoded_captions, wordMap['<end>'], decoder.t5_pad_token_id, 50)
                loss = criterion(scoresUpdated, targetsUpdated)

            top5 = accuracy(scoresUpdated, targetsUpdated, 5, 'single')
            losses.update(loss.item(), totalTokensEvaluated)
            top5accs.update(top5, totalTokensEvaluated)
            batchTime.update(time.time() - start)

            start = time.time()

            # References
            allcaps = allcaps.to(device)
            for j in range(allcaps.shape[0]): # Iterate through each image in the batch
                imgCaps = allcaps[j].tolist() # This would be a list of lists, where each inner list is a reference
                imgCaptions = []
                for c_list in imgCaps: # Iterate through each reference caption for the current image
                    filtered_caption = [w for w in c_list if w not in {wordMap['<start>'], wordMap['<pad>']}]
                    imgCaptions.append(filtered_caption)
                references.append(imgCaptions)
            
            # Hypotheses
            batchHypotheses = [] # Create a temporary list to hold all captions for this batch
            for j, p_seq_tensor in enumerate(sequences):
                truncated_predicted_list = p_seq_tensor[:actualDecodeLengths[j]].tolist()
                batchHypotheses.append(truncated_predicted_list) 
            hypotheses.extend(batchHypotheses) 

            assert len(references) == len(hypotheses)
        
        # bleu4 = corpus_bleu(references, hypotheses)
        bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0.0, 0.0, 0.0))
        bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0.0, 0.0))
        bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0.0))
        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

        print(f"No TF, Validation Loss = {losses.avg:.4f}, Top-5 Accuracy = {top5accs.avg:.4f}, Bleu-1 = {bleu1:.4f}, Bleu-2 = {bleu2:.4f}, Bleu-3 = {bleu3:.4f}, Bleu-4 = {bleu4:.4f}", flush=True)
    
    return losses.avg, top5accs.avg, bleu1, bleu2, bleu3, bleu4


    
if __name__ == '__main__':
    main()