import os
import socket
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

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
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
import pickle


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
batchSize = 32
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


def reduce_loss(loss, world_size):
    if world_size < 2:
        return loss
    with torch.no_grad():
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss = loss/world_size
    return loss

def gather_all_data(data, world_size, device):
    # Serialize local data (list of lists)
    data_bytes = pickle.dumps(data)
    data_tensor = torch.ByteTensor(list(data_bytes)).to(device)
    # Gather sizes from all processes
    local_size = torch.tensor([data_tensor.numel()], device=device)
    sizes = [torch.tensor([0], device=device) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    max_size = max([s.item() for s in sizes])
    # Pad data tensors to max size
    if local_size < max_size:
        padding = torch.zeros(max_size - local_size, dtype=torch.uint8, device=device)
        data_tensor = torch.cat([data_tensor, padding], dim=0)
    # Gather all tensors
    gathered = [torch.zeros(max_size, dtype=torch.uint8, device=device) for _ in range(world_size)]
    dist.all_gather(gathered, data_tensor)
    # Deserialize and combine on rank 0
    all_data = []
    if dist.get_rank() == 0:
        for i, tensor in enumerate(gathered):
            size = sizes[i].item()
            bytes_i = tensor[:size].cpu().numpy().tobytes()
            data_i = pickle.loads(bytes_i)
            all_data.extend(data_i)
    return all_data

def setup_distributed():
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')  # Use a fixed or random free port
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print(f"[Rank {rank}] is using GPU {local_rank}")
    return rank, local_rank, world_size, device

def main():

    rank, local_rank, world_size, device = setup_distributed()
    global bestBleu4, epochsSinceImprovement, checkpoint, startEpoch, fineTuneEncoder, dataName, wordMap

    # Load word map
    wordMapFile = os.path.join(dataFolder, 'WORDMAP_' + dataName + '.json')
    with open(wordMapFile, 'r') as j:
        wordMap = json.load(j)

    if checkpoint is None:
        if lstmDecoder is True:
            decoder = DecoderWithAttention(attention_dim=attentionDim, embed_dim=embDim, decoder_dim=decoderDim, vocab_size=len(wordMap), dropout=dropout, device=device)
        else:
            decoder = TransformerDecoder(embed_dim=embDim, decoder_dim=decoderDim, vocab_size=len(wordMap), maxLen=maxLen, dropout=dropout, device=device)
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
    decoder = DDP(decoder, device_ids=[local_rank], output_device=local_rank)
    if fineTuneEncoder is True:
        encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank)
    criterion = nn.CrossEntropyLoss().to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    trainDataset = CaptionDataset(dataFolder, dataName, 'TRAIN', transform=transforms.Compose([normalize]))
    trainSampler = torch.utils.data.distributed.DistributedSampler(trainDataset, num_replicas=world_size, rank=rank, shuffle=True)
    # trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=workers, persistent_workers=True, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=False, num_workers=workers, persistent_workers=True, pin_memory=True, sampler=trainSampler)

    valDataset = CaptionDataset(dataFolder, dataName, 'VAL', transform=transforms.Compose([normalize]))
    valSampler = torch.utils.data.distributed.DistributedSampler(valDataset, num_replicas=world_size, rank=rank, shuffle=True)
    # valDataLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=True, num_workers=workers, persistent_workers=True, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    valDataLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=workers, persistent_workers=True, pin_memory=True, sampler=valSampler)

    for epoch in range(startEpoch, epochs):
        trainSampler.set_epoch(epoch) 
        valSampler.set_epoch(epoch)  
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
            epoch=epoch,
            device=device,
            world_size=world_size)
        
        valLoss, valTop5Acc, bleu1, bleu2, bleu3, recentBleu4 = validate(valDataLoader=valDataLoader,
                            encoder=encoder,
                            decoder=decoder,
                            criterion=criterion,
                            device=device,
                            world_size=world_size)
        
        if dist.get_rank() == 0:
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
            
    if dist.get_rank() == 0:
        resultsDF = pd.DataFrame(results)
        os.makedirs('results', exist_ok=True)
        resultsDF.to_csv('results/metrics-lstmDecoder(6workers-45gbRAM-noReproducibility-2GPUs).csv', index=False)



def train(trainDataLoader, encoder, decoder, criterion, encoderOptimizer, decoderOptimizer, epoch, device, world_size):

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

        if (i == 50):
            break

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
            loss = reduce_loss(loss, world_size)
        else: 
            tgt_key_padding_mask = (caps == wordMap['<pad>'])
            scores, capsSorted, decodeLengths = decoder(imgs, caps, caplens, tgt_key_padding_mask)
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = capsSorted[:, 1:]  # still in the form of indices
            scores = pack_padded_sequence(scores, decodeLengths, batch_first=True, enforce_sorted=False).data  # scores are logits
            targets = pack_padded_sequence(targets, decodeLengths, batch_first=True, enforce_sorted=False).data
            loss = criterion(scores, targets)
            loss = reduce_loss(loss, world_size)

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

        correct5, total = accuracy(scores, targets, 5)
        correct5 = torch.tensor(correct5, dtype=torch.float32, device=device)
        total = torch.tensor(total, dtype=torch.float32, device=device)
        dist.all_reduce(correct5, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        top5 = (correct5 / total).item() * 100

        # Keep track of metrics
        losses.update(loss.item(), sum(decodeLengths))
        top5accs.update(top5, total.item())
        batchTime.update(time.time() - start)

        start = time.time()

    batchTimeTensor = torch.tensor(batchTime.avg).to(device)
    dataTimeTensor = torch.tensor(dataTime.avg).to(device)
    dist.all_reduce(batchTimeTensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(dataTimeTensor, op=dist.ReduceOp.SUM)
    batchTimeAvg = batchTimeTensor.item() / world_size
    dataTimeAvg = dataTimeTensor.item() / world_size

    print(f"Epoch {epoch}: Training Loss = {losses.avg:.4f}, Top-5 Accuracy = {top5accs.avg:.4f}")
    # return losses.avg, top5accs.avg, batchTime.avg, dataTime.avg
    return losses.avg, top5accs.avg, batchTimeAvg, dataTimeAvg


def validate(valDataLoader, encoder, decoder, criterion, device, world_size):

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

            if (i == 50):
                break

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
                loss = reduce_loss(loss, world_size)
            else:     
                tgt_key_padding_mask = (caps == wordMap['<pad>'])
                scores, capsSorted, decodeLengths = decoder(imgs, caps, caplens, tgt_key_padding_mask)
                targets = capsSorted[:, 1:]
                scoresCopy = scores.clone()
                scores = pack_padded_sequence(scores, decodeLengths, batch_first=True, enforce_sorted=False).data
                targets = pack_padded_sequence(targets, decodeLengths, batch_first=True, enforce_sorted=False).data
                loss = criterion(scores, targets)
                loss = reduce_loss(loss, world_size)

            correct5, total = accuracy(scores, targets, 5)
            correct5 = torch.tensor(correct5, dtype=torch.float32, device=device)
            total = torch.tensor(total, dtype=torch.float32, device=device)
            dist.all_reduce(correct5, op=dist.ReduceOp.SUM)
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            top5 = (correct5 / total).item() * 100

            losses.update(loss.item(), sum(decodeLengths))
            top5accs.update(top5, total.item())
            batchTime.update(time.time() - start)

            start = time.time()

            # References
            if lstmDecoder is True:
                sortInd = sortInd.to(device)
                allcaps = allcaps.to(device)
                allcaps = allcaps[sortInd]  # because images were sorted in the decoder
            else:
                allcaps = allcaps.to(device)

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

        batchTimeTensor = torch.tensor(batchTime.avg).to(device)
        dist.all_reduce(batchTimeTensor, op=dist.ReduceOp.SUM)
        batchTimeAvg = batchTimeTensor.item() / world_size

        all_references = gather_all_data(references, world_size, device)
        all_hypotheses = gather_all_data(hypotheses, world_size, device)

        if dist.get_rank() == 0:
            bleu1 = corpus_bleu(all_references, all_hypotheses, weights=(1.0, 0.0, 0.0, 0.0))
            bleu2 = corpus_bleu(all_references, all_hypotheses, weights=(0.5, 0.5, 0.0, 0.0))
            bleu3 = corpus_bleu(all_references, all_hypotheses, weights=(0.33, 0.33, 0.33, 0.0))
            bleu4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
            print(f"Validation Loss = {losses.avg:.4f}, Top-5 Accuracy = {top5accs.avg:.4f}, Bleu-1 = {bleu1:.4f}, Bleu-2 = {bleu2:.4f}, Bleu-3 = {bleu3:.4f}, Bleu-4 = {bleu4:.4f}")
        else:
            bleu1 = bleu2 = bleu3 = bleu4 = None
    
    return losses.avg, top5accs.avg, bleu1, bleu2, bleu3, bleu4



if __name__ == '__main__':
    main()
           