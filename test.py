import torch
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random
import numpy as np

def set_seed(seed):
    rank = dist.get_rank() if dist.is_initialized() else 0
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataLoader import CaptionDataset
import torchvision.transforms as transforms
import json
import time
import os
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
from utils.utils import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.serialization import add_safe_globals


# device = torch.device("mps")

# Data parameters
dataFolder = 'flickr8kDataset/inputFiles'
dataName = 'flickr8k_5_cap_per_img_5_min_word_freq'
# dataFolder = 'cocoDataset/inputFiles'
# dataName = 'coco_5_cap_per_img_5_min_word_freq'

batchSize = 32
workers = 6
alphaC = 1  # regularization parameter for 'doubly stochastic attention', as in the paper
cudnn.benchmark = False # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.deterministic = True # for reproducibility
lstmDecoder = False  # if True, use LSTM decoder; if False, use Transformer decoder


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
    set_seed(42)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    print(f"[Rank {rank}] is using GPU {local_rank}")
    return rank, local_rank, world_size, device


def main():

    rank, local_rank, world_size, device = setup_distributed()
    g = torch.Generator()
    g.manual_seed(42)

    global wordMap

    wordMapFile = os.path.join(dataFolder, 'WORDMAP_' + dataName + '.json')
    with open(wordMapFile, 'r') as j:
        wordMap = json.load(j)

    modelPath = 'BEST_checkpoint_Transformer_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'
    # checkpoint = torch.load(modelPath, map_location=str(device), weights_only=False)
    checkpoint = torch.load(modelPath, weights_only=False)
    decoder = checkpoint['decoder']
    encoder = checkpoint['encoder']

    if hasattr(decoder, 'module'):
        decoder = decoder.module
    if hasattr(encoder, 'module'):
        encoder = encoder.module

    decoder = decoder.to(device)
    encoder = encoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    testDataset = CaptionDataset(dataFolder, dataName, 'TEST', transform=transforms.Compose([normalize]))
    testDataLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False, num_workers=workers, persistent_workers=True, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    results = []

    testLoss, testTop5Acc, bleu1, bleu2, bleu3, bleu4 = test(testDataLoader=testDataLoader,
                            encoder=encoder,
                            decoder=decoder,
                            criterion=criterion,
                            device=device)
    
    results.append({
        'testLoss': testLoss,
        'testTop5Acc': testTop5Acc,
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu3': bleu3,
        'bleu4': bleu4
    })

    resultsDF = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    resultsDF.to_csv('results/test-transformerDecoder(6workers-45gbRAM-reproducibility-2GPUs).csv', index=False)
    



def test(testDataLoader, encoder, decoder, criterion, device):
    """
    Test the model on the test dataset.
    :param testDataLoader: DataLoader for the test dataset
    :param encoder: Encoder model
    :param decoder: Decoder model
    :param criterion: Loss function
    :return: Average loss and accuracy on the test set
    """

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
        for i, (imgs, caps, caplens, allcaps) in enumerate(testDataLoader):

            print(f"Test Batch {i + 1}/{len(testDataLoader)}")

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
                scores = pack_padded_sequence(scores, decodeLengths, batch_first=True, enforce_sorted=False).data  # scores are logits
                targets = pack_padded_sequence(targets, decodeLengths, batch_first=True, enforce_sorted=False).data
                loss = criterion(scores, targets)


            top5 = accuracySingleGPU(scores, targets, 5)
            losses.update(loss.item(), sum(decodeLengths))
            top5accs.update(top5, sum(decodeLengths))
            batchTime.update(time.time() - start)

            start = time.time()

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

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
        
        # bleu4 = corpus_bleu(references, hypotheses)
        bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0.0, 0.0, 0.0))
        bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0.0, 0.0))
        bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0.0))
        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

        print(f"Test Loss = {losses.avg:.4f}, Top-5 Accuracy = {top5accs.avg:.4f}, Bleu-1 = {bleu1:.4f}, Bleu-2 = {bleu2:.4f}, Bleu-3 = {bleu3:.4f}, Bleu-4 = {bleu4:.4f}")
    
    return losses.avg, top5accs.avg, bleu1, bleu2, bleu3, bleu4
    


if __name__ == '__main__':
    main()