import torch
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

device = torch.device("mps")

# Data parameters
dataFolder = 'flickr8kDataset/inputFiles'
dataName = 'flickr8k_5_cap_per_img_5_min_word_freq'

batchSize = 32
workers = 4
alphaC = 1  # regularization parameter for 'doubly stochastic attention', as in the paper
cudnn.benchmark = True


def main():

    global wordMap

    wordMapFile = os.path.join(dataFolder, 'WORDMAP_' + dataName + '.json')
    with open(wordMapFile, 'r') as j:
        wordMap = json.load(j)

    modelPath = 'BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'

    checkpoint = torch.load(modelPath, map_location=str(device), weights_only=False)
    decoder = checkpoint['decoder']
    encoder = checkpoint['encoder']

    decoder = decoder.to(device)
    encoder = encoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    testDataset = CaptionDataset(dataFolder, dataName, 'TEST', transform=transforms.Compose([normalize]))
    testDataLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=True, num_workers=workers, persistent_workers=True, pin_memory=True)

    results = []

    testLoss, testTop5Acc, bleu1, bleu2, bleu3, bleu4 = test(testDataLoader=testDataLoader,
                            encoder=encoder,
                            decoder=decoder,
                            criterion=criterion)
    
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
    resultsDF.to_csv('results/test.csv', index=False)
    



def test(testDataLoader, encoder, decoder, criterion):
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
            if (i == 10):
                break

            print(f"Test Batch {i + 1}/{len(testDataLoader)}")

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
            # Add doubly stochastic attention regularization
            loss += alphaC * ((1. - alphas.sum(dim=1)) ** 2).mean()

            top5 = accuracy(scores, targets, 5)

            losses.update(loss.item(), sum(decodeLengths))
            top5accs.update(top5, sum(decodeLengths))
            batchTime.update(time.time() - start)

            start = time.time()


            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            sortInd = sortInd.to(torch.device('mps'))
            allcaps = allcaps.to(torch.device('mps'))
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
        
        # bleu4 = corpus_bleu(references, hypotheses)
        bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0.0, 0.0, 0.0))
        bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0.0, 0.0))
        bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0.0))
        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

        print(f"Test Loss = {losses.avg:.4f}, Top-5 Accuracy = {top5accs.avg:.4f}, Bleu-1 = {bleu1:.4f}, Bleu-2 = {bleu2:.4f}, Bleu-3 = {bleu3:.4f}, Bleu-4 = {bleu4:.4f}")
    
    return losses.avg, top5accs.avg, bleu1, bleu2, bleu3, bleu4
    


if __name__ == '__main__':
    main()