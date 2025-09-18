import torch
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random
import numpy as np
from models.encoder import Encoder 
from models.decoder import DecoderWithAttention
from models.lstmNoAttention import DecoderWithoutAttention
from models.transformerDecoder import TransformerDecoder
from models.transformerDecoderAttVis import TransformerDecoderForAttentionViz

def set_seed(seed):
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
import argparse


device = torch.device("cuda")

# Model parameters
embDim = 512  # dimension of word embeddings
attentionDim = 512  # dimension of attention linear layers
decoderDim = 512  # dimension of decoder RNN
dropout = 0.5
maxLen = 52 # maximum length of captions (in words), used for padding

# Data parameters
dataFolder = 'cocoDataset/inputFiles'
dataName = 'coco_5_cap_per_img_5_min_word_freq'

batchSize = 32
workers = 6
alphaC = 1  # regularization parameter for 'doubly stochastic attention', as in the paper
cudnn.benchmark = False # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.deterministic = True # for reproducibility
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
parser.add_argument('--lstmDecoder', action='store_true', help='Use LSTM decoder instead of Transformer')
parser.add_argument('--startingLayer', type=int, default=None, help='Starting layer index for encoder fine-tuning encoder')
parser.add_argument('--embeddingName', type=str, default=None, help='Pretrained embedding name from gensim')
args = parser.parse_args()
modelPath = args.checkpoint
lstmDecoder = args.lstmDecoder
startingLayer = args.startingLayer
pretrainedEmbeddingsName = args.embeddingName  # word2vec-google-news-300

if pretrainedEmbeddingsName == 'word2vec-google-news-300':
    embDim = 300
    pretrainedEmbeddingsPath = 'wordEmbeddings/word2vec-google-news-300.gz'
elif pretrainedEmbeddingsName == 'glove-wiki-gigaword-200':
    embDim = 200
    pretrainedEmbeddingsPath = 'wordEmbeddings/glove-wiki-gigaword-200.gz'
else:
    pretrainedEmbeddingsPath = None

# The main function has been adapted from the main function in train.py hence the same citations apply.
# It has been modified to handle testing the Transformer decoder as well which is a contribution of this study.

def main():
    g = torch.Generator()
    g.manual_seed(42)

    global wordMap

    wordMapFile = os.path.join(dataFolder, 'WORDMAP_' + dataName + '.json')
    with open(wordMapFile, 'r') as j:
        wordMap = json.load(j)

    checkpoint = torch.load(modelPath, map_location=device, weights_only=False)
    
    if lstmDecoder is True:
        decoder = DecoderWithAttention(attention_dim=attentionDim, embed_dim=embDim, decoder_dim=decoderDim, vocab_size=len(wordMap), dropout=dropout, device=device)
    else:
        decoder = TransformerDecoder(embed_dim=embDim, decoder_dim=decoderDim, vocab_size=len(wordMap), maxLen=maxLen, dropout=dropout, device=device,
                                    wordMap=wordMap, pretrained_embeddings_path=pretrainedEmbeddingsPath, fine_tune_embeddings=True)
    encoder = Encoder()
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

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
    if lstmDecoder is True:
        resultsDF.to_csv(f'results/test-lstmDecoder-TeacherForcing-Finetuning{startingLayer}.csv', index=False)
    else:
        resultsDF.to_csv(f'results/test-TransformerDecoder-TeacherForcing-Finetuning{startingLayer}-{pretrainedEmbeddingsName}.csv', index=False)
    

# The original study (Ramos et al., 2024) did not have a test function hence this test function has been adapted from
# the validation function in train.py thus the same citations apply. The test method calls the corresponding non-teacher 
# forcing forward method of each decoder and aligns their outputs in the preprocessDecoderOutputForMetrics function for 
# the evaluation metrics. It also calculates all four BLEU scores. These are contribution of this study.

def test(testDataLoader, encoder, decoder, criterion):
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
                scores, alphas, sequences = decoder(teacherForcing=False, encoder_out=imgs, wordMap=wordMap, maxDecodeLen=51)
                scoresUpdated, targetsUpdated, totalTokensEvaluated, actualDecodeLengths = preprocessDecoderOutputForMetrics(scores, sequences, caps, wordMap['<end>'], wordMap['<pad>'], 51)
                loss = criterion(scoresUpdated, targetsUpdated)
                # Add doubly stochastic attention regularization
                loss += alphaC * ((1. - alphas.sum(dim=1)) ** 2).mean()
            else:
                scores, sequences = decoder(teacherForcing=False, encoder_out=imgs, wordMap=wordMap, maxDecodeLen=51)
                scoresUpdated, targetsUpdated, totalTokensEvaluated, actualDecodeLengths = preprocessDecoderOutputForMetrics(scores, sequences, caps, wordMap['<end>'], wordMap['<pad>'], 51)
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

        print(f"Test Loss = {losses.avg:.4f}, Top-5 Accuracy = {top5accs.avg:.4f}, Bleu-1 = {bleu1:.4f}, Bleu-2 = {bleu2:.4f}, Bleu-3 = {bleu3:.4f}, Bleu-4 = {bleu4:.4f}")
    
    return losses.avg, top5accs.avg, bleu1, bleu2, bleu3, bleu4
    


if __name__ == '__main__':
    main()