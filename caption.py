import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from PIL import Image
from models.encoder import Encoder 
from models.decoder import DecoderWithAttention
from models.lstmNoAttention import DecoderWithoutAttention
from models.transformerDecoder import TransformerDecoder
from models.transformerDecoderAttVis import TransformerDecoderForAttentionViz
import csv
import pandas as pd

device = torch.device("cpu")
embDim = 512  
attentionDim = 512 
decoderDim = 512 
dropout = 0.5
maxLen = 52   
lstmDecoder = False

dataFolder = 'cocoDataset/inputFiles'
dataName = 'coco_5_cap_per_img_5_min_word_freq'

# The caption_image_beam_search and visualize_att functions are adapted from the codebase of the original study (Ramos et al., 2024).
# Link to their GitHub repository: https://github.com/Leo-Thomas/ConvNeXt-for-Image-Captioning/tree/main
# The original study (Ramos et al., 2024) seem to have adapted their code from another repository (Vinodababu, 2019) 
# which is a popular open source implementation of the 'Show, Attend and Tell' paper (Xu et al., 2015).
# Link to the (Vinodababu, 2019) repository: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning 
# Some modifications were made to the visualize_att function to overcome errors with displaying the attention weights

def caption_image_beam_search(encoder, decoder, imagePath, wordMap, beamSize=3):
    """
    Reads an image and captions it with beam search.
    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """
    k = beamSize
    vocabSize = len(wordMap)

    # Read image and process
    # img = imread(imagePath)
    img = Image.open(imagePath).convert('RGB')
    img = img.resize((256, 256), Image.Resampling.BICUBIC)
    img = np.array(img)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoderOut = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    encImageSize = encoderOut.size(1)
    encoderDim = encoderOut.size(3)

    # Flatten encoding
    encoderOut = encoderOut.view(1, -1, encoderDim)  # (1, num_pixels, encoder_dim)
    numPixels = encoderOut.size(1)
    # We'll treat the problem as having a batch size of k
    encoderOut = encoderOut.expand(k, numPixels, encoderDim)  # (k, num_pixels, encoder_dim)
    # Tensor to store top k previous words at each step; now they're just <start>
    kPrevWords = torch.LongTensor([[wordMap['<start>']]] * k).to(device)  # (k, 1)
    # Tensor to store top k sequences; now they're just <start>
    seqs = kPrevWords  # (k, 1)
    # Tensor to store top k sequences' scores; now they're just 0
    topKScores = torch.zeros(k, 1).to(device)  # (k, 1)
    # Tensor to store top k sequences' alphas; now they're just 1s
    seqsAlpha = torch.ones(k, 1, encImageSize, encImageSize).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    completeSeqs = list()
    completeSeqsAlpha = list()
    completeSeqsScores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoderOut)

    while True:
        embeddings = decoder.embedding(kPrevWords).squeeze(1)  # (k, embed_dim)
        awe, alpha = decoder.attention(encoderOut, h)  # (k, encoder_dim), (k, num_pixels)
        alpha = alpha.view(-1, encImageSize, encImageSize)  # (k, enc_image_size, enc_image_size)
        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (k, encoder_dim)
        awe = gate * awe
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (k, decoder_dim)

        scores = decoder.fc(h)  # (k, vocab_size)
        scores = F.log_softmax(scores, dim=1)
        # Add
        scores = topKScores.expand_as(scores) + scores  # (k, vocab_size)
        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            topKScores, topKWords = scores[0].topk(k, 0, True, True)  # (k)
        else:
            # Unroll and find top scores, and their unrolled indices
            topKScores, topKWords = scores.view(-1).topk(k, 0, True, True)  # (k)

        # Convert unrolled indices to actual indices of scores
        prevWordInds = topKWords / vocabSize  # (k)
        nextWordInds = topKWords % vocabSize  # (k)
        prevWordInds = prevWordInds.long()  # my addition

         # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prevWordInds], nextWordInds.unsqueeze(1)], dim=1)  # (k, step+1)
        seqsAlpha = torch.cat([seqsAlpha[prevWordInds], alpha[prevWordInds].unsqueeze(1)], dim=1)  # (k, step+1, enc_image_size, enc_image_size)
        
         # Which sequences are incomplete (didn't reach <end>)?
        incompleteInds = [ind for ind, nextWord in enumerate(nextWordInds) if nextWord != wordMap['<end>']]
        completeInds = list(set(range(len(nextWordInds))) - set(incompleteInds))

        # Set aside complete sequences
        if len(completeInds) > 0:
            completeSeqs.extend(seqs[completeInds].tolist())
            completeSeqsAlpha.extend(seqsAlpha[completeInds].tolist())
            completeSeqsScores.extend(topKScores[completeInds])
        k -= len(completeInds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incompleteInds]
        seqsAlpha = seqsAlpha[incompleteInds]
        h = h[prevWordInds[incompleteInds]]
        c = c[prevWordInds[incompleteInds]]
        encoderOut = encoderOut[prevWordInds[incompleteInds]]
        topKScores = topKScores[incompleteInds].unsqueeze(1)
        kPrevWords = nextWordInds[incompleteInds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = completeSeqsScores.index(max(completeSeqsScores))
    seq = completeSeqs[i]
    alphas = completeSeqsAlpha[i]

    return seq, alphas


# This function generates a caption using the transformer decoder but does not return the attention weights since it uses the TransformerDecoder
# class in transformerDecoder.py
def caption_image_beam_search_transformer(encoder, decoder, imagePath, wordMap, beamSize=3, max_decode_len= 51):
    # The initial section of this function is adapted from the caption_image_beam_search function hence the same citations apply.
    k = beamSize
    vocab_size = len(wordMap)
    end_token_idx = wordMap['<end>']

    img = Image.open(imagePath).convert('RGB')
    img = img.resize((256, 256), Image.Resampling.BICUBIC) 
    img = np.array(img) 
    if len(img.shape) == 2: 
        img = np.stack([img, img, img], axis=2) 
    img = img.transpose(2, 0, 1) 
    img = img / 255. 
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  
    image = image.unsqueeze(0) 
    encoderOut = encoder(image)  
    encoderDim = encoderOut.size(3)  

    encoderOutProj = decoder.encoder_proj(encoderOut.view(1, -1, encoderDim)).permute(1, 0, 2)
    encoderOutExpanded = encoderOutProj.expand(-1, k, -1)

    kPrevWords = torch.full((k, 1), wordMap['<start>'], dtype=torch.long, device=device)
    topKScores = torch.zeros(k, 1, device=device) # (k, 1)
    completeSeqs = list()
    completeSeqsScores = list()
    step = 0 
    finishedSequences = torch.zeros(k, dtype=torch.bool, device=device)
    
    # This section of the function is also adapted from the caption_image_beam_search function however, modifications have been 
    # made for the transformer decoder. These modifications are taken from the forwardWithoutTeacherForcing method in
    # transformerDecoder.py for which the Datacamp tutorial (Sarkar, 2025) was used to understand the general structure of the 
    # transformer decoder whereas the TransformerDecoderLayer and TransformerDecoder classes from the PyTorch documentation were 
    # used to implement it. The same citations as TransformerDecoder in transformerDecoder.py apply to this.

    while True:
        active = (~finishedSequences).nonzero(as_tuple=False).squeeze(1)
        if len(active) == 0:
            break 

        kPrevWordsActive = kPrevWords[active]
        encoderOutActive = encoderOutExpanded[:, active, :] 
        embeddingsActive = decoder.embedding(kPrevWordsActive)
        embeddingsActive = decoder.pos_encoding(decoder.dropout(embeddingsActive))
        
        tgtActive = embeddingsActive.permute(1, 0, 2)
        tgtMask = nn.Transformer.generate_square_subsequent_mask(tgtActive.size(0)).to(device).bool()
        
        decoderOutput = decoder.transformer_decoder(
            tgtActive,                             
            encoderOutActive,                
            tgt_mask=tgtMask) 
        
        lastTokenOutputActive = decoderOutput[-1, :, :]
        scoresActive = decoder.fc_out(lastTokenOutputActive) 
        scoresActive = F.log_softmax(scoresActive, dim=1)
        topKScoresActive = topKScores[active] 
        scoresActive = topKScoresActive.expand_as(scoresActive) + scoresActive

        if step == 0: 
            topKScoresNew, topKUnrolledIndices = scoresActive[0].topk(k, 0, True, True) 
        else:
            topKScoresNew, topKUnrolledIndices = scoresActive.view(-1).topk(k, 0, True, True) 

        prevWordActiveIndices = topKUnrolledIndices / vocab_size 
        nextWordsIds = topKUnrolledIndices % vocab_size 
        prevWordActiveIndices = prevWordActiveIndices.long()
        kIndicesForNextStep = active[prevWordActiveIndices]
        newKPrevWordsIds = torch.cat([kPrevWords[kIndicesForNextStep], nextWordsIds.unsqueeze(1)], dim=1) 

        newTopKScores = topKScoresNew.unsqueeze(1) 
        justCompletedMask = (nextWordsIds == end_token_idx)
        justCompletedIndices = torch.nonzero(justCompletedMask, as_tuple=False).squeeze(1)

        if len(justCompletedIndices) > 0:
            completeSeqs.extend(newKPrevWordsIds[justCompletedIndices].tolist())
            completeSeqsScores.extend(newTopKScores[justCompletedIndices].squeeze(1).tolist())
        
        incompleteMask = ~justCompletedMask
        incompleteIndices = torch.nonzero(incompleteMask, as_tuple=False).squeeze(1)
        k -= len(justCompletedIndices) 
        if k == 0:
            break 

        kPrevWords = newKPrevWordsIds[incompleteIndices]
        topKScores = newTopKScores[incompleteIndices]
        finishedSequences = finishedSequences[kIndicesForNextStep[incompleteIndices]] 
        if step + 1 >= max_decode_len:
            break 
        step += 1

    i = completeSeqsScores.index(max(completeSeqsScores))
    seq = completeSeqs[i]
    return seq, None


# This function generates a caption using the transformer decoder and it also returns the attention weights since it uses the
# TransformerDecoderForAttentionViz class in transformerDecoderAttVis.py
def caption_image_beam_search_transformer_attention(encoder, decoder, imagePath, wordMap, filename, beamSize=3, max_decode_len=51):
    # The initial section of this function is adapted from the caption_image_beam_search function hence the same citations apply.
    k = beamSize
    vocab_size = len(wordMap)
    end_token_idx = wordMap['<end>']

    img = Image.open(imagePath).convert('RGB')
    img = img.resize((256, 256), Image.Resampling.BICUBIC) 
    img = np.array(img) 
    if len(img.shape) == 2: 
        img = np.stack([img, img, img], axis=2) 
    img = img.transpose(2, 0, 1) 
    img = img / 255. 
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  
    image = image.unsqueeze(0) 

    encoderOut = encoder(image)  
    encoderDim = encoderOut.size(3)  
    encoderOutProj = decoder.encoder_proj(encoderOut.view(1, -1, encoderDim)).permute(1, 0, 2)
    num_pixels = encoderOutProj.size(0)
    encoderOutExpanded = encoderOutProj.expand(-1, k, -1) # [num_pixels, k, embed_dim]

    kPrevWordsIds = torch.full((k, 1), wordMap['<start>'], dtype=torch.long, device=device)
    topKScores = torch.zeros(k, 1, device=device) # (k, 1)
    seqsAlphas = torch.zeros(k, max_decode_len, num_pixels, device=device) 
    completeSeqs = list()
    completeSeqsAlphas = list()
    completeSeqsScores = list()
    step = 0 
    finishedSequences = torch.zeros(k, dtype=torch.bool, device=device)

    # This section of the function is also adapted from the caption_image_beam_search function however, modifications have been 
    # made for the transformer decoder. These modifications are taken from the forwardWithoutTeacherForcing method in
    # transformerDecoderAttVis.py for which the Datacamp tutorial (Sarkar, 2025) was used to understand the general structure of the 
    # transformer decoder whereas PyTorch's Transformer's official GitHub repository linked to its TransformerDecoderLayer 
    # documentation section was used to implement the CustomerTransformerDecoderLayer. The same citations as TransformerDecoderForAttentionViz 
    # in transformerDecoderAttVis.py apply to this.

    while True:
        active = (~finishedSequences).nonzero(as_tuple=False).squeeze(1)
        if len(active) == 0:
            break 

        kPrevWordsIdsActive = kPrevWordsIds[active] # (active_k, current_seq_len)
        encoderOutActive = encoderOutExpanded[:, active, :] # (num_pixels, active_k, embed_dim)
        embeddingsActive = decoder.embedding(kPrevWordsIdsActive)
        embeddingsActive = decoder.pos_encoding(decoder.dropout(embeddingsActive)) 
        
        tgtActive = embeddingsActive.permute(1, 0, 2)
        tgtMask = nn.Transformer.generate_square_subsequent_mask(tgtActive.size(0)).to(device).bool()
        currentLayerOutput = tgtActive 
        allLayerCrossAttentionsForStep = [] 

        for layer_idx, layer in enumerate(decoder.decoder_layers):
            layer_output, self_attn_weights, cross_attn_weights_current_layer = layer(
                currentLayerOutput, 
                encoderOutActive,      
                tgt_mask=tgtMask,
                output_attentions=True)
            currentLayerOutput = layer_output
            allLayerCrossAttentionsForStep.append(cross_attn_weights_current_layer)

        lastTokenOutputActive = currentLayerOutput[-1, :, :] # [active_k, embed_dim]
        # Project to vocabulary size to get logits
        scoresActive = decoder.fc_out(lastTokenOutputActive) # [active_k, vocab_size]
        scoresActive = F.log_softmax(scoresActive, dim=1) 
        topKScoresActive = topKScores[active] 
        scoresActive = topKScoresActive.expand_as(scoresActive) + scoresActive # (active_k, vocab_size)

        # This next 3 lines of the function were generated using Gemini. They compute the average cross-attention weights
        # across all layers for the current word

        stackedCrossAttentions = torch.stack(allLayerCrossAttentionsForStep, dim=0)
        crossAttnForCurrentToken = stackedCrossAttentions[:, :, :, -1, :]
        avgCrossAttentionPerToken = crossAttnForCurrentToken.mean(dim=(0, 2)) 
        
        if step == 0: 
            topKScoresNew, topKUnrolledIndices = scoresActive[0].topk(k, 0, True, True)
        else:
            topKScoresNew, topKUnrolledIndices = scoresActive.view(-1).topk(k, 0, True, True)

        prevWordActiveIndices = topKUnrolledIndices / vocab_size 
        nextWordIds = topKUnrolledIndices % vocab_size 
        prevWordActiveIndices = prevWordActiveIndices.long()
        originalKIndicesForNextStep = active[prevWordActiveIndices]
        newKPrevWordsIds = torch.cat([kPrevWordsIds[originalKIndicesForNextStep], nextWordIds.unsqueeze(1)], dim=1) 
        newSeqsALphas = torch.zeros(k, max_decode_len, num_pixels, device=device)

        if step > 0:
            newSeqsALphas[:, :step, :] = seqsAlphas[originalKIndicesForNextStep, :step, :] 
        newSeqsALphas[:, step, :] = avgCrossAttentionPerToken[prevWordActiveIndices] 

        newTopKScores = topKScoresNew.unsqueeze(1) 
        justCompletedMask = (nextWordIds == end_token_idx)
        justCompletedIndices = torch.nonzero(justCompletedMask, as_tuple=False).squeeze(1)

        if len(justCompletedIndices) > 0:
            completeSeqs.extend(newKPrevWordsIds[justCompletedIndices].tolist())
            completeSeqsAlphas.extend(newSeqsALphas[justCompletedIndices].tolist())
            completeSeqsScores.extend(newTopKScores[justCompletedIndices].squeeze(1).tolist())
        
        incompleteMask = ~justCompletedMask
        incompleteIndices = torch.nonzero(incompleteMask, as_tuple=False).squeeze(1)

        k -= len(justCompletedIndices) 
        if k == 0:
            break 

        kPrevWordsIds = newKPrevWordsIds[incompleteIndices]
        topKScores = newTopKScores[incompleteIndices]
        seqsAlphas = newSeqsALphas[incompleteIndices]
        finishedSequences = finishedSequences[originalKIndicesForNextStep[incompleteIndices]] 

        if step + 1 >= max_decode_len:
            break 
        step += 1

    i = completeSeqsScores.index(max(completeSeqsScores))
    seq = completeSeqs[i]
    alphas = completeSeqsAlphas[i]
    return seq, alphas


def visualize_att(imagePath, seq, alphas, revWordMap, smooth=True, enc_image_size=7): 

    image = Image.open(imagePath)
    image = image.resize([enc_image_size * 24, enc_image_size * 24], Image.Resampling.LANCZOS) 
    words = [revWordMap[ind] for ind in seq]
    
    num_cols = 5 
    num_rows = int(np.ceil(len(words) / num_cols))
    caption = ' '.join(words)
    print(f"Caption: {caption}")
    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(num_rows, num_cols, t + 1)
        plt.text(0, 1.09, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12, va='bottom', transform=plt.gca().transAxes)
        plt.imshow(image)
        currentAlpha = alphas[t, :] 
        currentAlpha_2d = currentAlpha.reshape(enc_image_size, enc_image_size) 
        if smooth:
            alpha = skimage.transform.pyramid_expand(currentAlpha_2d.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(currentAlpha_2d.numpy(), [enc_image_size * 24, enc_image_size * 24]) 
        if t == 0: 
            plt.imshow(alpha, alpha=0) 
        else:
            plt.imshow(alpha, alpha=0.8) 
        plt.set_cmap(cm.Greys_r) 
        plt.axis('off') 

    plt.subplots_adjust(hspace=0.05)
    plt.show()
 
def remap_transformer_decoder_keys(old_state_dict):
    new_state_dict = {}
    for key, value in old_state_dict.items():
        if key.startswith('transformer_decoder.layers.'):
            new_key = key.replace('transformer_decoder.layers.', 'decoder_layers.')
        elif key.startswith('transformer_decoder.encoder_proj.'):
            new_key = key.replace('transformer_decoder.encoder_proj.', 'encoder_proj.')
        elif key.startswith('dropout.'):
             new_key = key.replace('dropout.', 'dropout_layer.')
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    args = parser.parse_args()

    # img = 'cocoDataset/trainval2014/val2014/COCO_val2014_000000394240.jpg'
    # img = 'cocoDataset/trainval2014/val2014/COCO_val2014_000000184791.jpg'
    img = 'cocoDataset/trainval2014/val2014/COCO_val2014_000000334321.jpg'
    # img = 'cocoDataset/trainval2014/val2014/COCO_val2014_000000292301.jpg'
    # img = 'cocoDataset/trainval2014/val2014/COCO_val2014_000000154971.jpg'
    # image_list = ['COCO_val2014_000000561100.jpg', 'COCO_val2014_000000354533.jpg', 'COCO_val2014_000000334321.jpg', 
    #               'COCO_val2014_000000368117.jpg', 'COCO_val2014_000000165547.jpg', 'COCO_val2014_000000455859.jpg',
    #               'COCO_val2014_000000290570.jpg', 'COCO_val2014_000000017756.jpg', 'COCO_val2014_000000305821.jpg', 
    #               'COCO_val2014_000000459374.jpg']

    # LSTM
    # model = 'bestCheckpoints/mscoco/17-07-2025(lstmDecoder-trainingTF-inferenceNoTF-noFinetuning)/BEST_checkpoint_LSTM_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    # model = 'bestCheckpoints/mscoco/01-09-2025(lstmNoAttDecoder-trainingTF-inferenceNoTF-noFinetuning)/BEST_checkpoint_LSTM_FinetuningNone_None_coco_5_cap_per_img_5_min_word_freq.pth.tar'

    # training strategies
    # model = 'bestCheckpoints/mscoco/06_20-07-2025(lstmDecoder-trainingNoTF-inferenceNoTF-noFinetuning)/BEST_checkpoint_LSTM_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    # model = 'bestCheckpoints/mscoco/07_20-07-2025(transformerDecoder-trainingNoTF-inferenceNoTF-noFinetuning)/BEST_checkpoint_Transformer_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    # model = 'bestCheckpoints/mscoco/04_17-07-2025(lstmDecoder-trainingTF-inferenceNoTF-noFinetuning)/BEST_checkpoint_LSTM_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    # model = 'bestCheckpoints/mscoco/05_17-07-2025(transformerDecoder-trainingTF-inferenceNoTF-noFinetuning)/BEST_checkpoint_Transformer_coco_5_cap_per_img_5_min_word_freq.pth.tar'

    # Transformer
    # model = 'bestCheckpoints/mscoco/05_17-07-2025(transformerDecoder-trainingTF-inferenceNoTF-noFinetuning)/BEST_checkpoint_Transformer_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    # model = 'bestCheckpoints/mscoco/08_24-07-2025(transformerDecoder-trainingTF-inferenceNoTF-Finetuning5-lr1e4)/BEST_checkpoint_Transformer_Finetuning5_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    # model = 'bestCheckpoints/mscoco/10_28-07-2025(transformerDecoder-trainingTF-inferenceNoTF-Finetuning5-lr1e5-40epochs)/BEST_checkpoint_Transformer_Finetuning5_1e-05_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    # model = 'bestCheckpoints/mscoco/11_01-08-2025(transformerDecoder-trainingTF-inferenceNoTF-Finetuning5-lr1e6-40epochs)/BEST_checkpoint_Transformer_Finetuning5_1e-06_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    # model = 'bestCheckpoints/mscoco/09_24-07-2025(transformerDecoder-trainingTF-inferenceNoTF-Finetuning3-lr1e4)/BEST_checkpoint_Transformer_Finetuning3_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    # model = 'bestCheckpoints/mscoco/12_12-08-2025(transformerDecoder-trainingTF-inferenceNoTF-Finetuning1-lr1e6-40epochs)/BEST_checkpoint_Transformer_Finetuning1_1e-06_coco_5_cap_per_img_5_min_word_freq.pth.tar'

    # model = 'bestCheckpoints/mscoco/04-09-2025(transformerAttDecoder-trainingTF-inferenceNoTF-noFinetuning)/BEST_checkpoint_TransformerAtt_FinetuningNone_None_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    # model = 'bestCheckpoints/mscoco/03-09-2025(transformerAttDecoder-trainingTF-inferenceNoTF-Finetuning5-lr1e4)/BEST_checkpoint_TransformerAtt_Finetuning5_0.0001_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    # model = 'bestCheckpoints/mscoco/03-09-2025(transformerAttDecoder-trainingTF-inferenceNoTF-Finetuning3-lr1e4)/BEST_checkpoint_TransformerAtt_Finetuning3_0.0001_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    # model = 'bestCheckpoints/mscoco/10-09-2025(transformerAttDecoder-trainingTF-inferenceNoTF-Finetuning5-lr1e6)/BEST_checkpoint_TransformerAtt_Finetuning5_1e-06_coco_5_cap_per_img_5_min_word_freq.pth.tar'

    # word embeddings
    # model = 'bestCheckpoints/mscoco/11_01-08-2025(transformerDecoder-trainingTF-inferenceNoTF-Finetuning5-lr1e6-40epochs)/BEST_checkpoint_Transformer_Finetuning5_1e-06_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    # model = 'bestCheckpoints/mscoco/14_31-08-2025(transformerDecoder-trainingTF-Finetuning5-lr1e6-40epochs-wordEmbeddings)/BEST_checkpoint_Transformer_Finetuning5_1e-06_word2vec-google-news-300_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    # model = 'bestCheckpoints/mscoco/14_31-08-2025(transformerDecoder-trainingTF-Finetuning5-lr1e6-40epochs-wordEmbeddings)/BEST_checkpoint_Transformer_Finetuning5_1e-06_glove-wiki-gigaword-200_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    # model = 'bestCheckpoints/mscoco/20_26-09-2025(transformerDecoder-trainingTF-Finetuning5-lr1e6-40epochs-wordEmbeddings)/BEST_checkpoint_Transformer_Finetuning5_1e-06_word2vec-google-news-300_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    model = 'bestCheckpoints/mscoco/20_26-09-2025(transformerDecoder-trainingTF-Finetuning5-lr1e6-40epochs-wordEmbeddings)/BEST_checkpoint_Transformer_Finetuning5_1e-06_glove-wiki-gigaword-200_coco_5_cap_per_img_5_min_word_freq.pth.tar'

    word_map = 'cocoDataset/inputFiles/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
    beamSize = 1
    smooth = False

    wordMapFile = os.path.join(dataFolder, 'WORDMAP_' + dataName + '.json')
    with open(wordMapFile, 'r') as j:
        wordMap = json.load(j)

    checkpoint = torch.load(model, map_location=device, weights_only=False)

    encoder = Encoder()
    encoder.load_state_dict(checkpoint['encoder'])
    if lstmDecoder is True:
        decoder = DecoderWithAttention(attention_dim=attentionDim, embed_dim=embDim, decoder_dim=decoderDim, vocab_size=len(wordMap), dropout=dropout, device=device)
        # decoder = DecoderWithoutAttention(embed_dim=embDim, decoder_dim=decoderDim, vocab_size=len(wordMap), dropout=dropout, device=device)
        decoder.load_state_dict(checkpoint['decoder']) 
    else: 
        # decoder = TransformerDecoderForAttentionViz(embed_dim=embDim, decoder_dim=decoderDim, vocab_size=len(wordMap), maxLen=maxLen, dropout=dropout, device=device) 
        # remapped_decoder_state_dict = remap_transformer_decoder_keys(checkpoint['decoder'])
        # decoder.load_state_dict(remapped_decoder_state_dict)
        # decoder = TransformerDecoder(embed_dim=embDim, decoder_dim=decoderDim, vocab_size=len(wordMap), maxLen=maxLen, dropout=dropout, device=device,
        #                             wordMap=None, pretrained_embeddings_path=None, fine_tune_embeddings=None)
        decoder = TransformerDecoder(embed_dim=200, decoder_dim=decoderDim, vocab_size=len(wordMap), maxLen=maxLen, dropout=dropout, device=device,
                                    wordMap=None, pretrained_embeddings_path='wordEmbeddings/glove-wiki-gigaword-200.gz', fine_tune_embeddings=None)
        decoder.load_state_dict(checkpoint['decoder']) 

    decoder = decoder.to(device)
    encoder = encoder.to(device)
    decoder.eval()
    encoder.eval()
    revWordMap = {v: k for k, v in wordMap.items()}

    if lstmDecoder is True:
        seq, alphas = caption_image_beam_search(encoder, decoder, img, wordMap, beamSize)
        # seq, alphas = caption_image_beam_search_noAtt(encoder, decoder, img, wordMap, beamSize)
    else:
        seq, alphas = caption_image_beam_search_transformer(encoder, decoder, img, wordMap, beamSize, max_decode_len=51)
        # seq, alphas = caption_image_beam_search_transformer_attention(encoder, decoder, img, wordMap, beamSize, max_decode_len=51)

    alphas = torch.FloatTensor(alphas)
    visualize_att(img, seq, alphas, revWordMap, smooth)