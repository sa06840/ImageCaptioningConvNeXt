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
# from scipy.misc import imread, imresize
from PIL import Image
from models.encoder import Encoder 
from models.decoder import DecoderWithAttention
from models.transformerDecoder import TransformerDecoder
from models.transformerDecoderAttVis import TransformerDecoderForAttentionViz

device = torch.device("mps")
embDim = 512  # dimension of word embeddings
attentionDim = 512  # dimension of attention linear layers
decoderDim = 512  # dimension of decoder RNN
dropout = 0.5
maxLen = 52   
lstmDecoder = False

dataFolder = 'cocoDataset/inputFiles'
dataName = 'coco_5_cap_per_img_5_min_word_freq'


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
        embeddings = decoder.embedding(kPrevWords).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoderOut, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, encImageSize, encImageSize)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = topKScores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            topKScores, topKWords = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            topKScores, topKWords = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prevWordInds = topKWords / vocabSize  # (s)
        nextWordInds = topKWords % vocabSize  # (s)

        prevWordInds = prevWordInds.long()  # my addition

         # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prevWordInds], nextWordInds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqsAlpha = torch.cat([seqsAlpha[prevWordInds], alpha[prevWordInds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)
        
         # Which sequences are incomplete (didn't reach <end>)?
        incompleteInds = [ind for ind, nextWord in enumerate(nextWordInds) if
                           nextWord != wordMap['<end>']]
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

def caption_image_beam_search_transformer(encoder, decoder, imagePath, wordMap, beamSize=3, max_decode_len= 51):
    k = beam_size
    vocab_size = len(wordMap)
    end_token_idx = wordMap['<end>']
    pad_token_idx = wordMap['<pad>'] 

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
    enc_image_size = encoderOut.size(1) 
    encoderDim = encoderOut.size(3)  

    encoder_out_proj = decoder.encoder_proj(encoderOut.view(1, -1, encoderDim)).permute(1, 0, 2)
    num_pixels = encoder_out_proj.size(0)
    encoder_out_expanded = encoder_out_proj.expand(-1, k, -1)

    k_prev_words_ids = torch.full((k, 1), wordMap['<start>'], dtype=torch.long, device=device)
    top_k_scores = torch.zeros(k, 1, device=device) # (k, 1)

    complete_seqs = list()
    complete_seqs_scores = list()

    step = 0 
    finished_sequences = torch.zeros(k, dtype=torch.bool, device=device) # False for all beams initially
    
    while True:
        active_beam_indices = (~finished_sequences).nonzero(as_tuple=False).squeeze(1)
        if len(active_beam_indices) == 0:
            break 

        k_prev_words_ids_active = k_prev_words_ids[active_beam_indices] # (active_k, current_seq_len)
        encoder_out_active = encoder_out_expanded[:, active_beam_indices, :] # (num_pixels, active_k, embed_dim)

        embeddings_active = decoder.embedding(k_prev_words_ids_active)
        embeddings_active = decoder.pos_encoding(decoder.dropout(embeddings_active)) # Use dropout_layer
        
        tgt_active = embeddings_active.permute(1, 0, 2)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_active.size(0)).to(device).bool()
        
        decoder_output = decoder.transformer_decoder(
            tgt_active,                             # [current_seq_len, k, embed_dim]
            encoder_out_active,                # [num_pixels, k, embed_dim] (full k)
            tgt_mask=tgt_mask) 
        
        last_token_output_active = decoder_output[-1, :, :] # Shape: [active_k, embed_dim]

        # Project to vocabulary size to get logits
        scores_active = decoder.fc_out(last_token_output_active) # [active_k, vocab_size]
        scores_active = F.log_softmax(scores_active, dim=1) # Convert to log-probabilities

        # Add current scores to cumulative scores
        top_k_scores_active = top_k_scores[active_beam_indices] 
        scores_active = top_k_scores_active.expand_as(scores_active) + scores_active # (active_k, vocab_size)

        if step == 0: 
            top_k_scores_new, top_k_unrolled_indices = scores_active[0].topk(k, 0, True, True) # (k)
        else:
            top_k_scores_new, top_k_unrolled_indices = scores_active.view(-1).topk(k, 0, True, True) # (k)


        prev_word_active_indices = top_k_unrolled_indices / vocab_size 
        next_word_ids = top_k_unrolled_indices % vocab_size 
        prev_word_active_indices = prev_word_active_indices.long()

        original_k_indices_for_next_step = active_beam_indices[prev_word_active_indices]
        new_k_prev_words_ids = torch.cat([k_prev_words_ids[original_k_indices_for_next_step], next_word_ids.unsqueeze(1)], dim=1) 

        new_top_k_scores = top_k_scores_new.unsqueeze(1) 
        just_completed_mask = (next_word_ids == end_token_idx)
        just_completed_indices = torch.nonzero(just_completed_mask, as_tuple=False).squeeze(1)

        if len(just_completed_indices) > 0:
            complete_seqs.extend(new_k_prev_words_ids[just_completed_indices].tolist())
            complete_seqs_scores.extend(new_top_k_scores[just_completed_indices].squeeze(1).tolist())
        
        incomplete_mask = ~just_completed_mask
        incomplete_indices = torch.nonzero(incomplete_mask, as_tuple=False).squeeze(1)

        k -= len(just_completed_indices) 
        if k == 0:
            break 

        k_prev_words_ids = new_k_prev_words_ids[incomplete_indices]
        top_k_scores = new_top_k_scores[incomplete_indices]
        finished_sequences = finished_sequences[original_k_indices_for_next_step[incomplete_indices]] 

        if step + 1 >= max_decode_len:
            break 
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    return seq, None

def caption_image_beam_search_transformer_attention(encoder, decoder, image_path, word_map, beam_size=3, max_decode_len=51):
    
    k = beam_size
    vocab_size = len(word_map)
    end_token_idx = word_map['<end>']
    pad_token_idx = word_map['<pad>'] # For padding sequences if needed

    img = Image.open(image_path).convert('RGB')
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
    encoder_out = encoder(image)  
    enc_image_size = encoder_out.size(1) 
    encoder_dim = encoder_out.size(3)  

    encoder_out_proj = decoder.encoder_proj(encoder_out.view(1, -1, encoder_dim)).permute(1, 0, 2)
    num_pixels = encoder_out_proj.size(0)
    encoder_out_expanded = encoder_out_proj.expand(-1, k, -1) # [num_pixels, k, embed_dim]

    k_prev_words_ids = torch.full((k, 1), word_map['<start>'], dtype=torch.long, device=device)
    top_k_scores = torch.zeros(k, 1, device=device) # (k, 1)

    seqs_alphas = torch.zeros(k, max_decode_len, num_pixels, device=device) 

    complete_seqs = list()
    complete_seqs_alphas = list()
    complete_seqs_scores = list()

    step = 0 
    finished_sequences = torch.zeros(k, dtype=torch.bool, device=device) # False for all beams initially

    while True:
        active_beam_indices = (~finished_sequences).nonzero(as_tuple=False).squeeze(1)
        if len(active_beam_indices) == 0:
            break 

        k_prev_words_ids_active = k_prev_words_ids[active_beam_indices] # (active_k, current_seq_len)
        encoder_out_active = encoder_out_expanded[:, active_beam_indices, :] # (num_pixels, active_k, embed_dim)

        embeddings_active = decoder.embedding(k_prev_words_ids_active)
        embeddings_active = decoder.pos_encoding(decoder.dropout(embeddings_active)) # Use dropout_layer
        
        tgt_active = embeddings_active.permute(1, 0, 2)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_active.size(0)).to(device).bool()
        
        current_layer_output = tgt_active # Input to the first decoder layer for active sequences
        all_layer_cross_attentions_for_step = [] # To collect cross-attention from each layer for *this step 't'*

        for layer_idx, layer in enumerate(decoder.decoder_layers): # Iterate through layers (nn.ModuleList of CustomTransformerDecoderLayer)
            layer_output, self_attn_weights, cross_attn_weights_current_layer = layer(
                current_layer_output, 
                encoder_out_active,      # Sliced encoder_memory
                tgt_mask=tgt_mask,
                output_attentions=True # Request attention weights here
            )
            current_layer_output = layer_output # Update output for next layer (input to next layer)
            all_layer_cross_attentions_for_step.append(cross_attn_weights_current_layer)

        last_token_output_active = current_layer_output[-1, :, :] # Shape: [active_k, embed_dim]

        # Project to vocabulary size to get logits
        scores_active = decoder.fc_out(last_token_output_active) # [active_k, vocab_size]
        scores_active = F.log_softmax(scores_active, dim=1) # Convert to log-probabilities

        # Add current scores to cumulative scores
        top_k_scores_active = top_k_scores[active_beam_indices] 
        scores_active = top_k_scores_active.expand_as(scores_active) + scores_active # (active_k, vocab_size)

        stacked_cross_attentions = torch.stack(all_layer_cross_attentions_for_step, dim=0)
        cross_attn_for_current_token = stacked_cross_attentions[:, stacked_cross_attentions.size(1) - 1, :, :, :] 
        avg_cross_attention_per_token = cross_attn_for_current_token.mean(dim=(0, 2)) 
        
        if step == 0: 
            top_k_scores_new, top_k_unrolled_indices = scores_active[0].topk(k, 0, True, True) # (k)
        else:
            top_k_scores_new, top_k_unrolled_indices = scores_active.view(-1).topk(k, 0, True, True) # (k)

        prev_word_active_indices = top_k_unrolled_indices / vocab_size 
        next_word_ids = top_k_unrolled_indices % vocab_size 
        prev_word_active_indices = prev_word_active_indices.long()

        original_k_indices_for_next_step = active_beam_indices[prev_word_active_indices]
        new_k_prev_words_ids = torch.cat([k_prev_words_ids[original_k_indices_for_next_step], next_word_ids.unsqueeze(1)], dim=1) 
        
        new_seqs_alphas = torch.zeros(k, max_decode_len, num_pixels, device=device)
        if step > 0:
            new_seqs_alphas[:, :step, :] = seqs_alphas[original_k_indices_for_next_step, :step, :] 
        new_seqs_alphas[:, step, :] = avg_cross_attention_per_token[prev_word_active_indices] # Add current step's alpha

        new_top_k_scores = top_k_scores_new.unsqueeze(1) 
        just_completed_mask = (next_word_ids == end_token_idx)
        just_completed_indices = torch.nonzero(just_completed_mask, as_tuple=False).squeeze(1)

        if len(just_completed_indices) > 0:
            complete_seqs.extend(new_k_prev_words_ids[just_completed_indices].tolist())
            complete_seqs_alphas.extend(new_seqs_alphas[just_completed_indices].tolist())
            complete_seqs_scores.extend(new_top_k_scores[just_completed_indices].squeeze(1).tolist())
        
        incomplete_mask = ~just_completed_mask
        incomplete_indices = torch.nonzero(incomplete_mask, as_tuple=False).squeeze(1)

        k -= len(just_completed_indices) 
        if k == 0:
            break 

        k_prev_words_ids = new_k_prev_words_ids[incomplete_indices]
        top_k_scores = new_top_k_scores[incomplete_indices]
        seqs_alphas = new_seqs_alphas[incomplete_indices]
        finished_sequences = finished_sequences[original_k_indices_for_next_step[incomplete_indices]] 

        if step + 1 >= max_decode_len:
            break 
        step += 1

    if len(complete_seqs) == 0: 
        best_seq_score_idx = top_k_scores.argmax()
        final_seq = k_prev_words_ids[best_seq_score_idx].tolist()
        final_alphas = seqs_alphas[best_seq_score_idx].tolist()
    else:
        best_seq_score_idx = complete_seqs_scores.index(max(complete_seqs_scores))
        final_seq = complete_seqs[best_seq_score_idx]
        final_alphas = complete_seqs_alphas[best_seq_score_idx]

    return final_seq, final_alphas

def visualize_att(imagePath, seq, alphas, revWordMap, smooth=True, enc_image_size=7): # Added enc_image_size parameter

    image = Image.open(imagePath)
    # Resize original image for display, e.g., to enc_image_size*24 x enc_image_size*24 pixels for a 7x7 or 14x14 feature map
    image = image.resize([enc_image_size * 24, enc_image_size * 24], Image.Resampling.LANCZOS) 
    words = [revWordMap[ind] for ind in seq]
    # Calculate subplot grid dimensions
    num_cols = 5 # Number of subplots per row
    num_rows = int(np.ceil(len(words) / num_cols))
    for t in range(len(words)):
        if t > 50: # Limit displayed words to avoid too many plots
            break
        plt.subplot(num_rows, num_cols, t + 1)
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)

        # currentAlpha = alphas[t, :] # Shape (num_pixels,)
        # currentAlpha_2d = currentAlpha.reshape(enc_image_size, enc_image_size) 
        # if smooth:
        #     alpha = skimage.transform.pyramid_expand(currentAlpha_2d.numpy(), upscale=24, sigma=8)
        # else:
        #     alpha = skimage.transform.resize(currentAlpha_2d.numpy(), [enc_image_size * 24, enc_image_size * 24]) 
        
        # if t == 0: 
        #     plt.imshow(alpha, alpha=0) 
        # else:
        #     plt.imshow(alpha, alpha=0.8) 
            
        # plt.set_cmap(cm.Greys_r) 

        plt.axis('off') 
        
    plt.show() 

def remap_transformer_decoder_keys(old_state_dict):
    new_state_dict = {}
    for key, value in old_state_dict.items():
        if key.startswith('transformer_decoder.layers.'):
            new_key = key.replace('transformer_decoder.layers.', 'decoder_layers.')
        elif key.startswith('transformer_decoder.encoder_proj.'):
            new_key = key.replace('transformer_decoder.encoder_proj.', 'encoder_proj.')
        elif key.startswith('dropout.'): # If you also changed dropout layer name
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

    # args.img = 'cocoDataset/trainval2014/val2014/COCO_val2014_000000391895.jpg'
    # args.model = 'bestCheckpoints/mscoco/17-07-2025(lstmDecoder-trainingTF-inferenceNoTF-noFinetuning)/BEST_checkpoint_LSTM_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    # args.word_map = 'cocoDataset/inputFiles/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
    # args.beam_size = 5
    # args.smooth = False

    img = 'cocoDataset/trainval2014/val2014/COCO_val2014_000000394240.jpg'
    # img = 'cocoDataset/trainval2014/val2014/COCO_val2014_000000184791.jpg'
    # model = 'bestCheckpoints/mscoco/17-07-2025(lstmDecoder-trainingTF-inferenceNoTF-noFinetuning)/BEST_checkpoint_LSTM_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    model = 'bestCheckpoints/mscoco/17-07-2025(transformerDecoder-trainingTF-inferenceNoTF-noFinetuning)/BEST_checkpoint_Transformer_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    word_map = 'cocoDataset/inputFiles/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
    beam_size = 5
    smooth = False

    wordMapFile = os.path.join(dataFolder, 'WORDMAP_' + dataName + '.json')
    with open(wordMapFile, 'r') as j:
        wordMap = json.load(j)

    checkpoint = torch.load(model, map_location=device, weights_only=False)

    encoder = Encoder()
    encoder.load_state_dict(checkpoint['encoder'])
    if lstmDecoder is True:
        decoder = DecoderWithAttention(attention_dim=attentionDim, embed_dim=embDim, decoder_dim=decoderDim, vocab_size=len(wordMap), dropout=dropout, device=device)
        decoder.load_state_dict(checkpoint['decoder']) 
    else: 
        # decoder = TransformerDecoderForAttentionViz(embed_dim=embDim, decoder_dim=decoderDim, vocab_size=len(wordMap), maxLen=maxLen, dropout=dropout, device=device) 
        # remapped_decoder_state_dict = remap_transformer_decoder_keys(checkpoint['decoder'])
        # decoder.load_state_dict(remapped_decoder_state_dict)
        decoder = TransformerDecoder(embed_dim=embDim, decoder_dim=decoderDim, vocab_size=len(wordMap), maxLen=maxLen, dropout=dropout, device=device)
        decoder.load_state_dict(checkpoint['decoder']) 

    decoder = decoder.to(device)
    encoder = encoder.to(device)
    decoder.eval()
    encoder.eval()

    revWordMap = {v: k for k, v in wordMap.items()}
    if lstmDecoder is True:
        seq, alphas = caption_image_beam_search(encoder, decoder, img, wordMap, beam_size)
    else:
        seq, alphas = caption_image_beam_search_transformer(encoder, decoder, img, wordMap, beam_size, max_decode_len=51)

    # alphas = torch.FloatTensor(alphas)

    visualize_att(img, seq, alphas, revWordMap, smooth)

