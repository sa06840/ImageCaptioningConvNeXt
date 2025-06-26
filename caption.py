import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
# from scipy.misc import imread, imresize
from PIL import Image


device = torch.device("mps")

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


def visualize_att(imagePath, seq, alphas, revWordMap, smooth=True):
    """
    Visualizes caption with weights at every word.
    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """

    image = Image.open(imagePath)
    image = image.resize([14 * 24, 14 * 24], Image.Resampling.LANCZOS)

    words = [revWordMap[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(int(np.ceil(len(words) / 5)), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        currentAlpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(currentAlpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(currentAlpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()

    args.img = 'flickr8kDataset/Flicker8k_Dataset/3490736665_38710f4b91.jpg'
    args.model = 'bestCheckpoints/24-06-2025(0workers)/BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'
    args.word_map = 'flickr8kDataset/inputFiles/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'
    args.beam_size = 5
    args.smooth = False

    # Load model
    checkpoint = torch.load(args.model, map_location=str(device), weights_only=False)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        wordMap = json.load(j)
    revWordMap = {v: k for k, v in wordMap.items()}  # ix2word

    # Encode, decode with attention and beam search
    seq, alphas = caption_image_beam_search(encoder, decoder, args.img, wordMap, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    # Visualize caption and attention of best sequence
    visualize_att(args.img, seq, alphas, revWordMap, args.smooth)