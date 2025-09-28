import torch.nn as nn
import math
import torch
import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np
import gzip


# The PositionalEncoding class is adapted from a Datacamp tutorial on how to build a Transformer
# using PyTorch (Sarkar, 2025).
# Link to tutorial: https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, embed_dim)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

def loadPretrainedWordEmbeddings(wordMap, pretrained_embeddings_path, embed_dim):
    newEmbeddingMatrix = np.zeros((len(wordMap), embed_dim))
    if pretrained_embeddings_path == 'wordEmbeddings/word2vec-google-news-300.gz':
        # This line is adapted from a GeeksForGeeks tutorial (GeeksforGeeks, 2025).
        # Link to tutorial: https://www.geeksforgeeks.org/nlp/pre-trained-word-embedding-in-nlp/
        pretrainedEmbeddings = KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=True) 
    else:
        pretrainedEmbeddings = KeyedVectors.load_word2vec_format(pretrained_embeddings_path, binary=False)

    for word, idx in wordMap.items():
        if word in pretrainedEmbeddings:
            newEmbeddingMatrix[idx] = pretrainedEmbeddings[word]
    
    return torch.tensor(newEmbeddingMatrix, dtype=torch.float)


# The TransformerDecoder class is a contribution of this study. The Datacamp tutorial (Sarkar, 2025) 
# was used to understand the general structure of the transformer decoder whereas the TransformerDecoderLayer
# and TransformerDecoder classes from the PyTorch documentation were used to implement this class. 
# 1. PyTorch. TransformerDecoderLayer - PyTorch 2.8 documentation. 
# Available at: https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html 
# 2. PyTorch. TransformerDecoder - PyTorch 2.8 documentation. 
# Available at: https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html 

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, decoder_dim, vocab_size, maxLen, device, wordMap, pretrained_embeddings_path, fine_tune_embeddings,
                dropout=0.5, encoder_dim=1024, num_heads=8, num_layers=6):
        super(TransformerDecoder, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        if pretrained_embeddings_path == 'wordEmbeddings/word2vec-google-news-300.gz':
            num_heads = 6
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        if pretrained_embeddings_path and wordMap:
            pre_trained_embeddings_tensor = loadPretrainedWordEmbeddings(wordMap, pretrained_embeddings_path, embed_dim)
            if pre_trained_embeddings_tensor.shape[1] != embed_dim:
                print('Dimension mismatch for pre-trained embeddings')
                self.embedding = nn.Embedding(vocab_size, embed_dim)
            else:
                self.embedding = nn.Embedding.from_pretrained(pre_trained_embeddings_tensor, freeze=not fine_tune_embeddings, padding_idx=wordMap.get('<pad>'))
                print(f"Loaded and aligned embeddings from '{pretrained_embeddings_path}'")
        else:
            print("Initializing embeddings randomly.")
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.pos_encoding = PositionalEncoding(embed_dim, maxLen)
        self.dropout = nn.Dropout(p=self.dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=decoder_dim, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.encoder_proj = nn.Linear(encoder_dim, embed_dim) if encoder_dim != embed_dim else nn.Identity()
        self.device = device

    def forwardWithTeacherForcing(self, encoder_out, encoded_captions, caption_lengths, tgt_key_padding_mask):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        caption_lengths = caption_lengths.squeeze(1)
        decode_lengths = (caption_lengths - 1).tolist()
    
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        encoder_out = self.encoder_proj(encoder_out).permute(1, 0, 2)  # [num_pixels, batch_size, embed_dim]

        embeddings = self.embedding(encoded_captions)  # [batch_size, max_caption_length, embed_dim]
        embeddings = self.pos_encoding(self.dropout(embeddings))
        tgt = embeddings.permute(1, 0, 2)  # [max_len, batch_size, embed_dim]

        tgt_seq_len = tgt.size(0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(self.device).bool()  # [max_caption_length, max_caption_length]

        decoder_out = self.transformer_decoder(tgt, encoder_out, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)  # [max_len, batch_size, embed_dim]
        decoder_out = decoder_out.permute(1, 0, 2)  # [batch_size, max_caption_length, embed_dim]
        predictions = self.fc_out(decoder_out)  # [batch_size, max_caption_length, vocab_size]

        return predictions, encoded_captions, decode_lengths

    def forwardWithoutTeacherForcing(self, encoder_out, wordMap, maxDecodeLen):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        encoder_out = self.encoder_proj(encoder_out).permute(1, 0, 2)  # [num_pixels, batch_size, embed_dim]

        start_token_idx = wordMap['<start>']
        end_token_idx = wordMap['<end>']
        
        inputs = torch.full((batch_size, 1), start_token_idx, dtype=torch.long, device=self.device) 
        predictions = torch.zeros(batch_size, maxDecodeLen, self.vocab_size, device=self.device) 
        sequences = torch.zeros(batch_size, maxDecodeLen, dtype=torch.long, device=self.device) 
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device) 

        for t in range(maxDecodeLen):
            active_indices = (~finished).nonzero(as_tuple=False).squeeze(1) 
            if len(active_indices) == 0:
                break  

            embeddings = self.embedding(inputs[active_indices]) 
            embeddings = self.pos_encoding(self.dropout(embeddings)) 
            
            tgt = embeddings.permute(1, 0, 2)
            tgt_seq_len = tgt.size(0)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(self.device).bool() 
            
            decoder_output_sliced = self.transformer_decoder(
                tgt,                                    # [current_seq_len, active_batch_size, embed_dim]
                encoder_out[:, active_indices, :],      # [num_pixels, active_batch_size, embed_dim]
                tgt_mask=tgt_mask) # Output shape: [current_seq_len, active_batch_size, embed_dim]

            last_token_output_sliced = decoder_output_sliced[-1, :, :] # [active_batch_size, embed_dim]

            preds = self.fc_out(last_token_output_sliced) 
            predictions[active_indices, t, :] = preds 

            pred_ids = preds.argmax(dim=-1) 
            sequences[active_indices, t] = pred_ids 
            finished[active_indices] |= (pred_ids == end_token_idx) 

            new_full_inputs = torch.full( 
                (batch_size, t + 2), 
                wordMap['<pad>'], 
                dtype=torch.long, 
                device=self.device)
            
            new_full_inputs[:, :t+1] = inputs 
            new_full_inputs[active_indices, t+1] = pred_ids
            inputs = new_full_inputs 
        
        return predictions, sequences

    def forward(self, teacherForcing, encoder_out, encoded_captions=None, caption_lengths=None, tgt_key_padding_mask=None, wordMap=None, maxDecodeLen=None):
        if teacherForcing is True:
            predictions, encoded_captions, decode_lengths = self.forwardWithTeacherForcing(encoder_out, encoded_captions, caption_lengths, tgt_key_padding_mask)
            return predictions, encoded_captions, decode_lengths
        elif teacherForcing is not True:
            predictions, sequences = self.forwardWithoutTeacherForcing(encoder_out, wordMap, maxDecodeLen)
            return predictions, sequences