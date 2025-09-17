import torch
from torch import nn
import torchvision
from torchvision.models import ConvNeXt_Base_Weights
import torch.nn.functional as F

# This LSTM without Attention based decoder class is a replication of the DecoderWithAttention class in decoder.py
# with the attention mechanism removed which is explored in this study as a baseline.
# The citations in decoder.py also apply to this class.



class DecoderWithoutAttention(nn.Module):
    def __init__(self, embed_dim, decoder_dim, vocab_size, device, encoder_dim=1024, dropout=0.5):
        super(DecoderWithoutAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution
        self.device = device

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c
    
    def forwardWithTeacherForcing(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            h, c = self.decode_step(
                embeddings[:batch_size_t, t, :],
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, sort_ind

    def forwardWithoutTeacherForcing(self, encoder_out, wordMap, maxDecodeLen):   # This method adapts the forward with teacher forcing method
        batch_size = encoder_out.size(0)                                          # from (Vinodababu, 2019) to implement forward without 
        encoder_dim = encoder_out.size(-1)                                        # teacher forcing. This is a contribution of my study.
        vocab_size = self.vocab_size

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
    
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        start_token_idx = wordMap['<start>']
        end_token_idx = wordMap['<end>']
        inputs = torch.LongTensor([start_token_idx] * batch_size).to(self.device)
        inputs = self.embedding(inputs)  # (batch_size, embed_dim)
    
        predictions = torch.zeros(batch_size, maxDecodeLen, vocab_size).to(self.device)
        sequences = torch.zeros(batch_size, maxDecodeLen, dtype=torch.long).to(self.device) # To store predicted IDs
        # Track finished sequences (those that have predicted the <end> token)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)  # False for all

        # Decoding loop
        for t in range(maxDecodeLen):
            active_indices = (~finished).nonzero(as_tuple=False).squeeze(1)  # (number_of_currently_active_sentences,)
            if len(active_indices) == 0:
                break  # All sequences finished early
        
            h_new, c_new = self.decode_step(
                inputs[active_indices],
                (h[active_indices], c[active_indices]))
        
            preds = self.fc(self.dropout(h_new))  # (active_batch_size, vocab_size)
            predictions[active_indices, t, :] = preds
            
            predicted_ids = preds.argmax(dim=1)  # (active_batch_size) # Greedy prediction: choose the word with the highest probability
            sequences[active_indices, t] = predicted_ids   # stores the generated captions in the form of indices
            finished[active_indices] |= predicted_ids == end_token_idx    # Update finished flags
            inputs[active_indices] = self.embedding(predicted_ids)   #  # Prepare inputs for the next step
            h[active_indices] = h_new    # Update hidden and cell states for active sequences
            c[active_indices] = c_new

        return predictions, sequences

    def forward(self, teacherForcing, encoder_out, encoded_captions=None, caption_lengths=None, wordMap=None, maxDecodeLen=None):
        if teacherForcing is True:
            predictions, encoded_captions, decode_lengths, sort_ind = self.forwardWithTeacherForcing(encoder_out, encoded_captions, caption_lengths)
            return predictions, encoded_captions, decode_lengths, sort_ind
        
        elif teacherForcing is not True:
            predictions, sequences = self.forwardWithoutTeacherForcing(encoder_out, wordMap, maxDecodeLen)
            return predictions, sequences
