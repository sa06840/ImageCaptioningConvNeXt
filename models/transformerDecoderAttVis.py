import torch.nn as nn
import torch
import math
from typing import Optional, Tuple
import torch.nn.functional as F 


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
    
# Helper functon for CustomTransformerDecoderLayer taken PyTorch's Transformer's official GitHub repository.
def _get_activation_fn(activation): 
    if activation == "relu": 
        return F.relu
    elif activation == "gelu": 
        return F.gelu

# The CustomTransformerDecoderLayer class is adapted from PyTorch's Transformer's official GitHub repository
# linked to its TransformerDecoderLayer documentation section. 
# Link to the GitHub repository: https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/transformer.py#L966
# The forward function is modified to support capturing self-attention and cross-attention weights which are returned
# for each layer.

class CustomTransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout= 0.1, activation="relu",
                layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout_ffn = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.norm_first = norm_first
        self.batch_first = batch_first

    def forward(self, tgt, memory= None, tgt_mask= None, memory_mask = None, tgt_key_padding_mask= None, memory_key_padding_mask= None, is_causal= False, output_attentions = False):
        x = tgt; 
        attn_weights_sa = None
        if self.norm_first:
            _self_attn_input = self.norm1(x)
        else:
            _self_attn_input = x
        _self_attn_output, attn_weights_sa = self.self_attn(_self_attn_input, _self_attn_input, _self_attn_input, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, is_causal=is_causal, need_weights=output_attentions, average_attn_weights=False)
        x = x + self.dropout1(_self_attn_output)
        if not self.norm_first: 
            x = self.norm1(x)

        attn_weights_ca = None
        if memory is not None:
            if  self.norm_first:
                _cross_attn_input = self.norm2(x)
            else:
                _cross_attn_input = x
            _cross_attn_output, attn_weights_ca = self.multihead_attn(_cross_attn_input, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask, need_weights=output_attentions, average_attn_weights=False)
            x = x + self.dropout2(_cross_attn_output)
            if not self.norm_first: 
                x = self.norm2(x)

        if self.norm_first:
            _ffn_input = self.norm3(x)
        else:
            _ffn_input = x
        _ffn_output = self.linear2(self.dropout_ffn(self.activation(self.linear1(_ffn_input))))
        x = x + self.dropout3(_ffn_output)
        if not self.norm_first: 
            x = self.norm3(x)

        return x, attn_weights_sa, attn_weights_ca


# The TransformerDecoderForAttentionViz class is a contribution of this study. It is adapted from the  
# TransformerDecoder class defined in transformerDecoder.py however, PyTorch's default TransformerDecoderLayer
# is replaced by the CustomerTransformerDecoderLayer defined above to incorporate getting the self-attention and
# cross-attention weights from each decoder layer. The general structure is understood from the Datacamp tutorial
# (Sarkar, 2025) whereas PyTorch's Transformer's official GitHub repository linked to its TransformerDecoderLayer 
# documentation section is used for implementing the CustomerTransformerDecoderLayer.
# Link to the GitHub repository: https://github.com/pytorch/pytorch/blob/v2.8.0/torch/nn/modules/transformer.py#L966

class TransformerDecoderForAttentionViz(nn.Module): 
    def __init__(self, embed_dim, decoder_dim, vocab_size, maxLen, device, dropout=0.5, encoder_dim=1024, num_heads=8, num_layers=6):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, maxLen)
        self.dropout = nn.Dropout(p=self.dropout)

        self.decoder_layers = nn.ModuleList([
            CustomTransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=decoder_dim, dropout=dropout, batch_first=False) 
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.encoder_proj = nn.Linear(encoder_dim, embed_dim) if encoder_dim != embed_dim else nn.Identity()
        self.device = device

    def forwardWithTeacherForcing(self, encoder_out, encoded_captions, caption_lengths, tgt_key_padding_mask):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        caption_lengths_squeezed = caption_lengths.squeeze(1)
        decode_lengths = (caption_lengths_squeezed - 1).tolist()
        
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # [batch_size, num_pixels, encoder_dim]
        encoder_out = self.encoder_proj(encoder_out).permute(1, 0, 2)  # [num_pixels, batch_size, embed_dim]

        embeddings = self.embedding(encoded_captions) 
        embeddings = self.pos_encoding(self.dropout(embeddings))
        tgt = embeddings.permute(1, 0, 2) 

        tgt_seq_len = tgt.size(0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(self.device).bool() 
        output = tgt 
        all_cross_attentions_for_all_steps = [] 
        
        for layer_idx, layer in enumerate(self.decoder_layers):
            output, self_attn_weights, cross_attn_weights = layer(
                output, 
                encoder_out, 
                tgt_mask=tgt_mask, 
                tgt_key_padding_mask=tgt_key_padding_mask,
                output_attentions=True 
            )
            all_cross_attentions_for_all_steps.append(cross_attn_weights)

        decoder_out = output.permute(1, 0, 2) # [batch_size, max_caption_length, embed_dim]
        predictions = self.fc_out(decoder_out)
        
        stacked_cross_attentions = torch.stack(all_cross_attentions_for_all_steps, dim=0)
        alphas = stacked_cross_attentions.mean(dim=(0, 3)) 
        alphas = alphas.permute(1, 0, 2)  

        return predictions, encoded_captions, decode_lengths, alphas 


    def forwardWithoutTeacherForcing(self, encoder_out, wordMap, maxDecodeLen):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # [batch_size, num_pixels, encoder_dim]
        encoder_out = self.encoder_proj(encoder_out).permute(1, 0, 2)  # [num_pixels, batch_size, embed_dim]
        start_token_idx = wordMap['<start>']
        end_token_idx = wordMap['<end>']

        inputs = torch.full((batch_size, 1), start_token_idx, dtype=torch.long, device=self.device) 
        predictions = torch.zeros(batch_size, maxDecodeLen, self.vocab_size, device=self.device) 
        sequences = torch.zeros(batch_size, maxDecodeLen, dtype=torch.long, device=self.device) 
        alphas = torch.zeros(batch_size, maxDecodeLen, encoder_out.size(0), device=self.device) 
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device) 

        for t in range(maxDecodeLen): 
            active_indices = (~finished).nonzero(as_tuple=False).squeeze(1) 
            if len(active_indices) == 0: break  

            embeddings = self.embedding(inputs[active_indices]) 
            embeddings = self.pos_encoding(self.dropout(embeddings)) 

            tgt = embeddings.permute(1, 0, 2)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(self.device).bool() 

            current_layer_output = tgt 
            all_layer_cross_attentions_for_step = [] 

            for layer_idx, layer in enumerate(self.decoder_layers): 
                layer_output, self_attn_weights, cross_attn_weights = layer( 
                    current_layer_output, 
                    encoder_out[:, active_indices, :], 
                    tgt_mask=tgt_mask,
                    output_attentions=True 
                )
                current_layer_output = layer_output 
                all_layer_cross_attentions_for_step.append(cross_attn_weights) 

            last_token_output_sliced = current_layer_output[-1, :, :] 
            preds = self.fc_out(last_token_output_sliced) 
            predictions[active_indices, t, :] = preds 
            pred_ids = preds.argmax(dim=-1) 
            sequences[active_indices, t] = pred_ids 
            finished[active_indices] |= (pred_ids == end_token_idx) 

            new_full_inputs = torch.full((batch_size, t + 2), wordMap['<pad>'], dtype=torch.long, device=self.device)
            new_full_inputs[:, :t+1] = inputs 
            new_full_inputs[active_indices, t+1] = pred_ids
            inputs = new_full_inputs 

            stacked_cross_attentions = torch.stack(all_layer_cross_attentions_for_step, dim=0)
            cross_attn_for_current_token = stacked_cross_attentions[:, :, :, -1, :]    
            avg_cross_attention_per_token = cross_attn_for_current_token.mean(dim=(0, 2)) 
            alphas[active_indices, t, :] = avg_cross_attention_per_token

        return predictions, sequences, alphas
    

    def forward(self, teacherForcing, encoder_out, encoded_captions=None, caption_lengths=None, tgt_key_padding_mask=None, wordMap=None, maxDecodeLen=None):
        if teacherForcing is True:
            predictions, encoded_captions, decode_lengths, alphas = self.forwardWithTeacherForcing(encoder_out, encoded_captions, caption_lengths, tgt_key_padding_mask)
            return predictions, encoded_captions, decode_lengths, alphas
        elif teacherForcing is not True:
            predictions, sequences, alphas = self.forwardWithoutTeacherForcing(encoder_out, wordMap, maxDecodeLen)
            return predictions, sequences, alphas