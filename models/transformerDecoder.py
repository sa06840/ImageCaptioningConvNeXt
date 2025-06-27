import torch.nn as nn
import math
import torch

device = torch.device("mps")

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, embed_dim)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, decoder_dim, vocab_size, maxLen, dropout=0.5, encoder_dim=1024, num_heads=8, num_layers=6):
        super(TransformerDecoder, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 2. Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, maxLen)
        # 3. Transformer decoder
        self.dropout = nn.Dropout(p=self.dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=decoder_dim, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        # 4. Linear projection to vocab
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        # 5. Optional projection for encoder output
        self.encoder_proj = nn.Linear(encoder_dim, embed_dim) if encoder_dim != embed_dim else nn.Identity()

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        # encoder_out: (batch_size, enc_image_size, enc_image_size, encoder_dim)
        # encoded_captions: (batch_size, max_caption_length)
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        caption_lengths = caption_lengths.squeeze(1)
        decode_lengths = (caption_lengths - 1).tolist()
        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)

        # Project encoder output to decoder_dim and reshape to [num_pixels, batch_size, decoder_dim]
        encoder_out = self.encoder_proj(encoder_out).permute(1, 0, 2)  # [num_pixels, batch_size, decoder_dim]

        # Embed captions and apply positional encoding
        embeddings = self.embedding(encoded_captions)  # [batch_size, max_caption_length, embed_dim]
        embeddings = self.pos_encoding(self.dropout(embeddings))
        tgt = embeddings.permute(1, 0, 2)  # [max_len, batch_size, embed_dim]

        # Generate target mask for masked self-attention
        tgt_seq_len = tgt.size(0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)  # [max_caption_length, max_caption_length]

        # Transformer decoding
        decoder_out = self.transformer_decoder(tgt, encoder_out, tgt_mask=tgt_mask)  # [max_len, batch_size, decoder_dim]
        decoder_out = decoder_out.permute(1, 0, 2)  # [batch_size, max_caption_length, decoder_dim]
        # Final prediction scores
        predictions = self.fc_out(decoder_out)  # [batch_size, max_caption_length, vocab_size]

        return predictions, encoded_captions, decode_lengths
