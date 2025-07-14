from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

HF_MODEL_NAME = "t5-small"

class HFTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, device, wordMap, encoder_dim=1024):
        super().__init__()

        self.encoder_dim = encoder_dim 
        self.vocab_size = vocab_size
        self.wordMap = wordMap
        self.device = device

        config = AutoConfig.from_pretrained(HF_MODEL_NAME)
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME, config=config)

        if encoder_dim != config.d_model:
            self.encoder_proj = nn.Linear(encoder_dim, config.d_model)
        else:
            self.encoder_proj = nn.Identity()
        
        self.config = config

        # creating mapping
        try:
            t5_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
        except Exception as e:
            raise ValueError(f"Could not load tokenizer {HF_MODEL_NAME}. It's required to build the vocabulary mapping. Error: {e}")
        
        self.t5_pad_token_id = t5_tokenizer.pad_token_id if t5_tokenizer.pad_token_id is not None else config.pad_token_id if hasattr(config, 'pad_token_id') else 0
        self.t5_eos_token_id = t5_tokenizer.eos_token_id if t5_tokenizer.eos_token_id is not None else config.eos_token_id if hasattr(config, 'eos_token_id') else 1
        self.t5_unk_token_id = t5_tokenizer.unk_token_id if t5_tokenizer.unk_token_id is not None else config.unk_token_id if hasattr(config, 'unk_token_id') else 2 # Fallback if tokenizer doesn't have it
        self.t5_decoder_start_token_id = config.decoder_start_token_id
        if self.t5_decoder_start_token_id is None:
            self.t5_decoder_start_token_id = self.t5_pad_token_id
        
        custom_id_to_word_string = {idx: word for word, idx in self.wordMap.items()}
        self.custom_id_to_t5_id_tensor = torch.full(
            (self.vocab_size,), # Size of YOUR custom vocabulary
            fill_value=self.t5_unk_token_id, 
            dtype=torch.long,
            device=device)

        for custom_id in range(self.vocab_size): 
            word_string = custom_id_to_word_string.get(custom_id)
            if word_string is None:
                continue
            if word_string == '<start>':
                self.custom_id_to_t5_id_tensor[custom_id] = self.t5_decoder_start_token_id
            elif word_string == '<end>':
                self.custom_id_to_t5_id_tensor[custom_id] = self.t5_eos_token_id
            elif word_string == '<pad>':
                self.custom_id_to_t5_id_tensor[custom_id] = self.t5_pad_token_id
            elif word_string == '<unk>': 
                self.custom_id_to_t5_id_tensor[custom_id] = self.t5_unk_token_id
            else:
                if word_string in t5_tokenizer.vocab:
                    t5_id_for_word = t5_tokenizer.vocab[word_string]
                    self.custom_id_to_t5_id_tensor[custom_id] = t5_id_for_word
        
        self.register_buffer('customToT5', self.custom_id_to_t5_id_tensor)

        self.t5_id_to_custom_id_tensor = torch.full(
            (self.config.vocab_size,), # Size of T5's full vocabulary
            fill_value=self.wordMap['<unk>'], # Default to your custom UNK ID
            dtype=torch.long,
            device=device)
        
        if t5_tokenizer: 
            for t5_id, word_string in t5_tokenizer.vocab.items():
                if word_string in self.wordMap:
                    custom_id = self.wordMap[word_string]
                    if custom_id < self.vocab_size:
                        self.t5_id_to_custom_id_tensor[t5_id] = custom_id
            
            self.t5_id_to_custom_id_tensor[self.t5_decoder_start_token_id] = self.wordMap['<start>']
            self.t5_id_to_custom_id_tensor[self.t5_eos_token_id] = self.wordMap['<end>']
            self.t5_id_to_custom_id_tensor[self.t5_pad_token_id] = self.wordMap['<pad>']
            self.t5_id_to_custom_id_tensor[self.t5_unk_token_id] = self.wordMap['<unk>']

        self.register_buffer('t5ToCustom', self.t5_id_to_custom_id_tensor)


    def forwardWithTeacherForcing(self, encoder_out, encoded_captions, caption_lengths=None):
        batch_size = encoder_out.size(0)
        caption_lengths = caption_lengths.squeeze(1)
        decode_lengths = (caption_lengths - 1).tolist()

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)
        encoder_out = self.encoder_proj(encoder_out) # (batch_size, num_pixels, d_model)
        encoder_attention_mask = torch.ones(encoder_out.shape[0], encoder_out.shape[1], device=self.device)

        remapped_encoded_captions = self.customToT5[encoded_captions]

        decoder_outputs = self.t5_model.decoder(
            input_ids=remapped_encoded_captions,
            encoder_hidden_states=encoder_out, # This is the memory from your image encoder
            encoder_attention_mask=encoder_attention_mask,
            # T5's decoder automatically creates a causal attention mask for decoder_input_ids
            # T5 relies on padding_token_id in input_ids for masking, so tgt_key_padding_mask is not directly passed.
            return_dict=True # Always return as a dictionary for easier access
        )

        decoder_hidden_states = decoder_outputs.last_hidden_state # (batch_size, max_caption_length, d_model)
        predictions = self.t5_model.lm_head(decoder_hidden_states)
        return predictions, remapped_encoded_captions, decode_lengths
    
    def forwardWithoutTeacherForcingTrain(self, encoder_out, maxDecodeLen):
        # Inference with Hugging Face Transformer Decoder using a greedy generation loop.
        # This handles K-V caching internally via the decoder's forward pass.
        batch_size = encoder_out.size(0)

         # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)
        encoder_out = self.encoder_proj(encoder_out)  # (batch_size, num_pixels, d_model)
        encoder_attention_mask = torch.ones(encoder_out.shape[0], encoder_out.shape[1], device=self.device)

        start_token_id = self.wordMap['<start>']
        end_token_id = self.wordMap['<end>']
        inputs = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=self.device)

        predictions = torch.zeros(batch_size, maxDecodeLen, self.config.vocab_size, device=self.device)
        sequences = torch.zeros(batch_size, maxDecodeLen, dtype=torch.long, device=self.device) # Will store IDs in YOUR custom vocab space
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        past_key_values = None 

        for t in range(maxDecodeLen):
            if finished.all(): # Check if all elements in finished_sequences are True
                break 

            inputsT5 = self.customToT5[inputs]
            decoder_outputs = self.t5_model.decoder(
                input_ids=inputsT5, # Pass T5's IDs to the decoder
                encoder_hidden_states=encoder_out,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values, # Pass cache from previous step
                use_cache=True, # Crucial: tells decoder to return cache for next step
                return_dict=True
            )

            preds = self.t5_model.lm_head(decoder_outputs.last_hidden_state).squeeze(1)
            predictions[:, t, :] = preds

            predicted_ids_t5 = preds.argmax(dim=-1)
            predicted_ids_custom = self.t5ToCustom[predicted_ids_t5]
            sequences[:, t] = predicted_ids_custom
            finished |= (predicted_ids_custom == end_token_id)
            inputs = predicted_ids_custom.unsqueeze(1)
            past_key_values = decoder_outputs.past_key_values 
        
        return predictions, sequences
        
    def forwardWithoutTeacherForcingInference(self, encoder_out, maxDecodeLen):
        # Inference with Hugging Face Transformer Decoder using a greedy generation loop.
        # This handles K-V caching internally via the decoder's forward pass.
        batch_size = encoder_out.size(0)

         # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)
        encoder_out = self.encoder_proj(encoder_out)  # (batch_size, num_pixels, d_model)
        encoder_attention_mask = torch.ones(encoder_out.shape[0], encoder_out.shape[1], device=self.device)

        # 2. Prepare encoder_outputs object for model.generate()
        # model.generate() expects encoder_outputs to be an object (like BaseModelOutput)
        encoder_outputs_for_generate = BaseModelOutput(
            last_hidden_state=encoder_out
        )

        generation_config = GenerationConfig(
            max_length=maxDecodeLen,
            num_beams=1,           # For greedy search
            do_sample=False,       # For greedy search
            decoder_start_token_id=self.t5_decoder_start_token_id,
            eos_token_id=self.t5_eos_token_id,
            pad_token_id=self.t5_pad_token_id,
            output_scores=True,    # CRUCIAL: Get logits for each step
            return_dict_in_generate=True, # Return as a dictionary (GenerationOutput object)
        )

        generation_outputs = self.t5_model.generate( # type: ignore (for linting)
            encoder_outputs=encoder_outputs_for_generate, 
            generation_config=generation_config, 
            encoder_attention_mask=encoder_attention_mask
        )

        reconstructed_predictions_logits = torch.stack(generation_outputs.scores, dim=0)
        reconstructed_predictions_logits = reconstructed_predictions_logits.permute(1, 0, 2)   #(batch_size, actual_generated_length, vocab_size_t5)
        all_sequences_ids_t5 = generation_outputs.sequences 
        final_sequences_custom_ids = self.t5ToCustom[all_sequences_ids_t5] 

        padded_predictions_logits = torch.zeros(
            batch_size, maxDecodeLen, self.config.vocab_size, 
            dtype=reconstructed_predictions_logits.dtype, 
            device=self.device
        )
        padded_sequences = torch.full(
            (batch_size, maxDecodeLen), 
            fill_value=self.wordMap['<pad>'], # Use your custom PAD ID for padding
            dtype=torch.long, 
            device=self.device
        )
        padded_predictions_logits[:, :reconstructed_predictions_logits.size(1), :] = reconstructed_predictions_logits
        padded_sequences[:, :final_sequences_custom_ids.size(1)] = final_sequences_custom_ids

        return padded_predictions_logits, padded_sequences


    def forward(self, teacherForcing, encoder_out, state=None, encoded_captions=None, caption_lengths=None, maxDecodeLen=None):
        if teacherForcing is True:
            predictions, remapped_encoded_captions, decode_lengths = self.forwardWithTeacherForcing(encoder_out, encoded_captions, caption_lengths)
            return predictions, remapped_encoded_captions, decode_lengths
        elif teacherForcing is not True:
            if state == 'train':
                predictions, sequences = self.forwardWithoutTeacherForcingTrain(encoder_out, maxDecodeLen)
                return predictions, sequences
            if state == 'inference':
                predictions, sequences = self.forwardWithoutTeacherForcingInference(encoder_out, maxDecodeLen)
                return predictions, sequences
