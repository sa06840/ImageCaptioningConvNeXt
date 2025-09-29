# ImageCaptioningConvNeXt

This is a working architecture of an image captioning system using an encoder-decoder architecture. 

The encoder is a ConvNeXt in encoder.py. The two decoders that are compared are an LSTM + Attention module and a Transformer decoder present in decoder.py and transformerDecoder.py respectively. The file transformerDecoderAttVis.py contains an implementation of the transformer decoder which returns the attention weights for regularization and attention maps.

Replacing the required arguments and running createInputFiles.py will generate the required dataset files to train the models.

There is a single GPU training scrput (train.py) and a multi-GPU training script (trainMultiGPU.py). The file test.py evaluates the provided models and returns test loss and BLEU scores.
