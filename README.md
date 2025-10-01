# ImageCaptioningConvNeXt

This is a working architecture of an image captioning system using an encoder-decoder architecture. 

The encoder is a ConvNeXt in encoder.py. The two decoders that are compared are an LSTM + Attention module and a Transformer decoder present in decoder.py and transformerDecoder.py respectively. The file transformerDecoderAttVis.py contains an implementation of the transformer decoder which returns the attention weights for regularization and attention maps.

The following steps can be followed to run the code:

1. Download the 2014 MS COCO train and validation dataset linked at https://cocodataset.org/#download and the Karpathy split files linked at https://cocodataset.org/#download
2. Replace the file paths and run createInputFiles.py to get the images .HDF5 files and JSON files for captions and caption lengths
3. Connect to the HPC node and run the job script to train/test the desired model:

    #!/bin/bash
    #SBATCH -D <your working directory>            # Working directory  
    #SBATCH --job-name=image_captioning_Transformer    # Job name  
    #SBATCH --partition=gengpu              # GPU partition  
    #SBATCH --nodes=1                       # Use 1 node  
    #SBATCH --ntasks=2                      # 2 tasks total (1 per GPU). Keep this as 1 for train.py and test.py  
    #SBATCH --ntasks-per-node=2             # Run 2 task on the node. Keep this as 1 for train.py and test.py  
    #SBATCH --cpus-per-task=6               # Use 6 CPU cores  
    #SBATCH --mem=60GB                      # Allocate memory  
    #SBATCH --time=72:00:00                 # Job time limit  
    #SBATCH --gres=gpu:2                    # Request 2 GPU.  Keep this as 1 for train.py and test.py  
    #SBATCH -e results/%x_%j.e              # Standard error log  
    #SBATCH -o results/%x_%j.o              # Standard output log  
    
    source /opt/flight/etc/setup.sh  
    flight env activate gridware  # Activate the gridware environment (system environment)  
    
    module add compilers/gcc gnu  
    export https_proxy=http://hpc-proxy00.city.ac.uk:3128  
    
    srun python3 trainMultiGPU.py --port 29500 --teacherForcing     # trainMultiGPU.py can be replaced by train.py for training using a single GPU and test.py to test a desired model.  


4. Arguemnts for "srun python3 trainMultiGPU.py --port 29500 --teacherForcing": 
    - 'checkpoint', type=str, default=None, help='Path to checkpoint file'  
    - 'lstmDecoder', action='store_true', help='Use LSTM decoder instead of Transformer'  
    - 'port', type=str, default='29500', help='Master port for distributed training'  
    - 'teacherForcing', action='store_true', help='Use teacher forcing training strategy'  
    - 'startingLayer', type=int, default=7, help='Starting layer index for encoder fine-tuning encoder'  
    - 'encoderLr', type=float, default=1e-4, help='Learning rate for encoder if fine-tuning'  
    - 'embeddingName', type=str, default=None, help='Pretrained embedding name from gensim'  

    Replace/add the required arguments in the line "srun python3 trainMulticopy.py --port 29500 --teacherForcing".

    There is a single GPU training scrput (train.py) and a multi-GPU training script (trainMultiGPU.py). The file test.py evaluates the provided models and returns the test loss and BLEU scores.

6. Caption.py is run locally to generate a caption and attention map for a sample image using a trained model by providing the relevant paths.
