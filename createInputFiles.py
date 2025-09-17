from utils.utils import create_input_files

# This script uses create_input_files function from utils/utils.py to process the dataset and generate the
# input files for training, validation, and testing. It is adapted from the original codebase of the 
# study (Ramos et al., 2024).

create_input_files(dataset='coco',
                    karpathy_json_path='cocoDataset/caption_datasets/dataset_coco.json',
                    image_folder='cocoDataset/trainval2014',
                    captions_per_image=5,
                    min_word_freq=5,
                    output_folder='cocoDataset/inputFiles',
                    max_len=50)
