from utils.utils import create_input_files

# create_input_files(dataset='coco',
#                     karpathy_json_path='cocoDataset/caption_datasets/dataset_coco.json',
#                     # karpathy_json_path='/content/drive/MyDrive/dissertationImageCaptioning/cocoDataset/caption_datasets/dataset_coco.json',
#                     image_folder='cocoDataset/trainval2014',
#                     # image_folder='/content/drive/MyDrive/dissertationImageCaptioning/cocoDataset/trainval2014',
#                     captions_per_image=5,
#                     min_word_freq=5,
#                     output_folder='cocoDataset/inputFiles',
#                     # output_folder='/content/drive/MyDrive/dissertationImageCaptioning/cocoDataset/inputFiles',
#                     max_len=50)

create_input_files(dataset='flickr8k',
                    karpathy_json_path='flickr8kDataset/caption_datasets/dataset_flickr8k.json',
                    # karpathy_json_path='/content/drive/MyDrive/dissertationImageCaptioning/cocoDataset/caption_datasets/dataset_coco.json',
                    image_folder='flickr8kDataset/Flicker8k_Dataset',
                    # image_folder='/content/drive/MyDrive/dissertationImageCaptioning/cocoDataset/trainval2014',
                    captions_per_image=5,
                    min_word_freq=5,
                    output_folder='flickr8kDataset/inputFiles',
                    # output_folder='/content/drive/MyDrive/dissertationImageCaptioning/cocoDataset/inputFiles',
                    max_len=50)


