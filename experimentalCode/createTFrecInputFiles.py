import tensorflow as tf
import json
import os
from PIL import Image
from collections import Counter
from random import choice, sample, seed
from tqdm import tqdm
import numpy as np
import io

def create_tfrec_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.
    This version uses TFRecord for storing images only and JSON for storing captions.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occurring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'} # Ensure dataset is one of the expected values

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)
    
    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])
    
        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map (A dictionary that maps each word to a unique index)
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to TFRecord file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        # Prepare TFRecord writer
        tfrecord_filename = os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.tfrecord')
        with tf.io.TFRecordWriter(tfrecord_filename) as writer:
            print(f"\nReading {split} images and captions, storing to TFRecord...\n")

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):  # tqdm(impaths) is a progress bar that shows how much of the list has been processed

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image
        
                img = Image.open(impaths[i])
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((256, 256), Image.BICUBIC)
                img = np.array(img)
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)

                # img = img.transpose(2, 0, 1)  # Convert to (C, H, W) format for PyTorch
                # assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                # buffer = io.BytesIO()
                # img.save(buffer, format='JPEG')
                # img_bytes = buffer.getvalue()

                img_bytes = tf.io.encode_jpeg(tf.convert_to_tensor(img)).numpy()

                # Create a feature dictionary for TFRecord
                features = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                }
                # Create an Example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())


                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)
            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

            print(f"TFRecord saved to {tfrecord_filename}")

create_tfrec_files(dataset='coco',
                    karpathy_json_path='./cocoDataset/caption_datasets/dataset_coco.json',
                    # karpathy_json_path='/content/drive/MyDrive/ImageCaptioning/cocoDataset/caption_datasets/dataset_coco.json',
                    image_folder='./cocoDataset/trainval2014',
                    # image_folder='/content/drive/MyDrive/ImageCaptioning/cocoDataset/trainval2014',
                    captions_per_image=5,
                    min_word_freq=5,
                    output_folder='./cocoDataset/tfrecInputFiles',
                    # output_folder='/content/drive/MyDrive/ImageCaptioning/cocoDataset/inputFiles',
                    max_len=50)
