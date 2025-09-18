import h5py
import json
import numpy as np
import matplotlib.pyplot as plt

# Define file paths
image_file = 'cocoDataset/inputFiles/TRAIN_IMAGES_coco_5_cap_per_img_5_min_word_freq.hdf5'  # Path to your HDF5 file
captions_file = 'cocoDataset/inputFiles/TRAIN_CAPTIONS_coco_5_cap_per_img_5_min_word_freq.json'  # Path to your captions JSON
word_map_file = 'cocoDataset/inputFiles/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # Path to your word map JSON

# Load word map
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
    
# Reverse the word map (to convert indices back to words)
word_map_inv = {v: k for k, v in word_map.items()}

# Open HDF5 file
with h5py.File(image_file, 'r') as h:
    # Load image data
    images = h['images'][:]
    captions_per_image = h.attrs['captions_per_image']
    
    index = 0 
    image = images[index]
    captions = json.load(open(captions_file, 'r'))[index * captions_per_image:(index + 1) * captions_per_image]
    
    image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)
    plt.imshow(image)
    plt.axis('off')
    
    decoded_captions = []
    for caption in captions:
        decoded_caption = ' '.join([word_map_inv.get(word_id, '<unk>') for word_id in caption])
        decoded_captions.append(decoded_caption)
    
    print("Captions for this image:")
    for i, caption in enumerate(decoded_captions):
        print(f"Caption {i + 1}: {caption}")
    
    plt.show()