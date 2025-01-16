import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image

# Define category color mapping
category_colors = {
    1: (0, 0, 255),    # Shell - Blue
    2: (0, 255, 0),    # Flipper - Green
    3: (255, 0, 0)     # Head - Red
}

# Function to create mask from segmentation
def create_mask_from_segmentation(segmentation, category_id):
    rle = {'counts': segmentation['counts'], 'size': segmentation['size']}
    if isinstance(rle['counts'], list):
        rle = maskUtils.frPyObjects(rle, rle['size'][0], rle['size'][1])
    binary_mask = maskUtils.decode(rle)
    binary_mask = (binary_mask * category_id).astype(np.uint8)
    return binary_mask

# Define dataset class
class TurtleDataset(Dataset):
    def __init__(self, dataset_dir, csv_data, annotations_path, img_shape=(512, 512)):
        self.dataset_dir = dataset_dir
        self.csv_data = csv_data
        self.coco = COCO(annotations_path)
        self.img_shape = img_shape
        self.image_ids = self.csv_data['id'].tolist()
        self.file_names = self.csv_data['file_name'].tolist()
        self.transform = transforms.Compose([
            transforms.Resize(img_shape),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_shape, interpolation=Image.NEAREST),
        ])
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.dataset_dir, self.file_names[idx])
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Create mask
        mask = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        for ann in annotations:
            category_id = ann['category_id']
            if category_id < 4:
                binary_mask = create_mask_from_segmentation(ann['segmentation'], category_id)
                mask = np.maximum(mask, binary_mask)
        mask[mask >= 4] = 0
        # Convert to PIL image
        mask = Image.fromarray(mask)
        
        # Apply transforms
        image = self.transform(image)
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).long().squeeze()
        
        return image, mask
