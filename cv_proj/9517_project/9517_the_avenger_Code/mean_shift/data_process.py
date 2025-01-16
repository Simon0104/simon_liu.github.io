import os
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import UnidentifiedImageError
import tensorflow as tf

# Define category color mapping
category_colors = {
    1: (0, 0, 255),    # Shell - Blue
    2: (0, 255, 0),    # Fin - Green
    3: (255, 0, 0)     # Head - Red
}

def create_mask_from_segmentation(segmentation, category_id):
    rle = {'counts': segmentation['counts'], 'size': segmentation['size']}
    if isinstance(rle['counts'], list):
        rle = maskUtils.frPyObjects(rle, rle['size'][0], rle['size'][1])
    binary_mask = maskUtils.decode(rle)
    binary_mask = (binary_mask * category_id).astype(np.uint8)
    return binary_mask

def shiftmean(image):
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    std = np.std(image, axis=(0, 1), keepdims=True)
    normalized_image = (image - mean) / (std + 1e-5)
    return normalized_image, mean, std

def load_and_normalize_data(dataset_dir, csv_data, annotations_path, img_shape=(512, 512)):
    images, masks = [], []
    coco = COCO(annotations_path)
    for _, row in csv_data.iterrows():
        image_path = os.path.join(dataset_dir, row['file_name'])
        try:
            image = np.array(tf.keras.preprocessing.image.load_img(image_path, target_size=img_shape)) / 255.0
            normalized_image, mean, std = shiftmean(image)
            images.append(normalized_image)
        except (UnidentifiedImageError, FileNotFoundError):
            continue

        image_id = row['id']
        annotation_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(annotation_ids)
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        for ann in annotations:
            category_id = ann['category_id']
            if category_id in category_colors:
                binary_mask = create_mask_from_segmentation(ann['segmentation'], category_id)
                binary_mask = tf.image.resize(np.expand_dims(binary_mask, axis=-1), img_shape[:2]).numpy()
                mask = np.maximum(mask, binary_mask[..., 0])

        masks.append(mask)
    
    masks = np.array(masks)
    masks[masks >= 4] = 0
    masks = tf.keras.utils.to_categorical(masks, num_classes=4)
    return np.array(images), masks

def calculate_iou_per_class(predictions, ground_truth, num_classes):
    ious = []
    for class_id in range(num_classes):
        intersection = np.logical_and(predictions == class_id, ground_truth == class_id).sum()
        union = np.logical_or(predictions == class_id, ground_truth == class_id).sum()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)
    return ious
