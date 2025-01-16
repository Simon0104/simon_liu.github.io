import pandas as pd
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from pycocotools import mask as maskUtils
from tqdm import tqdm
import json
import copy
import cv2

# Load the dataset
file_path = r'metadata_splits.csv'

data = pd.read_csv(file_path)

data = data[['id','file_name','split_open']]

data_types = ['train','valid','test']

datasets ={}
for data_type in data_types:
    datasets[data_type] = data[data['split_open'] == data_type]
# datasets


output_dirs = ['train', 'test', 'valid']

for dir_name in output_dirs:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

for index, row in data.iterrows():
    source_file = row['file_name']  
    split = row['split_open']  
    
    destination_folder = split  
    destination_path = os.path.join(destination_folder, os.path.basename(source_file))
    

    if os.path.exists(source_file):  
        shutil.move(source_file, destination_path)


def modify_annotations(coco_data):
    """
    Modify COCO annotations based on the logic: carapace = turtle - head - flipper.

    Parameters:
        coco_data: Original COCO data in dictionary format.

    Returns:
        Modified COCO data.
    """

    new_coco_data = copy.deepcopy(coco_data)

    # Update categories, keeping only carapace, head, and flipper
    new_categories = [
        {'id': 0, 'name': 'carapace'},
        {'id': 1, 'name': 'head'},
        {'id': 2, 'name': 'flipper'}
    ]
    category_name_to_id = {cat['name']: cat['id'] for cat in new_categories}
    new_coco_data['categories'] = new_categories

    # Create a mapping from original category IDs to names
    original_category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Create a new list of annotations
    new_annotations = []
    annotation_id = 1  # Reset annotation ID

    # Process images one by one
    for image_info in tqdm(new_coco_data['images'], desc="Processing images"):
        image_id = image_info['id']
        height = image_info['height']
        width = image_info['width']

        # Get all original annotations for this image
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]

        # Initialize masks
        turtle_mask = np.zeros((height, width), dtype=np.uint8)
        head_masks = []
        flipper_masks = []

        # Collect masks for each part and combine turtle masks
        for ann in annotations:
            category_id = ann['category_id']
            category_name = original_category_id_to_name.get(category_id, 'N/A')
            mask = get_mask_from_annotation(ann, height, width)

            # Process masks based on category name
            if category_name == 'turtle':
                turtle_mask = np.maximum(turtle_mask, mask)  # Combine all turtle masks
            elif category_name == 'head':
                head_masks.append(mask)
            elif category_name == 'flipper':
                flipper_masks.append(mask)
            else:
                continue  # Ignore other categories

        # Generate combined mask for head and flipper
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        for head_mask in head_masks:
            combined_mask = np.maximum(combined_mask, head_mask)
        for flipper_mask in flipper_masks:
            combined_mask = np.maximum(combined_mask, flipper_mask)

        # Subtract head and flipper regions from turtle_mask to get carapace
        carapace_mask = np.where(turtle_mask - combined_mask < 0, 0, turtle_mask - combined_mask)

        # Create a new annotation for carapace if there is content in carapace_mask
        if carapace_mask.any():
            rle = maskUtils.encode(np.asfortranarray(carapace_mask))
            rle['counts'] = rle['counts'].decode('ascii')  # 'counts' must be a string
            bbox = maskUtils.toBbox(rle).tolist()
            new_annotations.append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': category_name_to_id['carapace'],
                'segmentation': rle,
                'area': int(carapace_mask.sum()),
                'bbox': bbox,
                'iscrowd': 0
            })
            annotation_id += 1

        # Create individual annotations for each head and flipper
        for mask, category_name in zip(head_masks, ['head'] * len(head_masks)):
            rle = maskUtils.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('ascii')
            bbox = maskUtils.toBbox(rle).tolist()
            new_annotations.append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': category_name_to_id[category_name],
                'segmentation': rle,
                'area': int(mask.sum()),
                'bbox': bbox,
                'iscrowd': 0
            })
            annotation_id += 1

        for mask, category_name in zip(flipper_masks, ['flipper'] * len(flipper_masks)):
            rle = maskUtils.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('ascii')
            bbox = maskUtils.toBbox(rle).tolist()
            new_annotations.append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': category_name_to_id[category_name],
                'segmentation': rle,
                'area': int(mask.sum()),
                'bbox': bbox,
                'iscrowd': 0
            })
            annotation_id += 1

    # Update annotations
    new_coco_data['annotations'] = new_annotations

    return new_coco_data

def get_mask_from_annotation(ann, height, width):
    segmentation = ann['segmentation']
    if isinstance(segmentation, list):
        rles = maskUtils.frPyObjects(segmentation, height, width)
        rle = maskUtils.merge(rles)
        mask = maskUtils.decode(rle)
    elif isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle['counts'], list):
            rle = maskUtils.frPyObjects([rle], height, width)[0]
        mask = maskUtils.decode(rle)
    else:
        print(f"Unknown segmentation format, Annotation ID: {ann['id']}")
        mask = np.zeros((height, width), dtype=np.uint8)
    return mask

# Load original data
with open('train.json', 'r', encoding='utf-8') as f1:
    train_data = json.load(f1)
with open('valid.json', 'r', encoding='utf-8') as f2:
    valid_data = json.load(f2)
with open('test.json', 'r', encoding='utf-8') as f3:
    test_data = json.load(f3)

# Modify annotations
modified_train_data = modify_annotations(train_data)
modified_valid_data = modify_annotations(valid_data)
modified_test_data = modify_annotations(test_data)

# Save modified data
with open('modified_train_annotations.json', 'w', encoding='utf-8') as f:
    json.dump(modified_train_data, f)

with open('modified_valid_annotations.json', 'w', encoding='utf-8') as f:
    json.dump(modified_valid_data, f)

with open('modified_test_annotations.json', 'w', encoding='utf-8') as f:
    json.dump(modified_test_data, f)


# convert coco(json) to yolo(txt)
def rle_to_polygon(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:  # At least 3 points are needed to form a polygon
            polygon = contour.flatten().tolist()
            polygons.append(polygon)
    return polygons

def convert_coco_to_yolo_segmentation(coco_data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a mapping from image_id to file information
    image_id_to_info = {image['id']: image for image in coco_data['images']}
    
    # Clear existing .txt files to avoid duplicate annotations
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if file_path.endswith(".txt"):
            os.remove(file_path)
    
    # Temporarily store masks for each image and category
    image_masks = {}

    # Create a mapping from category_id to YOLO ID
    category_id_to_yolo_id = {}
    categories = coco_data['categories']
    for idx, category in enumerate(categories):
        category_id_to_yolo_id[category['id']] = idx  # YOLO IDs start from 0

    # Accumulate masks by image and category
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        segmentation = annotation['segmentation']
        image_info = image_id_to_info[image_id]
        height = image_info['height']
        width = image_info['width']
        
        # Initialize mask dictionary for this image if it doesn't exist
        if image_id not in image_masks:
            image_masks[image_id] = {}
        
        # Initialize mask for this category if it doesn't exist
        if category_id not in image_masks[image_id]:
            image_masks[image_id][category_id] = np.zeros((height, width), dtype=np.uint8)
        
        # Process RLE
        if isinstance(segmentation, dict) and 'counts' in segmentation:
            rle = segmentation
            if isinstance(rle['counts'], list):
                # Convert uncompressed RLE to compressed RLE
                rle = maskUtils.frPyObjects([rle], height, width)[0]
        else:
            print(f"Unknown segmentation format, Annotation ID: {annotation['id']}")
            continue
        
        # Decode RLE to binary mask
        mask = maskUtils.decode(rle)
        
        # Accumulate mask
        image_masks[image_id][category_id] = np.maximum(image_masks[image_id][category_id], mask)
    
    # Convert accumulated masks to polygons and write in YOLO format
    for image_id, category_masks in image_masks.items():
        image_info = image_id_to_info[image_id]
        file_basename = os.path.splitext(image_info['file_name'])[0]
        height = image_info['height']
        width = image_info['width']
        
        annotations_list = []
        for category_id, mask in category_masks.items():
            yolo_category_id = category_id_to_yolo_id[category_id]
            # Get polygons from the merged mask
            polygons = rle_to_polygon(mask)
            for polygon in polygons:
                # Normalize polygon coordinates
                normalized_coords = []
                for i in range(0, len(polygon), 2):
                    x = polygon[i] / width
                    y = polygon[i + 1] / height
                    normalized_coords.extend([f"{x:.6f}", f"{y:.6f}"])
                polygon_str = f"{yolo_category_id} " + " ".join(normalized_coords)
                annotations_list.append(f"{polygon_str}\n")
        
        # Write label file
        label_file = os.path.join(output_dir, f"{file_basename}.txt")
        with open(label_file, 'w') as file:
            file.writelines(annotations_list)

# Load JSON files
with open('modified_train_annotations.json', 'r', encoding='utf-8') as f:
    modified_train_data = json.load(f)
with open('modified_valid_annotations.json', 'r', encoding='utf-8') as f:
    modified_valid_data = json.load(f)
with open('modified_test_annotations.json', 'r', encoding='utf-8') as f:
    modified_test_data = json.load(f)

# Convert to YOLO format
convert_coco_to_yolo_segmentation(modified_train_data, 'yolo_train_labels/labels')
convert_coco_to_yolo_segmentation(modified_valid_data, 'yolo_valid_labels/labels')
convert_coco_to_yolo_segmentation(modified_test_data, 'yolo_test_labels/labels')



def clean_label_files(directory):
    """
    Traverse all .txt files in the directory, and for each file, keep only the last class 0 annotation,
    deleting the others without changing class 1 and class 2 annotations.
    Parameters:
        directory (str): The directory containing the .txt label files.
    """
    # Traverse all .txt files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            
            # Read file content
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Record the line numbers of class 0 annotations
            class_0_indices = []
            for idx, line in enumerate(lines):
                stripped_line = line.strip()
                if not stripped_line:
                    continue  # Skip empty lines
                tokens = stripped_line.split()
                if tokens[0] == '0':
                    class_0_indices.append(idx)
            
            # If there are multiple class 0 annotations, keep the last one and delete the others
            indices_to_remove = class_0_indices[:-1]  # Remove all but the last one

            # Build a new list of lines, keeping the order
            new_lines = []
            for idx, line in enumerate(lines):
                if idx not in indices_to_remove:
                    new_lines.append(line)
                else:
                    print(f"Removed line {idx+1} in {filename}: {line.strip()}")
            
            # Write back to the file
            with open(file_path, 'w') as file:
                file.writelines(new_lines)
            
        
# Specify the label files directory
label_directory_train = 'yolo_train_labels/labels'
label_directory_valid = 'yolo_train_labels/labels'
label_directory_test = 'yolo_train_labels/labels'
clean_label_files(label_directory_train)
clean_label_files(label_directory_valid)
clean_label_files(label_directory_test)