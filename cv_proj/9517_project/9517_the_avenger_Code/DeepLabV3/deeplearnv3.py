import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tensorflow.keras.metrics import MeanIoU
from PIL import UnidentifiedImageError

# Define category color mapping
category_colors = {
    1: (0, 0, 255),    # Shell - Blue
    2: (0, 255, 0),    # Fin - Green
    3: (255, 0, 0)     # Head - Red
}

# Create mask function
def create_mask_from_segmentation(segmentation, category_id):
    rle = {'counts': segmentation['counts'], 'size': segmentation['size']}
    if isinstance(rle['counts'], list):
        rle = maskUtils.frPyObjects(rle, rle['size'][0], rle['size'][1])
    binary_mask = maskUtils.decode(rle)
    binary_mask = (binary_mask * category_id).astype(np.uint8)
    return binary_mask

# Define DeepLabV3 model
def deeplabv3_model(input_shape=(512, 512, 3), num_classes=4):
    base_model = tf.keras.applications.DenseNet121(input_shape=input_shape, include_top=False)
    x = base_model.output
    x = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    return model

# Data loading and augmentation function
def load_data(dataset_dir, csv_data, annotations_path, img_shape=(512, 512)):
    images, masks = [], []
    coco = COCO(annotations_path)
    for _, row in csv_data.iterrows():
        image_path = os.path.join(dataset_dir, row['file_name'])
        
        try:
            # Attempt to load the image
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=img_shape)
            image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Skipping file {image_path} due to error: {e}")
            continue  # Skip this image and move to the next

        image_id = row['id']
        annotation_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(annotation_ids)
        mask = np.zeros(img_shape[:2], dtype=np.uint8)

        for ann in annotations:
            category_id = ann['category_id']
            if category_id < 4:
                binary_mask = create_mask_from_segmentation(ann['segmentation'], category_id)
                binary_mask = tf.image.resize(np.expand_dims(binary_mask, axis=-1), img_shape[:2]).numpy()
                mask = np.maximum(mask, binary_mask[..., 0])

        images.append(image)
        masks.append(mask)

    masks = np.array(masks)
    masks[masks >= 4] = 0
    masks = tf.keras.utils.to_categorical(masks, num_classes=4)
    return np.array(images), masks

# Custom focal loss with class weights
def focal_loss_with_class_weights(class_weights, gamma=2.):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        weights = tf.reduce_sum(class_weights * y_true, axis=-1)
        weights = tf.expand_dims(weights, axis=-1)
        fl = -weights * tf.pow(1 - y_pred, gamma) * y_true * tf.math.log(y_pred)
        return tf.reduce_mean(fl)
    return focal_loss_fixed

# Calculate IoU for each class
def calculate_iou_per_class(predictions, ground_truth, num_classes):
    ious = []
    for class_id in range(num_classes):
        intersection = np.logical_and(predictions == class_id, ground_truth == class_id).sum()
        union = np.logical_or(predictions == class_id, ground_truth == class_id).sum()
        if union == 0:
            iou = float('nan')  # If there's no ground truth for this class
        else:
            iou = intersection / union
        ious.append(iou)
    return ious

# Define class weights
class_weights = np.array([0.1, 1.0, 2.0, 3.0])

# Load data paths
image_root_folder = '/content/drive/My Drive/9517/extracted_data/turtles-data/data'
file_path = os.path.join(image_root_folder, 'metadata_splits.csv')
annotations_path = os.path.join(image_root_folder, 'annotations.json')

# Read CSV file and sample 30% of the entire dataset
data = pd.read_csv(file_path)
subset_data = data.sample(frac=0.30, random_state=42)
train_data, temp_data = train_test_split(subset_data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Load training and validation data
train_images, train_masks = load_data(image_root_folder, train_data, annotations_path, img_shape=(512, 512))
val_images, val_masks = load_data(image_root_folder, val_data, annotations_path, img_shape=(512, 512))

# Create and compile DeepLabV3 model
model = deeplabv3_model(input_shape=(512, 512, 3), num_classes=4)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss=focal_loss_with_class_weights(class_weights, gamma=3.0), 
              metrics=['accuracy', MeanIoU(num_classes=4)])

# Train model
model.fit(train_images, train_masks, validation_data=(val_images, val_masks), epochs=20, batch_size=2)

# Generate predictions for the validation set
val_predictions = model.predict(val_images)
val_predictions = np.argmax(val_predictions, axis=-1)  # Get class labels from predictions
val_masks_true = np.argmax(val_masks, axis=-1)  # Get class labels from true masks

# Calculate IoU for each class
num_classes = 4
ious = calculate_iou_per_class(val_predictions, val_masks_true, num_classes)

# Print IoU for each category
categories = ["background", "head", "flippers", "carapace"]
for i, iou in enumerate(ious):
    print(f"{categories[i]} IoU: {iou}")
