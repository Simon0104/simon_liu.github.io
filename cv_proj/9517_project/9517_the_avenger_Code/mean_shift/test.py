import numpy as np
from data_process import calculate_iou_per_class, load_and_normalize_data
from tensorflow.keras.models import load_model

# Paths
dataset_dir = '/content/drive/My Drive/9517/extracted_data/turtles-data/data'
annotations_path = '/content/drive/My Drive/9517/extracted_data/turtles-data/data/annotations.json'

# Load model
model = load_model('/content/drive/My Drive/9517/sea_turtle_segmentation.h5')

# Load test data
test_images, test_masks = load_and_normalize_data(dataset_dir, test_data, annotations_path)

# Generate predictions
predictions = model.predict(test_images)
predictions = np.argmax(predictions, axis=-1)
test_masks_true = np.argmax(test_masks, axis=-1)

# Calculate IoU
ious = calculate_iou_per_class(predictions, test_masks_true, num_classes=4)

# Display IoU for each class
categories = ["background", "head", "flippers", "carapace"]
for i, iou in enumerate(ious):
    print(f"{categories[i]} IoU: {iou:.12f}")
