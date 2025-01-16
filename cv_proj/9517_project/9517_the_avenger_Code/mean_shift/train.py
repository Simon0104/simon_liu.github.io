from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from model import create_segmentation_model
from data_process import load_and_normalize_data
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
dataset_dir = '/content/drive/My Drive/9517/extracted_data/turtles-data/data'
file_path = '/content/drive/My Drive/9517/extracted_data/turtles-data/data/metadata_splits.csv'
annotations_path = '/content/drive/My Drive/9517/extracted_data/turtles-data/data/annotations.json'

# Load data
data = pd.read_csv(file_path)
subset_data = data.sample(frac=0.30, random_state=42)
train_data, temp_data = train_test_split(subset_data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_images, train_masks = load_and_normalize_data(dataset_dir, train_data, annotations_path)
val_images, val_masks = load_and_normalize_data(dataset_dir, val_data, annotations_path)

# Initialize and compile model
model = create_segmentation_model()

# Train model
history = model.fit(train_images, train_masks, validation_data=(val_images, val_masks), epochs=10, batch_size=8)

# Save model
model.save('/content/drive/My Drive/9517/sea_turtle_segmentation.h5')
