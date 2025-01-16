Here's the content in markdown format:

---

# Sea Turtle Segmentation Project

This project aims to perform semantic segmentation on turtle images using a deep learning model with TensorFlow. The goal is to identify different parts of a turtle, including the head, flippers, and shell, based on COCO-style annotated data.

## 👥 Team: Turtle Squad 🐢

## Table of Contents
- [Project Background](#project-background)
- [Environment Setup](#environment-setup)
- [Data Processing](#data-processing)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)
- [Results](#results)

## Project Background

The purpose of this project is to perform semantic segmentation on turtle images, identifying various parts of the turtle (head, flippers, and shell). The project defines a multi-class segmentation mask and uses a deep learning model to perform the segmentation.

## Environment Setup

Before running the code, ensure the following libraries are installed:

```bash
pip install tensorflow pycocotools numpy pandas scikit-learn opencv-python
```

## Data Processing

First, mount Google Drive and set the data file paths:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Ensure the following files are in the specified Google Drive path:
- **metadata_splits.csv**: A metadata file for data splits.
- **annotations.json**: Annotation data in COCO format.

Run the following command to process the data:

```bash
python data_process.py
```

This process generates images and segmentation masks used for model training and testing.

## Training the Model

Run `train.py` in Google Colab to train the model. The script uses the training and validation splits specified in the dataset:

```bash
python train.py
```

After training, the model weights will be saved in the specified Google Drive path for future loading and testing.

## Testing the Model

Run `test.py` to evaluate the model’s performance. The script will load the validation data and print the Intersection over Union (IoU) score for each class:

```bash
python test.py
```

## Results

The model achieved the following IoU scores for each class:

- **Background IoU**: 0.2408
- **Head IoU**: 0.0678
- **Flippers IoU**: 0.0270
- **Shell IoU**: 0.0175

--- 

