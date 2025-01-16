# Turtle Segmentation Project

This project implements semantic segmentation for turtle images using `PyTorch` and `YOLOv8-seg`, aiming to identify different parts of the turtle such as the head, flippers, and carapace. The project includes functionalities for data loading, data processing, model training, evaluation, and visualizing prediction results.

## 👥 Team: the avenger 🛡️

## Table of Contents

- [Project Background](#project-background)
- [Model](#Model)
    - [YOLO](YOLOv8-seg)
        - [Environment Setup](#environment-setup)
        - [How to Run](#how-to-run)
        - [Results](#result)
    

## Project Background

This project is focused on segmentation tasks for turtle images, aiming to identify different parts of the turtle (head, flippers, carapce). YOLOv8-seg model is used for segmentation, optimized with cross-entropy and Dice loss for better performance.

## MODEL 

The model used for this project is YOLOv8-seg, an advanced version of the YOLO (You Only Look Once) architecture tailored for segmentation tasks. YOLOv8-seg is implemented through the ultralytics library, specifically designed to balance accuracy and real-time performance in tasks like sea turtle segmentation. Below are detailed sections on the setup, model structure, and key features, focusing on the YOLOv8-seg model's capabilities and customization options for the segmentation task.

#### Environment Setup

Before running the code, please ensure the following dependencies are installed:


matplotlib==3.9.0
pycocotools==2.0.8
tqdm==4.66.4
scikit-image==0.24.0
scikit-learn==1.5.0
pandas==2.2.2
numpy==1.26.4
opencv-contrib-python==4.10.0.84
opencv-python==4.10.0.84
torch==2.3.0+cu121
torchaudio==2.3.0+cu121
torchmetrics==1.4.0.post0
torchvision==0.18.0+cu121
ultralytics==8.3.27
ultralytics-thop==2.0.10

Install these dependencies with the following command:

```bash
pip install torch torchvision segmentation-models-pytorch tqdm pandas scikit-learn matplotlib pycocotools
```
#### Data Process

Download the dataset:[Turtle dataset](https://drive.google.com/drive/folders/1dQZ52oFaJk2JWWjI7o3xMJBUsMXf2H6e?usp=drive_link)
and metadata_splits.csv file [metadata_splits.csv](https://drive.google.com/file/d/1OpMF-e619qX1qpqtxhLb_A9e1nCA-lJI/view?usp=sharing)

```
python data_process.py
```

The processed dataset including three folders: yolo_train_labels, yolo_valid_labels, yolo_test_labels. All of them have two folders: images and labels. Labels are converted to YOLO format and correspond to the images.

#### How to Run

Download the yaml file: yolo_seg_config.yaml

##### Testing

Check the dataset path and then run:
Use the pretrained model 'yolov8-seg-trained.pt'.

```
python test.py
```

##### Training

Check the dataset path and then run:
```
python train.py
```

**Class-wise mIoU scores**:
- Carapace mIoU: 0.9470
- Flipper mIoU: 0.9732
- Head mIoU: 0.9705

---

#### Result
Results of mIOU:
Class        Box(mAP50    mAP50-95)     Mask(mAP50     mAP50-95): 
all              0.79      0.622             0.785      0.6
carapace         0.456     0.356             0.46       0.372
head             0.965     0.72              0.949      0.682
flipper          0.949     0.792             0.946      0.744