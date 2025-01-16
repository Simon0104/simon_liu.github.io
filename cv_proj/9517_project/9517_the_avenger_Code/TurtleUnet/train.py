import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import TurtleDataset  
from model import get_model   
from loss import criterion
import numpy as np    

# Data paths
image_root_folder = 'archive/turtles-data/data'     # Set images path
file_path = 'archive/turtles-data/data/metadata_splits.csv' # Specify the path to the CSV file containing the data segmentation information
annotations_path = os.path.join(image_root_folder, 'annotations.json')

# Load and split data
data = pd.read_csv(file_path)
subset_data = data.sample(frac=0.3, random_state=42) # Set the proportion of the randomly selected sample

# Split the data according to the SeaTurtleID2022 dataset definition
train_data = subset_data[subset_data['split_open'] == 'train']  # Use 'train' in split_open as training set
val_data = subset_data[subset_data['split_open'] == 'test']     # Use 'test' in split_open as validation set


# Create datasets and data loaders
train_dataset = TurtleDataset(image_root_folder, train_data, annotations_path)
val_dataset = TurtleDataset(image_root_folder, val_data, annotations_path)

batch_size = 6
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Initialize model, device, and optimizer
num_classes = 4
model = get_model(num_classes=num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Move class weights to device
from loss import class_weights
class_weights = class_weights.to(device)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Function to calculate IoU for each class
def compute_iou(preds, labels, num_classes):
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)
    for cls in range(1, num_classes):  # Skip background class 0
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return ious

# Training loop
epochs = 100  # Set your traning loop 
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    for images, masks in train_loader_tqdm:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=loss.item())
    
    print(f"Epoch {epoch+1} - Training Loss: {running_loss / len(train_loader)}")

    # Validation with mIoU calculation
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_masks = []
    total_correct = 0  
    total_pixels = 0   

    val_loader_tqdm = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
    with torch.no_grad():
        for images, masks in val_loader_tqdm:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            val_loader_tqdm.set_postfix(loss=loss.item())
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_masks.append(masks.cpu())
            
            # Calculate accuracy
            total_correct += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)

    val_loss /= len(val_loader)
    accuracy = total_correct / total_pixels  
    print(f"Epoch {epoch+1} - Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Calculate mIoU for validation
    all_preds = torch.cat(all_preds)
    all_masks = torch.cat(all_masks)
    ious = compute_iou(all_preds, all_masks, num_classes)
    mean_iou = np.nanmean(ious)
    print(f"Validation Mean IoU (mIoU): {mean_iou}")

# Save the trained model
torch.save(model.state_dict(), "final_model.pth")
print("Model training complete and saved as 'final_model.pth'.")