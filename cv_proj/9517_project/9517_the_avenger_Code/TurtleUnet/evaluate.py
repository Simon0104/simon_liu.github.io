import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from dataset import TurtleDataset  
from model import get_model       
from loss import criterion    

# Data paths
image_root_folder = 'archive/turtles-data/data'     # Set images path
file_path = 'archive/turtles-data/data/metadata_splits.csv' # Specify the path to the CSV file containing the data segmentation information
annotations_path = os.path.join(image_root_folder, 'annotations.json')

# Load test data
data = pd.read_csv(file_path)
subset_data = data.sample(frac=0.3, random_state=42) # Set the proportion of the randomly selected sample
test_data = subset_data[subset_data['split_closed'] == 'test']  # Use 'test' in split_closed as test set
test_dataset = TurtleDataset(image_root_folder, test_data, annotations_path)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
# Load model
num_classes = 4  # Including background
model = get_model(num_classes=num_classes)
model.load_state_dict(torch.load("final_model.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

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
    return ious  # Return list of IoUs for each class

# Evaluate model with mIoU calculation
test_loss = 0.0
all_preds = []
all_masks = []
for images, masks in tqdm(test_loader, desc="Testing"):
    images, masks = images.to(device), masks.to(device)
    with torch.no_grad():
        outputs = model(images)
        loss = criterion(outputs, masks)
        test_loss += loss.item()
        
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())
        all_masks.append(masks.cpu())

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss}")

# Calculate mIoU for each class
all_preds = torch.cat(all_preds)
all_masks = torch.cat(all_masks)
ious = compute_iou(all_preds, all_masks, num_classes)
mean_iou = np.nanmean(ious)
print(f"Test Mean IoU (mIoU): {mean_iou}")

class_names = ["shell", "flipper", "head"]
for cls, class_name in enumerate(class_names, 1):  # Category IDs are 1, 2, 3
    iou = ious[cls-1]
    print(f"{class_name} mIoU: {iou:.4f}")

# Save predictions with visualizations
def save_predictions(dataset, model, device, output_dir="output_images", num_samples=5):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    for idx in indices:
        image, mask = dataset[idx]
        image_input = image.to(device).unsqueeze(0)
        with torch.no_grad():
            output = model(image_input)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        image_np = np.clip(image_np, 0, 1)
        true_mask = mask.numpy()
        
        # Ensure pred_mask and true_mask have the same shape and convert to uint8
        pred_mask = pred_mask.astype(np.uint8)
        true_mask = true_mask.astype(np.uint8)
        
        diff_mask = (true_mask != pred_mask).astype(np.uint8)
        
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(image_np)
        axs[0].set_title("Original Image")
        axs[0].axis('off')
        
        axs[1].imshow(true_mask, cmap='viridis')
        axs[1].set_title("True Mask")
        axs[1].axis('off')
        
        axs[2].imshow(pred_mask, cmap='viridis')
        axs[2].set_title("Predicted Mask")
        axs[2].axis('off')
        
        axs[3].imshow(diff_mask, cmap='hot')
        axs[3].set_title("Difference Mask (Errors)")
        axs[3].axis('off')
        
        # Calculate IoU for each category
        ious_sample = compute_iou(torch.tensor(pred_mask), torch.tensor(true_mask), num_classes)
        iou_text = "\n".join([f"{name} mIoU: {iou:.4f}" if not np.isnan(iou) else f"{name} mIoU: N/A" for name, iou in zip(class_names, ious_sample)])
        fig.suptitle(iou_text, fontsize=14)
        
        # Save figure
        output_path = os.path.join(output_dir, f"prediction_{idx}.png")
        plt.savefig(output_path)
        plt.close(fig)


# Run the visualization and save function
save_predictions(test_dataset, model, device)

print("Prediction images saved in the 'output_images' folder.")
