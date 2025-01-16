import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss

class_weights = torch.tensor([0.1, 1.0, 2.0, 3.0]).cuda()  # Set according to category weight

# Define criterion combining cross-entropy and Dice loss
dice_loss = DiceLoss(mode='multiclass')
ce_loss = nn.CrossEntropyLoss(weight=class_weights)

def criterion(outputs, masks):
    return ce_loss(outputs, masks) + dice_loss(outputs, masks)