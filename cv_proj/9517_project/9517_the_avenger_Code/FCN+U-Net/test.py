from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataset import T_Dataset
import option
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from UNet import *
from FCN import *
f = open('log_test.txt', 'w')

opt = option.init()
opt.device = torch.device("cuda:0")


def calculate_iou(pred_tensor, gt_tensor, class_label):
    pred_mask = (pred_tensor == class_label)
    gt_mask = (gt_tensor == class_label)

    intersection = torch.logical_and(pred_mask, gt_mask).sum().item()
    union = torch.logical_or(pred_mask, gt_mask).sum().item()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union
    return iou

def quantize_tensor(tensor):
    quantized_tensor = torch.round(tensor).int()

    quantized_tensor = torch.clamp(quantized_tensor, min=0, max=3)

    return quantized_tensor


def create_data_part(opt):
    train_csv_path =  'train.csv'
    val_csv_path = 'valid_half.csv'
    test_csv_path = 'test.csv'

    train_ds = T_Dataset(train_csv_path, opt.path_to_images, if_train = True )
    val_ds = T_Dataset(val_csv_path, opt.path_to_images, if_train = False )
    test_ds = T_Dataset(test_csv_path, opt.path_to_images, if_train=False )

    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(opt.batch_size/2), num_workers=opt.num_workers, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False, drop_last=True)

    return train_loader, val_loader,test_loader

train_loader, val_loader, test_loader = create_data_part(opt)

model = FCNWithASPP()
model_path = 'FCNWithASPP_epoch_6_train_loss_0.3181816271760247_val_loss0.3049879584381049.pth'
model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
model = model.to(opt.device)

IoU_1 = 0
IoU_2 = 0
IoU_3 = 0
flag = 0

model.eval()
for idx, (x, mask) in enumerate(tqdm(test_loader)):
    x = x.to(opt.device)
    mask = mask.to(opt.device)
    pred = model(x)
    pred = pred.argmax(1)
    flag = flag+1
    for i in range(x.shape[0]):
        IoU_1 = IoU_1 + calculate_iou(quantize_tensor(pred[i]), mask[i], class_label=1)
        IoU_2 = IoU_2 + calculate_iou(quantize_tensor(pred[i]), mask[i], class_label=2)
        IoU_3 = IoU_3 + calculate_iou(quantize_tensor(pred[i]), mask[i], class_label=3)

IoU_1 = IoU_1/(flag*x.shape[0])
IoU_2 = IoU_2/(flag*x.shape[0])
IoU_3 = IoU_3/(flag*x.shape[0])


print("head IoU：", IoU_1)
print("flippers IoU：", IoU_2)
print("carapace IoU：", IoU_3)
