import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import T_Dataset
from util import EDMLoss,AverageMeter
import option
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from UNet import UNet, UNetWithASPP
from FCN import  FCN, FCNWithASPP

f = open('log_test.txt', 'w')

opt = option.init()
opt.device = torch.device("cuda:0")

def adjust_learning_rate(params, optimizer, epoch):
    lr = params.init_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def criterion_s(inputs, target):
    losses = nn.functional.cross_entropy(inputs, target, ignore_index=255)
    # stop
    return losses

def create_data_part(opt):
    train_csv_path =  'train_half.csv'
    val_csv_path = 'valid.csv'
    test_csv_path = 'valid.csv'

    train_ds = T_Dataset(train_csv_path, opt.path_to_images, if_train = True )
    val_ds = T_Dataset(val_csv_path, opt.path_to_images, if_train = False )
    test_ds = T_Dataset(test_csv_path, opt.path_to_images, if_train=False )

    train_loader = DataLoader(train_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(opt.batch_size/2), num_workers=opt.num_workers, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False, drop_last=True)

    return train_loader, val_loader,test_loader

def train(opt,model, loader, optimizer, criterion, writer=None, global_step=None, name=None):
    model.train()
    train_losses = AverageMeter()

    for idx, (x,  mask) in enumerate(tqdm(loader)):
        x = x.to(opt.device)
        mask = mask.long()
        mask=mask.to(opt.device)
        pred = model(x)
        #print("y:",y)
        #print("pred:",pred)
        #print("mask shape:",mask.shape)
        #print("pred shape:",pred.shape)
        #print("pred:", pred.argmax(1).unique())
        #print("mask:", mask.unique())
        #stop
        #print("x scale:", torch.min(x), torch.max(x))
        #print("target_img scale:", torch.min(target_img), torch.max(target_img))
        #print("pred scale:", torch.min(pred), torch.max(pred))

        #save_image(x[0], 'x.jpg')
        #save_image(target_img[0], 'target_img.jpg')
        #stop

        loss=criterion(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.update(loss.item(), x.size(0))
    print("train loss(target_img-pred):", train_losses.avg)

    return train_losses.avg

def validate(opt,model, loader, criterion, writer=None, global_step=None, name=None):
    model.eval()
    validate_losses = AverageMeter()
    true_score = []
    pred_score = []
    Ssim = 0
    psnr_value = 0
    flag=0

    for idx, (x, mask) in enumerate(tqdm(loader)):
        x = x.to(opt.device)
        mask = mask.long()
        mask = mask.to(opt.device)

        pred = model(x)

        #save_image(pred[0], 'pred.jpg')
        #save_image(target_img[0], 'target_img.jpg')
        #stop
        loss = criterion(pred, mask)
        validate_losses.update(loss.item(), x.size(0))

        #Ssim = Ssim + torch.mean(
        #    ssim(pred.detach().to("cpu"), target_img.detach().to("cpu"), data_range=1, size_average=False))
        #for i in range(pred.shape[0]):
        #    psnr_value = psnr_value+calculate_psnr(pred[i], target_img[i])
        #batch_size = x.shape[0]
        #flag = flag+1

    #print("psnr:", psnr_value/(batch_size*flag))
    #Ssim = Ssim / flag
    #print("ssim:", Ssim.item())
    print("val loss:",validate_losses.avg)
    return validate_losses.avg, 0, 0, 0


def start_train(opt):

    train_loader, val_loader, test_loader = create_data_part(opt)

    model = UNetWithASPP()
    model_path = 'ckpt/UNetWithASPP_epoch_8_train_loss_0.27257630599267557_val_loss0.2381819197171026.pth'
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.init_lr)
    criterion = criterion_s
    #criterion.to(opt.device)
    model = model.to(opt.device)
    writer = None

    for e in range(opt.num_epoch):
        adjust_learning_rate(opt, optimizer, e)
        train_loss = train(opt, model=model, loader=train_loader,
                           optimizer=optimizer, criterion=criterion,
                           writer=writer, global_step=len(train_loader) * e,
                           name=f"{opt.experiment_dir_name}_by_batch")
        val_loss, vacc, ssim_val, psnr_val = validate(opt, model=model,
                                               loader=val_loader, criterion=criterion,
                                               writer=writer, global_step=len(val_loader) * e,
                                               name=f"{opt.experiment_dir_name}_by_batch")

        model_name = f"UNetWithASPP_epoch_{e}_train_loss_{train_loss}_val_loss{val_loss}.pth"
        torch.save(model.state_dict(), os.path.join(opt.experiment_dir_name, model_name))

    writer.close()
    f.close()



if __name__ =="__main__":
    import warnings

    warnings.filterwarnings("ignore")

    start_train(opt)