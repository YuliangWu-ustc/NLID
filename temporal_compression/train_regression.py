import os
import torch
import importlib
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.models as models
from dataset import DataFromfolders
from math import cos, pi
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter#tensorboard --logdir=log --port=6007
import sys
from tqdm import tqdm
sys.path.append('mobilenetv3')
CUDA_LAUNCH_BLOCKING=1

def normalize(a):
    an = (a-a.min())/(a.max()-a.min())
    return an
def build_module(module_path, module_name):
    module_path = module_path + '.' + module_name
    try:
        module = importlib.import_module(module_path)
        module = getattr(module, module_name)
    except:
        raise ModuleNotFoundError("No module named '{}'".format(module_path))
    return module

def warmup_cosine(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup_epoch = 10):
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    else:
        lr = lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def dataaug(evvoxels,p_horizontal=0.5):
    if random.random() < p_horizontal:
        evvoxels = torch.flip(evvoxels,dims=[4])

    shiftx = random.randint(0,8)
    shifty = random.randint(0,16)
    evvoxels = evvoxels[:,:,:,shifty:shifty+584,shiftx:shiftx+312]
    return evvoxels


def train(model, trainloader,valloader,criterion,optimizer,lr,num_epochs=300,device='cpu',mission_name='hh'):
    model.to(device)  
    model.train()
    minloss = 0.01

    max_iter = 15000
    iter = 0
    loss_iters = 50
    test_iters = 200
    test_iter = 100#用于test集中选取多少个iter
    train_loss = 0

    correctnum=0.0;total_samples = 0

    for epoch in range(num_epochs):
        
        for step, (label,evvoxels) in enumerate(trainloader):
            label = label.to(device).float().unsqueeze(dim=1)
            evvoxels = evvoxels.to(device).float()
            evvoxels = dataaug(evvoxels);evvoxels = evvoxels.reshape(evvoxels.shape[0], 60, evvoxels.shape[-2],evvoxels.shape[-1])
            optimizer.zero_grad()
            outputs = model(evvoxels)            
            loss = criterion(outputs,label)
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss1 = 1e-6 * l2_norm+loss
            loss1.backward()
            train_loss += loss.item()
            
            total_samples += label.size(0)

            correctnum += ( abs(outputs-label)<0.05 ).sum()
            optimizer.step()

            if iter % loss_iters==0 and iter!=0:
                current_lr= warmup_cosine(optimizer=optimizer,current_epoch=iter, max_epoch=max_iter,\
                    lr_min=lr[1],lr_max=lr[0], warmup_epoch = 10)
                
                print(f'iter{iter}, Loss: {train_loss /loss_iters:.8f},Acc:{correctnum/total_samples},LR: {current_lr:.8f}')
                with SummaryWriter(f'./log/{mission_name}/trainloss') as writer:
                    writer.add_scalar("loss",train_loss /loss_iters,iter)
                with SummaryWriter(f'./log/{mission_name}/train_acc') as writer:
                    writer.add_scalar("acc",correctnum/total_samples,iter)
                with SummaryWriter(f'./log/{mission_name}/LR') as writer:
                    writer.add_scalar("LR",current_lr,iter)

                train_loss=0
                correctnum=0.0;total_samples=0

            if iter % test_iters ==0 and iter!=0:
                valloss,valacc = val(model, valloader, criterion, device,test_iter)
                print(f'iter{iter}, ValLoss: {valloss:.8f},Acc: {valacc:.8f}') 

                with SummaryWriter(f'./log/{mission_name}/val_loss') as writer:
                    writer.add_scalar("loss",valloss,iter)
                with SummaryWriter(f'./log/{mission_name}/val_acc') as writer:
                    writer.add_scalar("acc",valacc,iter)

                if valloss < minloss:
                    state = {'model': model, 'optimizer': optimizer.state_dict()}
                    torch.save(state, 'ckpt/'+mission_name+f'model_minloss_{valloss:.8f}.pth')
                    print(f'Model saved at valloss {valloss:.6f}')
                    minloss = valloss
        
            if iter % (loss_iters*10) == 0 and iter!=0:
                state = {'model': model, 'optimizer': optimizer.state_dict()}
                torch.save(state, 'ckpt/'+mission_name+f'model.pth')
                print(f'Model saved at iter {iter}')
            
            iter+=1
        print(f'epoch:{epoch} finished')
        if iter>max_iter:
            break

            
def val(model, dataloader, criterion, device,test_iter):
    model.eval()  
    total_loss = 0.0
    total_samples = 0
    correctnum=0.0
    iter=0

    with torch.no_grad(): 
        for (label,evvoxels) in dataloader:
            label = label.to(device).float().unsqueeze(dim=1)
            evvoxels = evvoxels.to(device).float()
            evvoxels = dataaug(evvoxels);evvoxels = evvoxels.reshape(evvoxels.shape[0], 60, evvoxels.shape[-2],evvoxels.shape[-1])
            outputs = model(evvoxels)
            loss = criterion(outputs, label)

            total_loss += loss.item()
            total_samples += label.size(0)

            correctnum += ( abs(outputs-label)<0.05 ).sum()

            if iter>=test_iter:
                break
            iter+=1

    model.train()#
    return total_loss/test_iter,correctnum/total_samples
    

if __name__ == "__main__":
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    modelname = 'ResNet34_EST'

    lr = (5e-4,1e-11)
    # lr = (1e-3,1e-11)
    epoch=500
    from datetime import datetime
    import pytz
    tz = pytz.timezone('Asia/Shanghai')
    current_time = datetime.now(tz)
    mission_name=f'{modelname}-epoch_{epoch}-{current_time.strftime("%Y-%m-%d %H:%M:%S")}-RGB-lessimg'
    model = build_module('model',modelname)
    framenum=20*3
    batch_size = 16
    model = model(framenum=framenum)#model = model()# 

    pthpath = 'None'#'/home/user/Model_Lab/ckpt/MobileNetV3_Chem-epoch_500-2025-05-12 22:43:48-RGBmodel.pth'
    if os.path.exists(pthpath):
        checkpoint = torch.load(pthpath,map_location=device)
        model = checkpoint['model']
        optimizer = optim.Adam(model.parameters(),lr=2e-4)#lr=2e-4)
        print(f'Model loaded from {pthpath}')
    else:
        optimizer = optim.Adam(model.parameters(),lr=2e-4)
        print(f'No model found at {pthpath}. Starting with a new model.')



    data_folder_path = "/data/cls1-srv5-pool/Chem_liquid/2Ddata/Ethanol/RGB/train"
    testdata_folder_path = "/data/cls1-srv5-pool/Chem_liquid/2Ddata/Ethanol/RGB/test"
    # datanum = len(os.listdir(data_folder_path))//2
    # datalist = np.arange(datanum)
    # from sklearn.model_selection import train_test_split
    # trainlist,testlist,_,_ = train_test_split(datalist, datalist, test_size=0.2, random_state=42)

    trainlist = np.arange(len(os.listdir(data_folder_path))//2)
    testlist = np.arange(len(os.listdir(testdata_folder_path))//2)

    trainset = DataFromfolders(data_folder_path,trainlist)
    valset = DataFromfolders(testdata_folder_path,testlist)
    
    train_loader = data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True,  pin_memory=True,num_workers=4,drop_last=True)
    val_loader = data.DataLoader(dataset=valset, batch_size=batch_size, shuffle=True,  pin_memory=True,num_workers=4,drop_last=False)
    
    criterion = nn.MSELoss()
    model = model.to(device) 
    train(model, train_loader,val_loader, criterion, optimizer,lr = lr, num_epochs=epoch, device=device,mission_name=mission_name)

    print('ok')