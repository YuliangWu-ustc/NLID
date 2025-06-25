#这是浓度检测的dataset函数
import torch.utils.data as data
import torch
import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from  PIL import  Image
from sklearn.model_selection import train_test_split

def normalize(a):
    an = (a-a.min())/(a.max()-a.min())*255
    return an

    
class DataFromfolders(data.Dataset):
    def __init__(self, folderpath,datalist):
        '''
        k: 一次迭代给网络的事件帧数
        self.datasetnum: 不同基准图案数据集数量
        '''
        self.folderpath = folderpath
        self.datanum = len(datalist)
        self.datalist = datalist

    def __getitem__(self, idx):
        evvoxels = np.load(os.path.join(self.folderpath,f'{self.datalist[idx]:08d}.npy'))
        with open(os.path.join(self.folderpath,f'{self.datalist[idx]:08d}.txt'), 'r') as txt_file:
            label = txt_file.read()
        return int(label)/10,evvoxels
    
    def __len__(self):
        return self.datanum

class DataFromfolders_classification(data.Dataset):
    def __init__(self, folderpath,datalist):
        '''
        k: 一次迭代给网络的事件帧数
        self.datasetnum: 不同基准图案数据集数量
        '''
        self.folderpath = folderpath
        self.datanum = len(datalist)
        self.datalist = datalist

    def __getitem__(self, idx):
        evvoxels = np.load(os.path.join(self.folderpath,f'{self.datalist[idx]:08d}.npy'))
        with open(os.path.join(self.folderpath,f'{self.datalist[idx]:08d}.txt'), 'r') as txt_file:
            label = txt_file.read()
        return int(label),evvoxels
    
    def __len__(self):
        return self.datanum


if __name__ == '__main__':
    data_folder_path = "/data/cls1-srv5-pool/Chem_liquid/ethanol-concentration/voxel"
    datanum = len(os.listdir(data_folder_path))//2
    datalist = np.arange(datanum)
    trainset = DataFromfolders("/data/cls1-srv5-pool/Chem_liquid/ethanol-concentration/voxel",datalist)
    train_loader = data.DataLoader(dataset=trainset, batch_size=16, shuffle=True,  pin_memory=True)
    # viz = Visdom()
    for epoch in range(100):
        for step, (label,events) in enumerate(train_loader):
            print("step:", step)
            cv2.imwrite('f.png',normalize(events[0][10]).numpy())
            print(label[0])
            # viz.image(normalize(events[0][3]),win='ok')

            
            
            
   