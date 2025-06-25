'''
将EVENT raw数据转换成event voxel数据保存
'''
import sys
sys.path.append("/usr/lib/python3/dist-packages/")
import cv2        
import numpy as np
import os
import torch
from metavision_core.event_io import EventsIterator
import re
from tqdm import tqdm


def readraw(filepath):
    mv_iterator = EventsIterator(input_path=filepath, delta_t=100000)
    xs = []
    ys = []
    ps = []
    ts = []
    k = 0
    for evs in mv_iterator: 
        if len(evs) > 0 :
            print(len(evs))
            xs.append(evs['x'])
            ys.append(evs['y'])
            ps.append(evs['p'])
            ts.append(evs['t'])
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    ts = np.concatenate(ts)

    return xs,ys,ts,ps
def normalize(img):
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img

# def generate_evvoxels(x,y,t,p,fnum,shiftrate):
#     evvoxels = []
#     evnum = len(x)
#     evgap = evnum//fnum
#     shift = int(shiftrate*evgap)
#     voxelgap = evgap//12#t.max()//300#
    
#     for i in range(fnum):
#         start = i*evgap+shift
#         end= i*evgap+shift+voxelgap 
#         x_,y_,t_,p_ = x[start:end],y[start:end],t[start:end],p[start:end]
#         evvoxel = generate_evvoxel_PandN(x_,y_,t_,p_)
#         evvoxels.append(normalize(evvoxel.numpy()))
#     return evvoxels

def generate_evvoxel_PandN(x,y,t,p,accumulate=False):
    imgsize =(720,1280)
    evvoxel = torch.zeros(imgsize, dtype=torch.float32)

    x1 = x
    y1 = y
    # t1 = t
    p1 = p*2-1
    
    total = len(x)
    for i in range(total//1000):
        
        x_ = torch.from_numpy(x1[i*1000:(i+1)*1000].astype(np.int16)).long()
        y_ = torch.from_numpy(y1[i*1000:(i+1)*1000].astype(np.int16)).long()
        # t1 = torch.from_numpy(t1).float()
        p_ = torch.from_numpy(p1[i*1000:(i+1)*1000].astype(np.int16)).float() 
    
        evvoxel.index_put_((y_, x_), p_, accumulate=accumulate)
        
    return evvoxel[:600,520:840]
def generate_timesurface(x,y,t,p):
    imgsize =(720,1280)
    timesurface = torch.zeros(imgsize, dtype=torch.float32)
    timesurface_p = torch.zeros(imgsize, dtype=torch.float32)
    timesurface_n = torch.zeros(imgsize, dtype=torch.float32)

    x1 = x
    y1 = y
    t1 = (t-t.min())/(t.max()-t.min())
    p1 = p*2-1
    
    total = len(x)
    for i in range(total//1000):
        
        x_ = torch.from_numpy(x1[i*1000:(i+1)*1000].astype(np.int16)).long()
        y_ = torch.from_numpy(y1[i*1000:(i+1)*1000].astype(np.int16)).long()
        t_ = torch.from_numpy(t1[i*1000:(i+1)*1000]).float()
        p_ = torch.from_numpy(p1[i*1000:(i+1)*1000].astype(np.int16)).float() 
    
        timesurface.index_put_((y_, x_), p_*(2**t_-1), accumulate=False)
    timesurface_p = timesurface*(timesurface>0)
    timesurface_n = timesurface*(timesurface<0)
    timesurfaceout = torch.stack((timesurface_p, timesurface_n))
    return timesurfaceout[:,:600,520:840]

def VoxelGrid(x,y,t,p,normalize=False,T=5):
    input_size=(T,720,1280)
    voxel_grid = torch.zeros((input_size), dtype=torch.float, requires_grad=False)
    C, H, W = voxel_grid.shape
    with torch.no_grad():
        t_norm = t
        t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])
        
        x0 = x.int()
        y0 = y.int()
        t0 = t_norm.int()

        value = 2*p-1

        for xlim in [x0,x0+1]:
            for ylim in [y0,y0+1]:
                for tlim in [t0,t0+1]:

                    mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < input_size[0])
                    interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())

                    index = H * W * tlim.long() + \
                            W * ylim.long() + \
                            xlim.long()
                    voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

        if normalize:
            mask = torch.nonzero(voxel_grid, as_tuple=True)
            if mask[0].size()[0] > 0:
                mean = voxel_grid[mask].mean()
                std = voxel_grid[mask].std()
                if std > 0:
                    voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                else:
                    voxel_grid[mask] = voxel_grid[mask] - mean
    return voxel_grid[:,:600,520:840]

def generate_timesurfaces(x,y,t,p,fnum,shiftrate):
    timesurfaces = []
    evnum = len(x)
    evgap = evnum//fnum
    shift = int(shiftrate*evgap)
    voxelgap = evgap//12#t.max()//300#
    
    for i in range(fnum):
        start = i*evgap+shift
        end= i*evgap+shift+voxelgap 
        x_,y_,t_,p_ = x[start:end],y[start:end],t[start:end],p[start:end]
        timesurface = generate_timesurface(x_,y_,t_,p_)
        timesurfaces.append(normalize(timesurface.numpy()))
    return timesurfaces

def generate_eventframe(x,y,t,p,fnum,shiftrate):
    evframes = []
    evnum = len(x)
    evgap = evnum//fnum
    shift = int(shiftrate*evgap)
    voxelgap = evgap//12#t.max()//300#
    
    for i in range(fnum):
        start = i*evgap+shift
        end= i*evgap+shift+voxelgap 
        x_,y_,t_,p_ = x[start:end],y[start:end],t[start:end],p[start:end]
        evframe = generate_evvoxel_PandN(x_,y_,t_,p_,accumulate=False)
        evframes.append(evframe.unsqueeze(0).numpy())
    return evframes
        
def generate_voxelgrid(x,y,t,p,fnum,shiftrate):
    voxelgrids = []
    evnum = len(x)
    evgap = evnum//fnum
    shift = int(shiftrate*evgap)
    voxelgap = evgap//12#t.max()//300#
    
    for i in range(fnum):
        start = i*evgap+shift
        end= i*evgap+shift+voxelgap 
        x_,y_,t_,p_ = x[start:end],y[start:end],t[start:end],p[start:end]
        voxelgrid = VoxelGrid(x_,y_,t_,p_)
        voxelgrids.append(voxelgrid.numpy())
    return voxelgrids

def generate_eventcount(x,y,t,p,fnum,shiftrate):
    evcounts = []
    evnum = len(x)
    evgap = evnum//fnum
    shift = int(shiftrate*evgap)
    voxelgap = evgap//12#t.max()//300#
    
    for i in range(fnum):
        start = i*evgap+shift
        end= i*evgap+shift+voxelgap 
        x_,y_,t_,p_ = x[start:end],y[start:end],t[start:end],p[start:end]
        evcount = generate_evvoxel_PandN(x_,y_,t_,p_,accumulate=True)
        evcount_p = evcount*(evcount>0)
        evcount_n = evcount*(evcount<0)
        evcountout = torch.stack((evcount_p, evcount_n))
        evcounts.append(evcountout.numpy())
    return evcounts

class generate_rep:
    def __init__(self,evtype):
        self.type = evtype
    def forward(self,x,y,t,p,fnum,shiftrate):
        if self.type == 'timesurface':
            return generate_timesurfaces(x,y,t,p,fnum,shiftrate)
        elif self.type == 'eventframe':
            return generate_eventframe(x,y,t,p,fnum,shiftrate)
        elif self.type == 'voxelgrid':
            return generate_voxelgrid(x,y,t,p,fnum,shiftrate)
        elif self.type == 'eventcount':
            return generate_eventcount(x,y,t,p,fnum,shiftrate)

if __name__ == '__main__':
    from tqdm import tqdm
    reptypes = ['eventframe',
            'eventcount',
            # 'voxelgrid',
            'timesurface']
    # reptypes = ['timesurface']
    fnum=20
    shiftnum=10

    for reptype in reptypes:

        generate_evrepresentation = generate_rep(reptype)

        #用于分类数据集
        voxelpath = os.path.join("/data/cls1-srv5-pool/Chem_liquid/2Ddata/classification",reptype,'train')
        voxelpath_test = os.path.join("/data/cls1-srv5-pool/Chem_liquid/2Ddata/classification",reptype,'test')
        subfolder_paths = ['/data/cls1-srv5-pool/Chem_liquid/water-ethanol-black-background/H1/water_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/water-ethanol-black-background/H1/ethanol_20_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/water-ethanol-black-background/H1/ethanol_40_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/water-ethanol-black-background/H1/ethanol_60_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/water-ethanol-black-background/H1/ethanol_80_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/water-ethanol-black-background/H1/ethanol_100_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/chem-liquid/H1/DMSO-50',
                    '/data/cls1-srv5-pool/Chem_liquid/chem-liquid/H1/EtOAc_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/chem-liquid/H1/BnOH_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/chem-liquid/H1/iPrOH_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/chem-liquid/H1/C6H14_H1-50-spill-manual',
                    '/data/cls1-srv5-pool/Chem_liquid/chem-liquid/H1/NaOH_1M_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/life-liquid/H1/alcohol_56v_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/life-liquid/H1/alcohol_15v_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/life-liquid/H1/alcohol_vodka_40v_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/life-liquid/H1/vinegar_9_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/life-liquid/H1/coconut_water_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/life-liquid/H1/NaCl_1M_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/life-liquid/H1/NaCl_2M_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/life-liquid/H1/glucose_10_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/life-liquid/H1/NaHCO3_0.5M_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/water-OCC(O)CO/H1/C3H8O3_10_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/water-OCC(O)CO/H1/C3H8O3_20_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/water-OCC(O)CO/H1/C3H8O3_30_H1-50',
                    '/data/cls1-srv5-pool/Chem_liquid/water-OCC(O)CO/H1/C3H8O3_40_H1-50-B']


        i = 0;i_test=0
        label = 0
        for f in subfolder_paths:
            subfolder_path=f
            
            if not os.path.exists(voxelpath):
                os.makedirs(voxelpath)
            if not os.path.exists(voxelpath_test):
                os.makedirs(voxelpath_test)
            evpoint_files = [f1 for f1 in os.listdir(subfolder_path) if f1.endswith('.raw')]
            
            for k,rf in tqdm(enumerate(evpoint_files)):
                input_path = os.path.join(subfolder_path, rf)

                x,y,t,p = readraw(input_path)
                if reptype=='voxelgrid':
                    x = torch.tensor(x.astype(np.float32))
                    y = torch.tensor(y.astype(np.float32))
                    t = torch.tensor(t.astype(np.float32))
                    p = torch.tensor(p.astype(np.float32))
                
                if k<10:
                    for j in range(shiftnum):
                        out_path = os.path.join(voxelpath_test, f'{i_test:08d}.npy')
                        evvoxels = generate_evrepresentation.forward(x,y,t,p,fnum,j/shiftnum)
                        np.save(out_path, evvoxels)
                        with open(os.path.join(voxelpath_test, f'{i_test:08d}.txt'), 'w') as txt_file:
                            txt_file.write(f'{label:08d}')
                        i_test+=1
                else:
                    for j in range(shiftnum):
                        out_path = os.path.join(voxelpath, f'{i:08d}.npy')
                        evvoxels = generate_evrepresentation.forward(x,y,t,p,fnum,j/shiftnum) 

                        np.save(out_path, evvoxels)
                        with open(os.path.join(voxelpath, f'{i:08d}.txt'), 'w') as txt_file:
                            txt_file.write(f'{label:08d}')
                        i+=1
            label += 1

            print(f"Finish processing {f}")

    
    

    
