import os
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils.dataloader.TSC_data_loader import my_TSC_data_loader, simple_TSC_data_loader
from Classifier.InceptionNet_gradcam import Inception_module, InceptionNet
from Classifier.Easy_use_trainner import Easy_use_trainner
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

device = "cuda:1"

# 定义选择区域和 delta_t
SELECTED_AREAS = [
    # (530, 200, 810, 210), # y_division_val = 10, 1/54
    # (530, 480, 810, 490), # y_division_val = 10, 1/54
    # # (530, 460, 810, 540), # y_division_val = 80, 1/6.75
    (530, 270, 810, 540),  # y_division_val = 270, 1/2
    # # (530, -1, 810, 600) # y_division_val = inf
    ]
DELTA_T_LIST = [
    # 1000, # 1/8000
    # 3000, # 1/2700
    10000, # 1/800
    # 30000, # 1/270
    # 100000 # 1/80
]


# 定义 y_division_val 和 t_division_val
# y_division_val = ['1/54(y=200)', '1/54(y=480)', '1/6.75(y=460)', '1/2(y=270)', '1']
# t_division_val = ['1000', '3000', '10000', '30000', '100000']

SELECTED_AREA_DESCRIPTION = {
    (530, 270, 810, 540): '1/2',
}

ARCHIVE_PATH = '/data/cls1-srv5-pool/Chem_liquid/'

# dataset_name = 'regression'
dataset_name = 'classification'
# dataset_name = 'all'

_dataset_info_dict = {
    'regression':{
        'dataset_path' : os.path.join(ARCHIVE_PATH, 'regression/processed/'),
        'Results_path' : './Results/regression/',
        'folder_labels_dict' : {
            'water_H1-50':      0,
            'ethanol_10_H1-50': 1,
            'ethanol_20_H1-50': 2,
            'ethanol_30_H1-50': 3,
            'ethanol_40_H1-50': 4,
            'ethanol_50_H1-50': 5,
            'ethanol_60_H1-50': 6,
            'ethanol_70_H1-50': 7,
            'ethanol_80_H1-50': 8,
            'ethanol_90_H1-50': 9,
            'ethanol_100_H1-50':    10,
        },
        'folder_labels_dict_ex':{
            'ethanol_20_H1-50-T2':  11,
        }
    },
    
    'classification':{
        'dataset_path' : os.path.join(ARCHIVE_PATH, 'classification/processed/'),
        'Results_path' : './Results/classification/',
        'folder_labels_dict' : {
            'water_H1-50':          0,
            'ethanol_20_H1-50':     1,
            'ethanol_40_H1-50':     2,
            'ethanol_60_H1-50':     3,
            'ethanol_80_H1-50':     4,
            'ethanol_100_H1-50':    5,
            
            'DMSO-50':              6, # DMSO
            'EtOAc_H1-50':          7, # 乙酸乙酯
            # 'CH3COOCH2CH3_H1-50':   15, # 乙酸乙酯
            'BnOH_H1-50':           8, # 苯甲醇
            # 'PhCO_H1-50':           17, # 苯甲醇
            'iPrOH_H1-50':          9, # 异丙醇
            # 'CC(O)C_H1-50':         15, # 异丙醇
            'C6H14_H1-50-spill-manual': 10, # 正己烷
            'NaOH_1M_H1-50':        11, # NaOH
            
            'alcohol_56v_H1-50':    12, # 白酒
            'alcohol_15v_H1-50':    13, # 清酒
            'alcohol_vodka_40v_H1-50': 14, # 伏特加
            'vinegar_9_H1-50':      15, # 醋
            # 'medical_alcohol_75v_H1-50': 16, # 医用酒精
            'coconut_water_H1-50':  16, # 椰子水
            'NaCl_1M_H1-50':        17, # NaCl
            'NaCl_2M_H1-50':        18, # NaCl
            # 'NaCl_10_H1-50':        27,
            'glucose_10_H1-50':     19, # 葡萄糖
            'NaHCO3_0.5M_H1-50':    20, # NaHCO3
            
            'C3H8O3_10_H1-50':      21, # 甘油
            'C3H8O3_20_H1-50':      22, # 甘油
            'C3H8O3_30_H1-50':      23, # 甘油
            'C3H8O3_40_H1-50-B':    24, # 甘油
            
        },
        'folder_labels_dict_ex':{
            'ethanol_20_H1-50-T2':  26,
        },
        'label_descriptions':{
            0: 'Water',
            1: 'Ethanol $20_{\%}$',
            2: 'Ethanol $40_{\%}$',
            3: 'Ethanol $60_{\%}$',
            4: 'Ethanol $80_{\%}$',
            5: 'Ethanol $100_{\%}$',
            
            6: 'DMSO',
            7: 'ethyl acetate',
            8: 'benzyl alcohol',
            9: 'isopropyl alcohol',
            10: 'n-hexane',
            11: 'NaOH 1mol/L',
            12: 'liquor',
            13: 'liquor',
            14: 'Vodka $40_{vol\%}$',
            15: 'Vinegar $9_{\%}$',
            16: 'Coconut\nWater',
            17: 'NaCl 1mol/L',
            18: 'NaCl 2mol/L',
            19: 'Glucose $10_{\%}$',
            20: 'NaHCO3 0.5mol/L',
            21: 'Glycerol $10_{\%}$',
            22: 'Glycerol $20_{\%}$',
            23: 'Glycerol $30_{\%}$',
            24: 'Glycerol $40_{\%}$',
        }
        
    },
    # 'all':{
    #     'dataset_path' : os.path.join(ARCHIVE_PATH, 'classification/processed/'),
    #     'folder_labels_dict' : {
    #         'water_H1-50':      0,
    #         'ethanol_10_H1-50': 1,
    #         'ethanol_20_H1-50': 2,
    #         'ethanol_30_H1-50': 3,
    #         'ethanol_40_H1-50': 4,
    #         'ethanol_50_H1-50': 5,
    #         'ethanol_60_H1-50': 6,
    #         'ethanol_70_H1-50': 7,
    #         'ethanol_80_H1-50': 8,
    #         'ethanol_90_H1-50': 9,
    #         'ethanol_100_H1-50':    10,
            
    #         'DMSO-50':              11, # DMSO
    #         'EtOAc_H1-50':          12, # 乙酸乙酯
    #         # 'CH3COOCH2CH3_H1-50':   15, # 乙酸乙酯
    #         'BnOH_H1-50':           13, # 苯甲醇
    #         # 'PhCO_H1-50':           17, # 苯甲醇
    #         'iPrOH_H1-50':          14, # 异丙醇
    #         # 'CC(O)C_H1-50':         15, # 异丙醇
    #         'C6H14_H1-50-spill-manual': 15, # 正己烷
    #         'NaOH_1M_H1-50':        16, # NaOH
            
    #         'alcohol_56v_H1-50':    17, # 白酒
    #         'alcohol_15v_H1-50':    18, # 清酒
    #         'alcohol_vodka_40v_H1-50': 19, # 伏特加
    #         'vinegar_9_H1-50':      20, # 醋
    #         'medical_alcohol_75v_H1-50': 21, # 医用酒精
    #         'coconut_water_H1-50':  22, # 椰子水
    #         'NaCl_1M_H1-50':        23, # NaCl
    #         'NaCl_2M_H1-50':        24, # NaCl
    #         # 'NaCl_10_H1-50':        27,
    #         'glucose_10_H1-50':     25, # 葡萄糖
    #         'NaHCO3_0.5M_H1-50':    26, # NaHCO3
            
    #         'C3H8O3_10_H1-50':      27, # 甘油
    #         'C3H8O3_20_H1-50':      28, # 甘油
    #         'C3H8O3_30_H1-50':      29, # 甘油
    #         'C3H8O3_40_H1-50-B':    30, # 甘油
            
    #     },
    #     'folder_labels_dict_ex':{
    #         'ethanol_20_H1-50-T2':  31,
    #     }
    # },
}


ROI_description = {
    (530, 480, 810, 490) : '1/50',
    (530, 270, 810, 540) : '1/2'
}

# 定义颜色映射
colors_dark = ['#598e9d', '#da4453', '#51BF86', '#b9b0bf', '#f99339'] # 蓝色、红色、绿色、紫色、橙色
colors_light = ['#C8D9DE', '#dacccc', '#d3efe0', '#E8E4E9', '#FDE4CD'] # 浅蓝色、浅红色、浅绿色、浅紫色、浅橙色

dataset_path = _dataset_info_dict[dataset_name]['dataset_path']
folder_labels_dict = _dataset_info_dict[dataset_name]['folder_labels_dict']
label_descriptions = _dataset_info_dict[dataset_name]['label_descriptions']

results_path = _dataset_info_dict[dataset_name]['Results_path']
    
# 遍历每个选择区域
Result_log_path = './Results/' + dataset_name + '/'

##########################################################
# customize

batchsize = 1  
config = {
    0: {
        'SELECTED_AREA': (530, 270, 810, 540),  # y_division_val = 270, 1/2
        'DELTA_T': 10000,
        'SELECTED_LABEL': 0,
        'FOLDER_LABELS_DICT_SINGLE_LABEL_ROI_1': {
            'water_H1-50': 0,
        }
    },
    1: {
        'SELECTED_AREA': (530, 270, 810, 540),  # y_division_val = 270, 1/2
        'DELTA_T': 10000,
        'SELECTED_LABEL': 23,
        'FOLDER_LABELS_DICT_SINGLE_LABEL_ROI_1': {
            'C3H8O3_30_H1-50': 23, # 甘油
        }
    },
    2: {
        'SELECTED_AREA': (530, 270, 810, 540),  # y_division_val = 270, 1/2
        'DELTA_T': 10000,
        'SELECTED_LABEL': 5,
        'FOLDER_LABELS_DICT_SINGLE_LABEL_ROI_1': {
            'ethanol_100_H1-50':    5,
        }
    },
    3: {
        'SELECTED_AREA': (530, 270, 810, 540),  # y_division_val = 270, 1/2
        'DELTA_T': 10000,
        'SELECTED_LABEL': 6,
        'FOLDER_LABELS_DICT_SINGLE_LABEL_ROI_1': {
            'DMSO-50':              6, # DMSO
        }
    }
}

# end customize
##########################################################
##########################################################
# 创建主画布和网格布局

n_subplots = 4
vmargin = 0.5
fig = plt.figure(figsize=(10, 2 * n_subplots))

# 主网格分为3部分：上、中、下，高度比例为（4单位，0.5单位间隔，4单位）
main_gs = gridspec.GridSpec(1, 3, width_ratios=[0.0001, 1, 0.0001], wspace = 0)

content1_gs = gridspec.GridSpecFromSubplotSpec(n_subplots * 2, 1, subplot_spec=main_gs[1], # 上层
                                        height_ratios=[
                                            0.5, 1,
                                            vmargin, 1,
                                            vmargin, 1,
                                            vmargin, 1,
                                        ],
                                        # width_ratios=[1.5, 1, 1, 1], 
                                        hspace=0)

ax_list = []
ax_spacer_list = []
for i in range(n_subplots):
    ax_spacer = fig.add_subplot(content1_gs[2 * i])
    ax_spacer_list.append(ax_spacer)
    ax_spacer.axis('off')
    
    ax = fig.add_subplot(content1_gs[2 * i + 1])
    ax_list.append(ax)

for i_subplot in range(n_subplots):

    ##########################################################
    # model init

    SELECTED_AREA = config[i_subplot]['SELECTED_AREA']
    DELTA_T = config[i_subplot]['DELTA_T']
    SELECTED_LABEL = config[i_subplot]['SELECTED_LABEL']
    FOLDER_LABELS_DICT_SINGLE_LABEL = config[i_subplot]['FOLDER_LABELS_DICT_SINGLE_LABEL_ROI_1']

    print(f'###' * 5)
    print(f'subplot {i_subplot} - {SELECTED_AREA} - {DELTA_T} - {label_descriptions[SELECTED_LABEL]} - {FOLDER_LABELS_DICT_SINGLE_LABEL}')

    # 加载数据
    selected_area_list = [SELECTED_AREA]
    X_train, y_train, X_test, y_test = my_TSC_data_loader(dataset_path, folder_labels_dict, selected_area_list, DELTA_T)

    indices = np.where(y_train == SELECTED_LABEL)[0]
    X_this_label = X_train[indices]
    y_this_label = y_train[indices]

    # SELECTED_SAMPLE_INDICES = [0]
    # X_selected_sample = X_this_label[SELECTED_SAMPLE_INDICES]
    # y_selected_sample = y_this_label[SELECTED_SAMPLE_INDICES]

    X_train = X_this_label
    y_train = y_this_label

    # 构建模型
    model = InceptionNet(input_channle_size=1, nb_classes=len(folder_labels_dict))

    # 加载最佳模型
    model_path = Result_log_path + f'{dataset_name}_select_{SELECTED_AREA}_t{DELTA_T}/' + f'{dataset_name}_select_{SELECTED_AREA}_t{DELTA_T}' + 'Best_model'

    # 创建trainer并加载模型
    trainer = Easy_use_trainner(
        Result_log_folder=Result_log_path,
        dataset_name=f'{dataset_name}_select_{SELECTED_AREA}_t{DELTA_T}',
        device=device
    )
    trainer.get_model(model)
    trainer.load_model(model_path)

    ##########################################################
    # eval model & cal grad-cam

    print('code is running on ',device)

    # covert numpy to pytorch tensor and put into gpu
    X_train = torch.from_numpy(X_train).float()
    X_train.requires_grad = True
    X_train = X_train.to(device)
    y_train = torch.from_numpy(y_train).long().to(device)


    X_test = torch.from_numpy(X_test).float()
    X_test.requires_grad = True
    X_test = X_test.to(device)
    y_test = torch.from_numpy(y_test).long().to(device)


    # add channel dimension to time series data
    if len(X_train.shape) == 2:
        X_train = X_train.unsqueeze_(1)
        X_test = X_test.unsqueeze_(1)

    # loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    # build dataloader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

    for sample in train_loader:
        model.eval()
        X = sample[0].requires_grad_(True)
        y = sample[1]
        y_predict = model(X)
        loss = criterion(y_predict, y)
        loss.backward()
        break # 只利用第一个batch，只计算一次梯度


    layer_idx_list = []
    grads = []
    features = []

    last_grad = None

    # 直接遍历layer_list中的所有模块
    for layer_idx, layer_module in enumerate(model.layer_list):
        # 检查是否是Inception_module类型
        if isinstance(layer_module, Inception_module):
            # 检查梯度列表是否为空
            if len(layer_module.grad) > 0:
                last_grad = layer_module.grad[-1]
                last_feature = layer_module.features[-1]
                
                layer_idx_list.append(layer_idx)
                grads.append(last_grad)
                features.append(last_feature)
                
                print(f"Layer {layer_idx} - Gradient Shape: {last_grad.shape}")
                print(f"Layer {layer_idx} - Feature Shape: {last_feature.shape}")


    # 计算grad_cam
    grad_cam_list = []

    for i, layer_idx in enumerate(layer_idx_list):
        
        grad_this_layer = grads[i].cpu().numpy()
        feature_this_layer = features[i].detach().cpu().numpy()
        
        mean_grad = np.mean(grad_this_layer, axis=2) # shape = (batch, channel, sequence(time))
        
        cam = np.zeros((feature_this_layer.shape[0], feature_this_layer.shape[2]), dtype=np.float32)
        
        for channel_idx in range(mean_grad.shape[1]):
            mean_grad_this_channel = mean_grad[:, channel_idx] # shape = (batch, sequence(time))

            cam += mean_grad_this_channel * feature_this_layer[:, channel_idx, :]
            
        cam = np.maximum(cam, 0)
        print(f"cam.shape: {cam.shape}")
        print(f"cam.nunique: {len(np.unique(cam))}")
        
        grad_cam_list.append(cam)
        
    last_layer_grad_cam = grad_cam_list[-1]

    X = X.detach().cpu().numpy()
    X = X.squeeze(1)
    X = X / DELTA_T
    print(f'selected_y: {y.cpu().numpy()}')
    label_name = label_descriptions[y.cpu().numpy()[0]]
    print(f"y[0]: {label_name}")

    ######################################################################################
    # 绘制grad-cam

    label_name = label_descriptions[y.cpu().numpy()[0]]
    print(f'selected label: {label_name}')
    cam_this_layer = last_layer_grad_cam

    ax_list[i_subplot].set_title(f'{label_name} ROI {SELECTED_AREA_DESCRIPTION[SELECTED_AREA]}')

    for sameple_idx in range(cam_this_layer.shape[0]):
        
        cam = cam_this_layer[sameple_idx]
        X_this_sample = X[sameple_idx, :]
        epsilon = 1e-10
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + epsilon)
        
        print(f'cam.shape: {cam.shape}')
        print(f'X_this_sample.shape: {X_this_sample.shape}')
        
        # 数据归一化到[0,1]区间（热力图常用范围）
        cam_normalized = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

        # 将一维数据转换为二维（热力图要求至少二维）
        heatmap_data = cam_normalized.reshape(1, -1)  # 形状变为 (1, length)

        ax_list[i_subplot].plot(X_this_sample)
        # 绘制热力图
        img = ax_list[i_subplot].imshow(heatmap_data, 
                        cmap='jet',  # 颜色映射方案
                        aspect='auto',   # 自动调整宽高比
                        extent=[0, len(cam), 0, 15],  # 坐标范围
                        origin='lower',
                        alpha = 0.5)
        
        ax_list[i_subplot].set_yticks([0, 5, 10, 15])

            
ax_spacer_list[0].set_title(f'grad-cam ROI {SELECTED_AREA_DESCRIPTION[SELECTED_AREA]} {DELTA_T} {label_name}', fontsize=12)
fig.tight_layout()
fig.show()

# plt.savefig(Result_log_path + f'/grad-cam_{SELECTED_AREA}_t{DELTA_T}.png')
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# plt.show()
