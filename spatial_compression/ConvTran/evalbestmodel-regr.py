tag_this_file = '''\neval best model, regr-gly 1/2, 10000, test=raw'''

from datetime import datetime
import json
import os
import argparse
import logging
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from art import *
# Import Project Modules -----------------------------------------------------------------------------------------------
from utils import Setup, Initialization, Data_Loader, create_dirs, dataset_class, Data_Verifier
from Models.model_regr2 import model_factory, count_parameters
from Models.optimizers import get_optimizer
from Models.loss_regr2 import get_loss_module
from Models.utils import load_model
from Training_regr2 import SupervisedTrainer, train_runner
from sklearn.utils import shuffle
from utils import dataset_class_regr

cuda_device = "1"

# best_model_path = r'./Results/regression_(530, 270, 810, 540)_10000/2025-05-15_08-28/Dataset/UEA/checkpoints/regressionmodel_last.pth'
# best_model_path = r'./Results/regression_(530, 480, 810, 490)_10000/2025-05-15_05-31/Dataset/UEA/checkpoints/regressionmodel_last.pth'
best_model_path = r'./Results/regression-glycerine_(530, 270, 810, 540)_10000/2025-05-15_12-16/Dataset/UEA/checkpoints/regression-glycerinemodel_last.pth'
# THIS IS FOR ROI EXP
SELECTED_AREAS = [
    # (530, 200, 810, 210), # y_division_val = 10, 1/54
    # (530, 480, 810, 490), # y_division_val = 10, 1/54
    # (530, 460, 810, 540), # y_division_val = 80, 1/6.75
    (530, 270, 810, 540),  # y_division_val = 270, 1/2
    # (530, -1, 810, 600) # y_division_val = inf
    ]
DELTA_T_LIST = [
    # 1000, # 1/8000
    # 3000, # 1/2700
    10000, # 1/800
    # 30000, # 1/270
    # 100000 # 1/80
]
SELECTED_AREA = SELECTED_AREAS[0]
DELTA_T = DELTA_T_LIST[0]

ARCHIVE_PATH = '/data/cls1-srv5-pool/Chem_liquid/'

selected_model = 'ConvTran'

# dataset_name = 'regression'
dataset_name = 'regression-glycerine'
# dataset_name = 'classification'
# dataset_name = 'all'
problem = 'regression-glycerine'

_dataset_info_dict = {
    'regression':{
        'dataset_path' : os.path.join(ARCHIVE_PATH, 'regression/processed/'),
        'folder_labels_dict' : {
            'water_H1-50':      0.0,
            'ethanol_10_H1-50': 0.1,
            'ethanol_20_H1-50': 0.2,
            'ethanol_30_H1-50': 0.3,
            'ethanol_40_H1-50': 0.4,
            'ethanol_50_H1-50': 0.5,
            'ethanol_60_H1-50': 0.6,
            'ethanol_70_H1-50': 0.7,
            'ethanol_80_H1-50': 0.8,
            'ethanol_90_H1-50': 0.9,
            'ethanol_100_H1-50':    1.0,
        },
        'folder_labels_dict_ex':{
            'ethanol_20_H1-50-T2':  0.2,
        }
    },
    
    'regression-glycerine':{
        'dataset_path' : os.path.join(ARCHIVE_PATH, 'classification/processed/'),
        'folder_labels_dict' : {
            'water_H1-50':      0.0,
            'C3H8O3_10_H1-50':      0.1, # 甘油
            'C3H8O3_20_H1-50':      0.2, # 甘油
            'C3H8O3_30_H1-50':      0.3, # 甘油
            'C3H8O3_40_H1-50-B':    0.4, # 甘油
        },
    },
    
    'classification':{
        'dataset_path' : os.path.join(ARCHIVE_PATH, 'classification/processed/'),
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
        }
    },
    'all':{
        'dataset_path' : os.path.join(ARCHIVE_PATH, 'classification/processed/'),
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
            
            'DMSO-50':              11, # DMSO
            'EtOAc_H1-50':          12, # 乙酸乙酯
            # 'CH3COOCH2CH3_H1-50':   15, # 乙酸乙酯
            'BnOH_H1-50':           13, # 苯甲醇
            # 'PhCO_H1-50':           17, # 苯甲醇
            'iPrOH_H1-50':          14, # 异丙醇
            # 'CC(O)C_H1-50':         15, # 异丙醇
            'C6H14_H1-50-spill-manual': 15, # 正己烷
            'NaOH_1M_H1-50':        16, # NaOH
            
            'alcohol_56v_H1-50':    17, # 白酒
            'alcohol_15v_H1-50':    18, # 清酒
            'alcohol_vodka_40v_H1-50': 19, # 伏特加
            'vinegar_9_H1-50':      20, # 醋
            'medical_alcohol_75v_H1-50': 21, # 医用酒精
            'coconut_water_H1-50':  22, # 椰子水
            'NaCl_1M_H1-50':        23, # NaCl
            'NaCl_2M_H1-50':        24, # NaCl
            # 'NaCl_10_H1-50':        27,
            'glucose_10_H1-50':     25, # 葡萄糖
            'NaHCO3_0.5M_H1-50':    26, # NaHCO3
            
            'C3H8O3_10_H1-50':      27, # 甘油
            'C3H8O3_20_H1-50':      28, # 甘油
            'C3H8O3_30_H1-50':      29, # 甘油
            'C3H8O3_40_H1-50-B':    30, # 甘油
            
        },
        'folder_labels_dict_ex':{
            'ethanol_20_H1-50-T2':  31,
        }
    },
}

dataset_path = _dataset_info_dict[dataset_name]['dataset_path']
folder_labels_dict = _dataset_info_dict[dataset_name]['folder_labels_dict']
folder_labels_dict_ex = _dataset_info_dict[dataset_name]['folder_labels_dict_ex'] if 'folder_labels_dict_ex' in _dataset_info_dict[dataset_name] else {}

logger = logging.getLogger('__main__')
parser = argparse.ArgumentParser()
# -------------------------------------------- Input and Output --------------------------------------------------------
parser.add_argument('--data_path', default='Dataset/UEA/', choices={'Dataset/UEA/', 'Dataset/Segmentation/'},
                    help='Data path')
parser.add_argument('--output_dir', default='Results',
                    help='Root output directory. Must exist. Time-stamped directories will be created inside.')
parser.add_argument('--Norm', type=bool, default=False, help='Data Normalization')
parser.add_argument('--val_ratio', type=float, default=0.2, help="Proportion of the train-set to be used as validation")
parser.add_argument('--print_interval', type=int, default=10, help='Print batch info every this many batches')
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------- Model Parameter and Hyperparameter ---------------------------------------------
parser.add_argument('--Net_Type', default=['C-T'], choices={'T', 'C-T'}, help="Network Architecture. Convolution (C)"
                                                                              "Transformers (T)")
# Transformers Parameters ------------------------------
parser.add_argument('--emb_size', type=int, default=16, help='Internal dimension of transformer embeddings')
parser.add_argument('--dim_ff', type=int, default=256, help='Dimension of dense feedforward part of transformer layer')
parser.add_argument('--num_heads', type=int, default=8, help='Number of multi-headed attention heads')
parser.add_argument('--Fix_pos_encode', choices={'tAPE', 'Learn', 'None'}, default='tAPE',
                    help='Fix Position Embedding')
parser.add_argument('--Rel_pos_encode', choices={'eRPE', 'Vector', 'None'}, default='eRPE',
                    help='Relative Position Embedding')
# Training Parameters/ Hyper-Parameters ----------------
parser.add_argument('--epochs', type=int, default=400, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.01, help='Droupout regularization ratio')
parser.add_argument('--val_interval', type=int, default=2, help='Evaluate on validation every XX epochs. Must be >= 1')
parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='loss',
                    help='Metric used for defining best epoch')
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ System --------------------------------------------------------
parser.add_argument('--gpu', type=int, default=cuda_device, help='GPU index, -1 for CPU')
parser.add_argument('--console', action='store_true', help="Optimize printout for console output; otherwise for file")
parser.add_argument('--seed', default=42, type=int, help='Seed used for splitting sets') # seed=1234: loss=5e-5
args = parser.parse_args()

def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a

def align_ev_freq_series_lengths(ev_freq_series_list, err_description):
    ev_freq_series_len_list = [len(ev_freq) for ev_freq in ev_freq_series_list]
    if((max(ev_freq_series_len_list) - min(ev_freq_series_len_list) ) / np.mean(ev_freq_series_len_list) > 0.1):
        print(f'{err_description} ev_freq_series_lengths vary too much')
        print(f'max: {max(ev_freq_series_len_list)}, min: {min(ev_freq_series_len_list)}, mean: {np.mean(ev_freq_series_len_list)}')
        raise ValueError('ev_freq_series_lengths vary too much')
    
    # 将所有ev_freq_series_list中的元素长度统一
    max_len = max(ev_freq_series_len_list)
    for i in range(len(ev_freq_series_list)):
        if len(ev_freq_series_list[i]) < max_len:
            # 使用 np.pad 在数组末尾填充 0
            ev_freq_series_list[i] = np.pad(ev_freq_series_list[i], (0, max_len - len(ev_freq_series_list[i])), mode='constant', constant_values=0)
            
def data_augmentation(original_data, original_labels, augment_times=10, noise_level=0.05):
    """
    序列数据增强函数
    参数：
        original_data: 原始数据，形状 (n_samples, n_features, seq_length)
        original_labels: 对应标签，形状 (n_samples,)
        augment_times: 需要增强的倍数
        noise_level: 噪声级别
    返回：
        增强后的数据和标签 (n_samples * augment_times, n_features, seq_length)
    """
    # 按类别分组数据
    class_dict = {}
    for data, label in zip(original_data, original_labels):
        if label not in class_dict:
            class_dict[label] = []
        class_dict[label].append(data)
    
    augmented_data = []
    augmented_labels = []
    
    # 对每个类别单独处理
    for label, class_data in class_dict.items():
        class_data = np.array(class_data)
        # n_samples, n_features, seq_len = class_data.shape
        n_samples, seq_len = class_data.shape
        required_samples = n_samples * augment_times
        
        # 插值增强
        interp_data = []
        while len(interp_data) < required_samples:
            # 随机选择两个不同样本
            idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
            sample1 = class_data[idx1]
            sample2 = class_data[idx2]
            
            # 生成随机theta矩阵 (每个时间步、每个特征单独的theta)
            # theta = np.random.rand(n_features, seq_len)  # 每个点的独立theta
            theta = np.random.rand(seq_len)  # 每个点的独立theta
            
            # 线性插值
            new_sample = theta * sample1 + (1 - theta) * sample2
            interp_data.append(new_sample)
        
        interp_data = np.array(interp_data[:required_samples])
        
        # 添加噪声
        noise = np.random.normal(scale=noise_level, size=interp_data.shape)
        interp_data += noise
        
        augmented_data.append(interp_data)
        augmented_labels.extend([label]*required_samples)
    
    # 合并所有类别
    augmented_data = np.concatenate(augmented_data, axis=0)
    augmented_labels = np.array(augmented_labels)
    
    # 合并原始数据（可选，根据需求）
    augmented_data = np.concatenate([original_data, augmented_data], axis=0)
    augmented_labels = np.concatenate([original_labels, augmented_labels], axis=0)
    
    return shuffle(augmented_data, augmented_labels, random_state=42)

def plot_original_vs_augmented(original_data, original_labels, augmented_data, augmented_labels, label, sample_index=0):
    """
    绘制原始数据和增强后的数据对比图。

    Args:
        original_data (np.ndarray): 原始数据，形状为 (num_samples, num_features)
        augmented_data (np.ndarray): 增强后的数据，形状为 (num_samples * augment_times, num_features)
        label (float): 要绘制的标签值
        sample_index (int): 要绘制的样本索引，默认为0
    """
    # 找到对应标签的原始数据和增强数据
    original_samples = original_data[original_labels == label]
    augmented_samples = augmented_data[augmented_labels == label]

    # 选择第一个样本进行绘制
    original_sample = original_samples[sample_index]
    augmented_sample = augmented_samples[sample_index]

    # 绘制原始数据和增强数据
    plt.figure(figsize=(12, 6))

    # 原始数据
    plt.subplot(1, 2, 1)
    plt.plot(original_sample, label='Original Data')
    plt.title(f'Original Data (Label: {label})')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()

    # 增强数据
    plt.subplot(1, 2, 2)
    plt.plot(augmented_sample, label='Augmented Data', color='orange')
    plt.title(f'Augmented Data (Label: {label})')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    # plt.savefig(f'./Results/{dataset_name}_original_vs_augmented_{label}.png')
    plt.show()

def my_TSC_data_loader(dataset_path, folder_labels_dict, selected_areas, delta_t):
    # RAW_FOLDERS = ['water_H1-50', 'ethanol_20_H1-50', 'ethanol_40_H1-50', 'ethanol_60_H1-50', 'ethanol_80_H1-50', 'ethanol_100_H1-50']
    # 不进行标签编码
    
    train_ev_freq_series_list = []
    test_ev_freq_series_list = []
    train_label_list = []
    test_label_list = []
    
    for dataset_name, label in folder_labels_dict.items():
        for select_area in selected_areas:
            # 加载数据
            series_list_this_dataset = np.load(os.path.join(dataset_path, f'ev_freq_series_list_{dataset_name}_select_{select_area}_t{delta_t}.npy'))
            
            # 将数据分为训练集和测试集
            train_data = series_list_this_dataset[:40]  # 前40个样本用于训练
            test_data = series_list_this_dataset[40:]   # 后10个样本用于测试
            
            # print(f'dataset_name: {dataset_name}, select_area: {select_area}, label: {label}')
            # 添加到训练集和测试集
            train_ev_freq_series_list.extend(train_data)
            test_ev_freq_series_list.extend(test_data)
            train_label_list.extend([[label] for _ in range(train_data.shape[0])])
            test_label_list.extend([[label] for _ in range(test_data.shape[0])])
            
            # print(f'len(train_ev_freq_series_list): {len(train_ev_freq_series_list)}')
            # print(f'len(test_ev_freq_series_list): {len(test_ev_freq_series_list)}')
    
    # 统一训练集和测试集的长度
    align_ev_freq_series_lengths(train_ev_freq_series_list, f'{dataset_path}_train')
    align_ev_freq_series_lengths(test_ev_freq_series_list, f'{dataset_path}_test')

    # 转换为 NumPy 数组
    train_ev_freq_series_list = np.array(train_ev_freq_series_list, dtype=np.float32)
    test_ev_freq_series_list = np.array(test_ev_freq_series_list, dtype=np.float32)
    train_label_list = np.array(train_label_list, dtype=np.float32)
    test_label_list = np.array(test_label_list, dtype=np.float32)
    
    train_label_list = np.squeeze(train_label_list, axis=1)
    test_label_list = np.squeeze(test_label_list, axis=1)
    
    # if len(train_ev_freq_series_list) > 0:
    #     # 调整输入维度 (假设原始数据为 2D, 添加序列维度)
    #     if train_ev_freq_series_list.ndim == 2:
    #         train_ev_freq_series_list = train_ev_freq_series_list[:, np.newaxis, :]  # 添加通道维度
    #         test_ev_freq_series_list = test_ev_freq_series_list[:, np.newaxis, :]  # 添加通道维度
        
    
    print('finished loading data')
    print(f'train_ev_freq_series_list.shape: {train_ev_freq_series_list.shape}')
    print(f'test_ev_freq_series_list.shape: {test_ev_freq_series_list.shape}')
    print(f'train_label_list.shape: {train_label_list.shape}')
    print(f'test_label_list.shape: {test_label_list.shape}')
    
    print(f'test_data_list: {np.unique(test_label_list)}')
    print(f'train_data_list: {np.unique(train_label_list)}')
    
    
    # 数据增强
    augmented_data, augmented_labels = data_augmentation(
        train_ev_freq_series_list, 
        train_label_list,
        augment_times=10,
        noise_level=0.05
    )
        
    # 选择一个标签进行绘制，例如 label = 0.2
    # label_to_plot = 0.2
    # plot_original_vs_augmented(train_ev_freq_series_list, train_label_list, augmented_data, augmented_labels, label_to_plot)

    # 替换原始数据
    train_ev_freq_series_list = augmented_data
    train_label_list = augmented_labels
    
    print(f'after data augmentation, train_ev_freq_series_list.shape: {train_ev_freq_series_list.shape}')
    print(f'after data augmentation, train_label_list.shape: {train_label_list.shape}')
    
    print(f'after data augmentation, train_data_list: {np.unique(train_label_list)}')
    print(f'after data augmentation, test_data_list: {np.unique(test_label_list)}')
    
    # return train_ev_freq_series_list, train_label_list, test_ev_freq_series_list, test_label_list
    
    # le = preprocessing.LabelEncoder()
    # le.fit(np.squeeze(train_label_list, axis=1))
    # train_label_list = le.transform(np.squeeze(train_label_list, axis=1))
    # le.fit(np.squeeze(test_label_list, axis=1))
    # test_label_list = le.transform(np.squeeze(test_label_list, axis=1))
    # print(f'train_label_list: {train_label_list}')
    # print(f'test_label_list: {test_label_list}')
    
    # raise ValueError('stop here')
    return set_nan_to_zero(train_ev_freq_series_list), train_label_list, set_nan_to_zero(test_ev_freq_series_list), test_label_list

def simple_TSC_data_loader(dataset_path, folder_labels_dict, selected_areas, delta_t):
    """不区分训练集和测试集、不进行label编码"""  
    
    if folder_labels_dict is None:
        return None, None
    
    ev_freq_series_list = []
    label_list = []
    
    for dataset_name, label in folder_labels_dict.items():
        for select_area in selected_areas:
            # 加载数据
            series_list_this_dataset = np.load(os.path.join(dataset_path, f'ev_freq_series_list_{dataset_name}_select_{select_area}_t{delta_t}.npy'))
            
            data = series_list_this_dataset
            
            # 添加到训练集和测试集
            ev_freq_series_list.extend(data)
            label_list.extend([[label] for _ in range(data.shape[0])])
    
    # 统一训练集和测试集的长度
    align_ev_freq_series_lengths(ev_freq_series_list, f'{dataset_path}')
    
    # 转换为 NumPy 数组
    ev_freq_series_list = np.array(ev_freq_series_list, dtype=np.float32)
    label_list = np.array(label_list, dtype=np.float32)
    
    print('finished loading data')
    print(f'ev_freq_series_list.shape: {ev_freq_series_list.shape}')
    print(f'label_list.shape: {label_list.shape}')
    
    # 直接使用 np.squeeze 将 label_list 从 (samplesize, 1) 转换为一维 (samplesize,)
    label_list = np.squeeze(label_list, axis=1)
    print(f'label_list: {label_list}')
    return set_nan_to_zero(ev_freq_series_list), label_list

if __name__ == '__main__':
    """
    主训练流程函数，负责整个模型的训练流程控制
    
    流程包括：
    1. 配置初始化
    2. 设备初始化
    3. 数据验证和加载
    4. 模型构建和训练
    5. 结果保存
    """
    
    config = Setup(args)  # ⭐ 初始化配置字典，包含所有训练参数
    initial_timestamp = datetime.now()
    
    output_dir = f'./Results/{problem}_{SELECTED_AREA}_{DELTA_T}/{initial_timestamp.strftime("%Y-%m-%d_%H-%M")}'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_dir = os.path.join(output_dir, config['data_path'])
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    
    device = Initialization(config)
    Data_Verifier(config)  # Download the UEA and HAR datasets if they are not in the directory
    All_Results = ['Datasets', 'ConvTran']  # Use to store the accuracy of ConvTran in e.g "Result/Datasets/UEA"

    """
    对数据目录中的每个数据集进行训练
    
    Args:
        problem: 数据集名称
    """
    print(text2art(problem, font='small'))
    
    # ------------------------------------ Load Data ---------------------------------------------------------------
    logger.info("Loading Data ...")
    Data = {}
    
    selected_area_list = [SELECTED_AREA]

    # 移除了测试集的加载
    _, _, X_val, y_val = my_TSC_data_loader(dataset_path, folder_labels_dict, selected_area_list, DELTA_T)
    # 合并 X_train 和 X_val
    # X_combined = np.concatenate((X_train, X_val), axis=0)
    # y_combined = np.concatenate((y_train, y_val), axis=0)
    # X_val = X_combined
    # y_val = y_combined
    # 增加维度保持不变
    # X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    
    # Data['train_data'] = X_train
    # Data['train_label'] = y_train
    Data['val_data'] = X_val
    Data['val_label'] = y_val  # 移除了测试集相关字段
    
    # 仅创建训练和验证数据集
    # train_dataset = dataset_class_regr(Data['train_data'], Data['train_label'])
    val_dataset = dataset_class_regr(Data['val_data'], Data['val_label'])

    # 移除了测试集的打印
    # print(f"train_dataset: {Data['train_data']}")
    print(f"val_dataset: {Data['val_data']}")  
    # print(f"train_dataset.shape: {Data['train_data'].shape}")
    print(f"val_dataset.shape: {Data['val_data'].shape}")

    # 仅创建训练和验证的DataLoader
    # train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    
    print(f'Val_loader #########################################################')
    
    for i, (data, label, index) in enumerate(val_loader):
        print(f'data: {data}')
        print(f'label: {label}')
        print(f'index: {index}')
        break
    
    # --------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- Build Model -----------------------------------------------------
    dic_position_results = [problem]  # 直接使用数据集名称 problem

    logger.info("Creating model ...")
    config['Data_shape'] = Data['val_data'].shape
    config['num_labels'] = 1 # int(max(Data['train_label']))+1
    model = model_factory(config)  # ⭐ 根据配置创建模型实例
    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(count_parameters(model)))
    # -------------------------------------------- Model Initialization ------------------------------------
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(model.parameters(), lr=config['lr'], weight_decay=0)
    config['loss_module'] = get_loss_module()
    save_path = os.path.join(config['save_dir'], problem + 'model_{}.pth'.format('last'))
    tensorboard_writer = SummaryWriter('summary')
    model.to(device)
    # ---------------------------------------------- Training The Model ------------------------------------
    # logger.info('Starting training...')
    # trainer = SupervisedTrainer(model, train_loader, device, config['loss_module'], config['optimizer'], l2_reg=0,
    #                             print_interval=config['print_interval'], console=config['console'], print_conf_mat=False)
    # val_evaluator = SupervisedTrainer(model, val_loader, device, config['loss_module'],
    #                                     print_interval=config['print_interval'], console=config['console'],
    #                                     print_conf_mat=False)

    # train_runner(config, model, trainer, val_evaluator, save_path)
    
    # 加载最佳模型并在验证集上最终评估
    best_model, optimizer, start_epoch = load_model(model, best_model_path, config['optimizer'])
    best_model.to(device)

    # 使用验证集代替测试集进行最终评估
    best_val_evaluator = SupervisedTrainer(best_model, val_loader, device, config['loss_module'],
                                            print_interval=config['print_interval'], console=config['console'],
                                            print_conf_mat=True)
    best_aggr_metrics_val, all_metrics, pred, targets = best_val_evaluator.evaluate(keep_all=True, return_pred_targets=True)
    
    print(f'pred: {pred}')
    print(f'targets: {targets}')
    
    # 将pred和targets转换为numpy数组
    pred = np.array(pred, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    # 保存pred和targets
    best_output_dir = os.path.join(f'./Results/{problem}_{SELECTED_AREA}_{DELTA_T}/', 'best_output')
    print(f'#########################################################')
    print(f'saved to best_output_dir: {best_output_dir}')
    if not os.path.isdir(best_output_dir):
        os.makedirs(best_output_dir)
    np.save(os.path.join(best_output_dir, 'pred.npy'), pred)
    np.save(os.path.join(best_output_dir, 'targets.npy'), targets)
    
    # 修改输出信息为验证集结果
    print_str = 'Best Model Validation Summary: '
    for k, v in best_aggr_metrics_val.items():
        print_str += '{}: {} | '.format(k, v)
    print(print_str)
    
    # 保存验证集结果
    dic_position_results.append(all_metrics['mse'])
    problem_df = pd.DataFrame(dic_position_results)
    problem_df.to_csv(os.path.join(config['pred_dir'] + '/' + problem + '.csv'))

    All_Results = np.vstack((All_Results, dic_position_results))

    All_Results_df = pd.DataFrame(All_Results)
    All_Results_df.to_csv(os.path.join(config['output_dir'], 'ConvTran_Results.csv'))  # ⭐ 保存所有问题的汇总结果
    print(f'{tag_this_file}')