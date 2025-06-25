from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from utils.utils import generate_results_csv, read_dataset_torch
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import os
import numpy as np
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device = "cuda:2"

# 定义选择区域和 delta_t
SELECTED_AREAS = [
    # (530, 200, 810, 210), # y_division_val = 10, 1/54
    # (530, 480, 810, 490), # y_division_val = 10, 1/54
    # (530, 460, 810, 540), # y_division_val = 80, 1/6.75
    (530, 270, 810, 540),  # y_division_val = 270, 1/2
    # (530, -1, 810, 600) # y_division_val = inf
    ]
DELTA_T_LIST = [
    1000, # 1/8000
    # 3000, # 1/2700
    # 10000, # 1/800
    # 30000, # 1/270
    # 100000 # 1/80
]


# 定义 y_division_val 和 t_division_val
y_division_val = ['1/54', '1/6.75', '1/2', '1']
t_division_val = ['1000', '3000', '10000', '30000']


ARCHIVE_PATH = '/data/cls1-srv5-pool/Chem_liquid/'

dataset_name = 'regression'
# dataset_name = 'classification'
# dataset_name = 'all'

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
folder_labels_dict_ex = _dataset_info_dict[dataset_name]['folder_labels_dict_ex']

#########################################################

def fit_classifier(x_train, y_train, x_test, y_test, \
    classifier_name, output_directory,\
        epochs=600, y_labels_description=None, device = 'cuda'):
    """
    训练分类器模型，包括数据预处理和模型训练全过程。

    处理步骤：
    1. 从全局数据集字典获取训练/测试数据
    2. 对标签进行one-hot编码
    3. 调整数据维度（单变量转多变量）
    4. 创建并训练指定分类器

    注意：函数依赖外部变量 datasets_dict, dataset_name, classifier_name, output_directory
    """
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()  # ⭐ 将标签转换为one-hot编码形式
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))  # ⭐ 单变量数据转为多变量格式
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    print(f'input_shape: {input_shape}')
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory, device)

    classifier.fit(x_train, y_train, x_test, y_test, y_true, nb_epochs=epochs, y_labels_description=y_labels_description)  # ⭐ 核心训练过程


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True, device='cuda'):
    """
    根据指定的分类器名称创建对应的分类器实例。

    Args:
        classifier_name (str): 分类器名称，支持'fcn'、'mlp'、'resnet'等
        input_shape (tuple): 输入数据的形状
        nb_classes (int): 类别数量
        output_directory (str): 输出目录路径
        verbose (bool, optional): 是否显示详细信息，默认为True

    Returns:
        object: 对应的分类器实例
    """
    if classifier_name == 'fcn_torch':
        from classifiers import fcn_torch
        return fcn_torch.Classifier_FCN_Torch(output_directory, input_shape, nb_classes, verbose)  # ⭐ 创建FCN分类器实例
    if classifier_name == 'cnn_torch':
        from classifiers import cnn_torch
        return cnn_torch.Classifier_CNN_Torch(output_directory, input_shape, nb_classes, verbose)  # ⭐ 创建CNN分类器实例
    if classifier_name == 'lstm_torch':
        from classifiers import lstm_torch
        return lstm_torch.Classifier_CNN_LSTM_Torch(output_directory, input_shape, nb_classes, verbose, device)  # ⭐ 创建LSTM分类器实例
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)


############################################### main

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
    
    classifier_name = 'lstm_torch'
    
    # archive_name = 'UCRArchive_2018'
    # ARCHIVE_PATH = './archives/UCRArchive_2018'
    # dataset_name = 'FiftyWords'
    # DATASET_PATH = ARCHIVE_PATH + '/' + dataset_name
    
    # output_directory = './results/' + classifier_name + '/' + archive_name + '/' + \
    #                 dataset_name + '/'
                    
    archive_name = 'chem-liquid-classification'

    SELECTED_AREAS = [SELECTED_AREAS[0]]
    DELTA_T = DELTA_T_LIST[0]
    
    output_directory = './results/' + classifier_name + '/' + archive_name + '/' + \
                       dataset_name + '/' + str(SELECTED_AREAS[0]) + str(DELTA_T) + '/'
    
    print('Method: ', archive_name, dataset_name, classifier_name)
    
    create_directory(output_directory)
    # x_train, y_train, x_test, y_test = read_dataset_torch(dataset_path, archive_name, dataset_name, folder_labels_dict=folder_labels_dict, selected_areas=SELECTED_AREAS, delta_t=DELTA_T)

    x_train, y_train, x_test, y_test = my_TSC_data_loader(
        dataset_path, folder_labels_dict, SELECTED_AREAS, DELTA_T
    )
    
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    
    fit_classifier(x_train, y_train, x_test, y_test, \
        classifier_name, output_directory,\
            epochs=400, y_labels_description=list(folder_labels_dict.keys()), device=device)

    print('DONE')

    # the creation of this directory means
    create_directory(output_directory + '/DONE')
