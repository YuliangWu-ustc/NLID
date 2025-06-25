from sklearn import preprocessing
from PIL import Image
from sklearn.preprocessing import minmax_scale
import numpy as np
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a


def check_dataset(sorted_label_data):
    #check sort value and check number of each number
    label = sorted_label_data[:,0:1]
    Nor_data = minmax_scale(sorted_label_data[:,1:])
    Nor_label = minmax_scale(sorted_label_data[:,0:1])
    print(Nor_label.shape)
    biglabel = np.repeat(Nor_label, 80, axis=1)
    big = np.concatenate((biglabel, Nor_data),axis=1)
    img = Image.fromarray(big*255)
    img.show()
    unique, counts = np.unique(label, return_counts=True)
    print(dict(zip(unique, counts)))

def trim_lable(dataset):
    data = dataset[:,1:]
    label = dataset[:,0:1]
    le = preprocessing.LabelEncoder()
    le.fit(np.squeeze(label, axis=1))
    label = le.transform(np.squeeze(label, axis=1))
    label = np.expand_dims(label,axis =1)
    dataset = np.concatenate((label, data), axis=1)
    return dataset

def split_by_class(sorted_label_data,train_list):
    train_index = []
    test_index = []
    for i in range(sorted_label_data.shape[0]):
        if sorted_label_data[i,0] in train_list:
            train_index.append(i)
        else:
            test_index.append(i)
    train_dataset = sorted_label_data[train_index,:]
    test_dataset = sorted_label_data[test_index,:]

    return trim_lable(train_dataset), trim_lable(test_dataset)


def normal_datset_to_few_shot_dataset(X_train, y_train, X_test, y_test,train_ratio=0.8, seed=None):
    #biuld big dataset

    data = np.concatenate((X_train,X_test))
    label = np.concatenate((y_train,y_test))
    label = np.expand_dims(label,axis =1 )
    label_data = np.concatenate((label, data),axis=1)

    # sample classes
    n_class = np.amax(label)+1
    sorted_label_data = label_data[label[:, 0].argsort()]
    if seed == None:
        list = np.random.choice(n_class, n_class, replace=False)
    else:
        np.random.seed(seed=seed)
        list = np.random.choice(n_class, n_class, replace=False)
    train_list = list[0:int(n_class*train_ratio)]

    # check_dataset(sorted_label_data)

    #split dataset
    train_dataset, test_dataset = split_by_class(sorted_label_data, train_list)

    return train_dataset, test_dataset, train_list

def normal_datset_to_few_shot_dataset_with_list(X_train, y_train, X_test, y_test,train_list):

    data = np.concatenate((X_train,X_test))
    label = np.concatenate((y_train,y_test))
    label = np.expand_dims(label,axis =1 )
    label_data = np.concatenate((label, data),axis=1)

    sorted_label_data = label_data[label[:, 0].argsort()]

    train_dataset, test_dataset = split_by_class(sorted_label_data, train_list)

    return train_dataset, test_dataset, train_list

def fill_out_with_Nan(data,max_length):
    #via this it can works on more dimensional array
    pad_length = max_length-data.shape[-1]
    if pad_length == 0:
        return data
    else:
        pad_shape = list(data.shape[:-1])
        pad_shape.append(pad_length)
        Nan_pad = np.empty(pad_shape)*np.nan
        return np.concatenate((data, Nan_pad), axis=-1)
    

def get_label_dict(file_path):
    label_dict ={}
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            if '@classLabel' in line:
                label_list = line.replace('\n','').split(' ')[2:]
                for i in range(len(label_list)):
                    label_dict[label_list[i]] = i 
                
                break
    return label_dict


def get_data_and_label_from_ts_file(file_path,label_dict):
    with open(file_path) as file:
        lines = file.readlines()
        Start_reading_data = False
        Label_list = []
        Data_list = []
        max_length = 0
        for line in lines:
            if Start_reading_data == False:
                if '@data'in line:
                    Start_reading_data = True
            else:
                temp = line.split(':')
                Label_list.append(label_dict[temp[-1].replace('\n','')])
                data_tuple= [np.expand_dims(np.fromstring(channel, sep=','), axis=0) for channel in temp[:-1]]
                max_channel_length = 0
                for channel_data in data_tuple:
                    if channel_data.shape[-1]>max_channel_length:
                        max_channel_length = channel_data.shape[-1]
                data_tuple = [fill_out_with_Nan(data,max_channel_length) for data in data_tuple]
                data = np.expand_dims(np.concatenate(data_tuple, axis=0), axis=0)
                Data_list.append(data)
                if max_channel_length>max_length:
                    max_length = max_channel_length
        
        Data_list = [fill_out_with_Nan(data,max_length) for data in Data_list]
        X =  np.concatenate(Data_list, axis=0)
        Y =  np.asarray(Label_list)
        
        return np.float32(X), Y

    
import scipy.io as sio

def get_from_X(X):
    data_list = []
    max_length = 0
    for data in X[0][0][0][:]:
        data = np.expand_dims(data,0)
        data_list.append(data)
        max_channel_length = data.shape[-1]
        if max_channel_length>max_length:
            max_length = max_channel_length        
    Data_list = [fill_out_with_Nan(data,max_length) for data in data_list]
    X =  np.concatenate(Data_list, axis=0)
    return np.float32(X)

def get_from_Y(y):
    y = y[0][0].flatten()
    return np.int64(y)



def TSC_multivariate_data_loader_from_mat(dataset_path, dataset_name):
    full_path = dataset_path+'/'+dataset_name+'/'+dataset_name+'.mat'
    mat_contents = sio.loadmat(full_path)
    X_train_raw = mat_contents['mts']['train']
    y_train_raw = mat_contents['mts']['trainlabels']
    X_test_raw =mat_contents['mts']['test']
    y_test_raw = mat_contents['mts']['testlabels']
    X_train = get_from_X(X_train_raw)
    y_train = get_from_Y(y_train_raw)
    X_test = get_from_X(X_test_raw)
    y_test = get_from_Y(y_test_raw)
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train  = le.transform(y_train)
    y_test  = le.transform(y_test)
    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test




def TSC_multivariate_data_loader(dataset_path, dataset_name):
    
    Train_dataset_path = dataset_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN.ts'
    Test_dataset_path = dataset_path + '/' + dataset_name + '/' + dataset_name + '_TEST.ts'
    label_dict = get_label_dict(Train_dataset_path)
    X_train, y_train = get_data_and_label_from_ts_file(Train_dataset_path,label_dict)
    X_test, y_test = get_data_and_label_from_ts_file(Test_dataset_path,label_dict)
    
    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test


def TSC_data_loader(dataset_path,dataset_name):
    Train_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN.tsv')
    Test_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TEST.tsv')
    Train_dataset = Train_dataset.astype(np.float32)
    Test_dataset = Test_dataset.astype(np.float32)

    X_train = Train_dataset[:, 1:]
    y_train = Train_dataset[:, 0:1]

    X_test = Test_dataset[:, 1:]
    y_test = Test_dataset[:, 0:1]
    print(f'X_train.shape: {X_train.shape}')
    print(f'y_train.shape: {y_train.shape}')
    print(f'X_test.shape: {X_test.shape}')
    print(f'y_test.shape: {y_test.shape}')
    le = preprocessing.LabelEncoder()
    le.fit(np.squeeze(y_train, axis=1))
    y_train = le.transform(np.squeeze(y_train, axis=1))
    y_test = le.transform(np.squeeze(y_test, axis=1))
    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test

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
    label_list = np.array(label_list, dtype=np.int64)
    
    print('finished loading data')
    print(f'ev_freq_series_list.shape: {ev_freq_series_list.shape}')
    print(f'label_list.shape: {label_list.shape}')
    
    # 直接使用 np.squeeze 将 label_list 从 (samplesize, 1) 转换为一维 (samplesize,)
    label_list = np.squeeze(label_list, axis=1)
    print(f'label_list: {label_list}')
    return set_nan_to_zero(ev_freq_series_list), label_list

def check_normalized(X_train,X_test,dataset_name):
    mean_of_feature_cols_train = np.nanmean(X_train, axis=1, keepdims= True)
    std_of_feature_cols_train = np.nanstd(X_train, axis=1, keepdims= True)
    if np.nanmean(abs(mean_of_feature_cols_train)) < 1e-7 and abs(np.nanmean(std_of_feature_cols_train)-1) < 0.05 :
        return X_train,X_test
    else:
        print(dataset_name,"is not normalized, let's do it")
        print('mean = ',np.nanmean(mean_of_feature_cols_train), 'std = ',np.nanmean(std_of_feature_cols_train))
        mean_of_feature_cols_test = np.nanmean(X_test, axis=1, keepdims= True)
        std_of_feature_cols_train = np.nanstd(X_train, axis=1, keepdims= True)
        std_of_feature_cols_test = np.nanstd(X_test, axis=1, keepdims= True)
        X_train = (X_train -mean_of_feature_cols_train)/std_of_feature_cols_train
        X_test = (X_test -mean_of_feature_cols_test)/std_of_feature_cols_test
        return X_train, X_test
    

def TSC_data_loader_with_z_normaliz_check(dataset_path,dataset_name):
    Train_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TRAIN.tsv')
    Test_dataset = np.loadtxt(
        dataset_path + '/' + dataset_name + '/' + dataset_name + '_TEST.tsv')
    Train_dataset = Train_dataset.astype(np.float32)
    Test_dataset = Test_dataset.astype(np.float32)

    X_train = Train_dataset[:, 1:]
    y_train = Train_dataset[:, 0:1]

    X_test = Test_dataset[:, 1:]
    y_test = Test_dataset[:, 0:1]
    le = preprocessing.LabelEncoder()
    le.fit(np.squeeze(y_train, axis=1))
    y_train = le.transform(np.squeeze(y_train, axis=1))
    y_test = le.transform(np.squeeze(y_test, axis=1))
    
    
    X_train,X_test = check_normalized(X_train,X_test,dataset_name)
    
    return set_nan_to_zero(X_train), y_train, set_nan_to_zero(X_test), y_test

################# UCI
from scipy.io import arff
import pandas as pd

def formate_data_to_classic_style(data):
    # replace clase label to number 
    label_set= np.unique(data[:,-1])
    class_number = 0
    for label in label_set:
        data = np.where(data == label ,class_number, data)
        class_number = class_number + 1
    data = data.astype('float32')
    data_value = data[:,:-1]
    label = data[:,-1].astype('int64')
    # remove data index
    return data_value,label

def check_normalized(data):
    mean_of_feature_cols_data = np.nanmean(data, axis=1, keepdims= True)
    std_of_feature_cols_data = np.nanstd(data, axis=1, keepdims= True)
    if np.nanmean(abs(mean_of_feature_cols_data)) < 1e-7 and abs(np.nanmean(std_of_feature_cols_data)-1) < 0.05 :
        print("The dataset is already z-normalized")
        return data
    else:
        print("The dataset is not z-normalized, let's do it")
        print('mean = ',np.nanmean(mean_of_feature_cols_data), 'std = ',np.nanmean(std_of_feature_cols_data))
        data = (data -mean_of_feature_cols_data)/std_of_feature_cols_data
        return data


def UCI_data_loader(dataset_path,dataset_name):
    data = arff.loadarff(dataset_path+'/'+dataset_name+'.arff')
    df = pd.DataFrame(data[0])
    df.head()
    temp = df.to_numpy()
    data, label = formate_data_to_classic_style(temp)
    data = check_normalized(data)
    return data, label

################# MITDB


def MITdb_data_loader(root_path):
    train_dataset = np.load(os.path.join(root_path, 'train_ecg.npy'),allow_pickle=True)
    test_dataset = np.load(os.path.join(root_path, 'test_ecg.npy'),allow_pickle=True)
    x_train = train_dataset.item().get('x_train')
    y_train = train_dataset.item().get('y_train')
    x_test = test_dataset.item().get('x_test')
    y_test = test_dataset.item().get('y_test')
    return check_normalized(np.float32(x_train)), check_normalized(np.float32(x_test)), np.int64(y_train), np.int64(y_test)

