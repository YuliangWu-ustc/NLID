from builtins import print
import numpy as np
import pandas as pd
import matplotlib
from sklearn import preprocessing

matplotlib.use('agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.sans-serif'] = 'Arial'
import os
import operator

import utils

from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES
from utils.constants import UNIVARIATE_DATASET_NAMES_2018 as DATASET_NAMES_2018
from utils.constants import ARCHIVE_NAMES  as ARCHIVE_NAMES
from utils.constants import CLASSIFIERS
from utils.constants import ITERATIONS
from utils.constants import MTS_DATASET_NAMES

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder

from scipy.interpolate import interp1d
from scipy.io import loadmat


def readucr(filename):
    """
    Read UCR format time series dataset from CSV file.

    Args:
        filename (str): Path to the CSV file containing the dataset in UCR format.
                        First column should be labels, remaining columns are features.

    Returns:
        tuple: (X, Y) where X is the feature matrix (numpy array) and Y is the label vector (numpy array).
    """
    data = np.loadtxt(filename, delimiter=',')  # ⭐ Load data from CSV file into numpy array
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def create_directory(directory_path):
    """
    创建指定路径的目录，如果目录已存在则返回None，否则尝试创建目录并返回路径。

    Args:
        directory_path (str): 需要创建的目录路径

    Returns:
        str/None: 成功创建则返回目录路径，否则返回None
    """
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)  # ⭐ 核心代码：实际创建多级目录
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def create_path(root_dir, classifier_name, archive_name):
    """
    在根目录下创建分类器和数据集特定的输出路径目录结构。

    Args:
        root_dir (str): 根目录路径
        classifier_name (str): 分类器名称
        archive_name (str): 数据集名称

    Returns:
        str|None: 如果目录不存在则创建并返回路径，若已存在则返回None
    """
    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'
    if os.path.exists(output_directory):
        return None
    else:
        os.makedirs(output_directory)  # ⭐ 核心代码：创建多级目录结构
        return output_directory


def read_dataset(root_dir, archive_name, dataset_name):
    """
    从指定归档目录读取时间序列数据集并进行标准化处理。

    Args:
        root_dir (str): 根目录路径（会自动去除'-temp'后缀）
        archive_name (str): 归档名称（支持'mts_archive'/'UCRArchive_2018'/其他格式）
        dataset_name (str): 具体数据集名称

    Returns:
        dict: 包含数据集名称和对应数据的字典，格式为 {dataset_name: (x_train, y_train, x_test, y_test)}
    """
    datasets_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')

    if archive_name == 'mts_archive':
        file_name = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'
        x_train = np.load(file_name + 'x_train.npy')  # ⭐ 加载numpy格式的训练集特征数据
        y_train = np.load(file_name + 'y_train.npy')
        x_test = np.load(file_name + 'x_test.npy')
        y_test = np.load(file_name + 'y_test.npy')

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())

    elif archive_name == 'UCRArchive_2018':
        root_dir_dataset = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'
        df_train = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)

        df_test = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)

        y_train = df_train.values[:, 0]
        y_test = df_test.values[:, 0]

        x_train = df_train.drop(columns=[0])
        x_test = df_test.drop(columns=[0])

        x_train.columns = range(x_train.shape[1])
        x_test.columns = range(x_test.shape[1])

        x_train = x_train.values
        x_test = x_test.values

        # znorm
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_  # ⭐ 执行Z-score标准化

        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())
    else:
        file_name = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/' + dataset_name
        x_train, y_train = readucr(file_name + '_TRAIN')
        x_test, y_test = readucr(file_name + '_TEST')
        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())

    return datasets_dict

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
            
def read_dataset_torch(dataset_path, archive_name, dataset_name, **kwargs):
    """
    从指定归档目录读取时间序列数据集并进行标准化处理。

    Args:
        root_dir (str): 根目录路径（会自动去除'-temp'后缀）
        archive_name (str): 归档名称（支持'mts_archive'/'UCRArchive_2018'/其他格式）
        dataset_name (str): 具体数据集名称

    Returns:
        dict: 包含数据集名称和对应数据的字典，格式为 {dataset_name: (x_train, y_train, x_test, y_test)}
    """

    # if archive_name == 'mts_archive':
    #     file_name = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'
    #     x_train = np.load(file_name + 'x_train.npy')  # ⭐ 加载numpy格式的训练集特征数据
    #     y_train = np.load(file_name + 'y_train.npy')
    #     x_test = np.load(file_name + 'x_test.npy')
    #     y_test = np.load(file_name + 'y_test.npy')

    #     datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
    #                                    y_test.copy())

    if archive_name == 'UCRArchive_2018':
        
        df_train = pd.read_csv(dataset_path + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)

        df_test = pd.read_csv(dataset_path + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)

        y_train = df_train.values[:, 0]
        y_test = df_test.values[:, 0]

        x_train = df_train.drop(columns=[0])
        x_test = df_test.drop(columns=[0])

        x_train.columns = range(x_train.shape[1])
        x_test.columns = range(x_test.shape[1])

        x_train = x_train.values
        x_test = x_test.values

        # znorm
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_  # ⭐ 执行Z-score标准化

        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

        return x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy()

    elif archive_name == 'chem-liquid-classification':
        
        folder_labels_dict = kwargs['folder_labels_dict']
        selected_areas = kwargs['selected_areas']
        delta_t = kwargs['delta_t']
        
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
        train_label_list = np.array(train_label_list, dtype=np.int64)
        test_label_list = np.array(test_label_list, dtype=np.int64)
        
        print('finished loading data')
        print(f'train_ev_freq_series_list.shape: {train_ev_freq_series_list.shape}')
        print(f'test_ev_freq_series_list.shape: {test_ev_freq_series_list.shape}')
        print(f'train_label_list.shape: {train_label_list.shape}')
        print(f'test_label_list.shape: {test_label_list.shape}')
        
        # return train_ev_freq_series_list, train_label_list, test_ev_freq_series_list, test_label_list
        
        le = preprocessing.LabelEncoder()
        le.fit(np.squeeze(train_label_list, axis=1))
        train_label_list = le.transform(np.squeeze(train_label_list, axis=1))
        le.fit(np.squeeze(test_label_list, axis=1))
        test_label_list = le.transform(np.squeeze(test_label_list, axis=1))
        return set_nan_to_zero(train_ev_freq_series_list), train_label_list, set_nan_to_zero(test_ev_freq_series_list), test_label_list

    raise ValueError('archive_name is not supported')

def read_all_datasets(root_dir, archive_name, split_val=False):
    """
    读取指定归档中的所有数据集，并进行标准化预处理。

    Args:
        root_dir (str): 数据集根目录路径（会自动去除'-temp'后缀）
        archive_name (str): 归档名称（支持'mts_archive'/'UCRArchive_2018'/其他默认格式）
        split_val (bool): 是否拆分验证集（当前未使用）

    Returns:
        dict: 以数据集名为键，值为元组(x_train, y_train, x_test, y_test)的字典
    """
    datasets_dict = {}
    cur_root_dir = root_dir.replace('-temp', '')
    dataset_names_to_sort = []

    if archive_name == 'mts_archive':
        for dataset_name in MTS_DATASET_NAMES:
            root_dir_dataset = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'

            x_train = np.load(root_dir_dataset + 'x_train.npy')  # ⭐ 加载numpy格式的训练数据
            y_train = np.load(root_dir_dataset + 'y_train.npy')
            x_test = np.load(root_dir_dataset + 'x_test.npy')
            y_test = np.load(root_dir_dataset + 'y_test.npy')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())
    elif archive_name == 'UCRArchive_2018':
        for dataset_name in DATASET_NAMES_2018:
            root_dir_dataset = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'

            df_train = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)

            df_test = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)

            y_train = df_train.values[:, 0]
            y_test = df_test.values[:, 0]

            x_train = df_train.drop(columns=[0])
            x_test = df_test.drop(columns=[0])

            x_train.columns = range(x_train.shape[1])
            x_test.columns = range(x_test.shape[1])

            x_train = x_train.values
            x_test = x_test.values

            # znorm
            std_ = x_train.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_  # ⭐ 执行Z-score标准化

            std_ = x_test.std(axis=1, keepdims=True)
            std_[std_ == 0] = 1.0
            x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())

    else:
        for dataset_name in DATASET_NAMES:
            root_dir_dataset = cur_root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'
            file_name = root_dir_dataset + dataset_name
            x_train, y_train = readucr(file_name + '_TRAIN')  # ⭐ 使用readucr函数读取UCR格式数据
            x_test, y_test = readucr(file_name + '_TEST')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())

            dataset_names_to_sort.append((dataset_name, len(x_train)))

        dataset_names_to_sort.sort(key=operator.itemgetter(1))

        for i in range(len(DATASET_NAMES)):
            DATASET_NAMES[i] = dataset_names_to_sort[i][0]

    return datasets_dict


def get_func_length(x_train, x_test, func):
    """
    计算训练集和测试集中时间序列数据的最大或最小长度。

    Args:
        x_train (numpy.ndarray): 训练集数据，形状为(n_samples, n_features, n_timesteps)
        x_test (numpy.ndarray): 测试集数据，形状为(n_samples, n_features, n_timesteps)
        func (function): 用于确定计算方式的函数，可以是min或max

    Returns:
        int: 计算得到的时间序列长度（最大或最小值）
    """
    if func == min:
        func_length = np.inf  # ⭐ 初始化最小长度为无穷大
    else:
        func_length = 0

    n = x_train.shape[0]
    for i in range(n):
        func_length = func(func_length, x_train[i].shape[1])  # ⭐ 核心计算逻辑：比较并更新长度值

    n = x_test.shape[0]
    for i in range(n):
        func_length = func(func_length, x_test[i].shape[1])

    return func_length


def transform_to_same_length(x, n_var, max_length):
    """
    将不同长度的时间序列数据通过线性插值转换为统一长度。

    Args:
        x (np.ndarray): 原始时间序列数据，形状为(n, variable, length)
        n_var (int): 时间序列的变量数量
        max_length (int): 目标统一长度

    Returns:
        np.ndarray: 转换后的统一长度时间序列数据，形状为(n, max_length, n_var)
    """
    n = x.shape[0]

    # the new set in ucr form np array
    ucr_x = np.zeros((n, max_length, n_var), dtype=float64)

    # loop through each time series
    for i in range(n):
        mts = x[i]
        curr_length = mts.shape[1]
        idx = np.array(range(curr_length))
        idx_new = np.linspace(0, idx.max(), max_length)
        for j in range(n_var):
            ts = mts[j]
            # linear interpolation
            f = interp1d(idx, ts, kind='cubic')  # ⭐ 使用三次样条插值生成插值函数
            new_ts = f(idx_new)
            ucr_x[i, :, j] = new_ts

    return ucr_x


def transform_mts_to_ucr_format():
    """
    将MTS格式的时间序列数据集转换为UCR兼容的格式。
    
    处理流程：
    1. 从指定路径加载.mat格式的MTS数据集
    2. 提取训练集/测试集数据和标签
    3. 计算并统一所有样本的时间序列长度
    4. 将处理后的数据保存为.npy格式
    
    数据保存位置：/mnt/nfs/casimir/archives/mts_archive/[数据集名称]/
    """
    mts_root_dir = '/mnt/Other/mtsdata/'
    mts_out_dir = '/mnt/nfs/casimir/archives/mts_archive/'
    for dataset_name in MTS_DATASET_NAMES:
        # print('dataset_name',dataset_name)

        out_dir = mts_out_dir + dataset_name + '/'

        # if create_directory(out_dir) is None:
        #     print('Already_done')
        #     continue

        a = loadmat(mts_root_dir + dataset_name + '/' + dataset_name + '.mat')
        a = a['mts']
        a = a[0, 0]

        dt = a.dtype.names
        dt = list(dt)

        for i in range(len(dt)):
            if dt[i] == 'train':
                x_train = a[i].reshape(max(a[i].shape))  # ⭐ 提取并重塑训练集数据
            elif dt[i] == 'test':
                x_test = a[i].reshape(max(a[i].shape))  # ⭐ 提取并重塑测试集数据
            elif dt[i] == 'trainlabels':
                y_train = a[i].reshape(max(a[i].shape))
            elif dt[i] == 'testlabels':
                y_test = a[i].reshape(max(a[i].shape))

        # x_train = a[1][0]
        # y_train = a[0][:,0]
        # x_test = a[3][0]
        # y_test = a[2][:,0]

        n_var = x_train[0].shape[0]

        max_length = get_func_length(x_train, x_test, func=max)  # ⭐ 计算最大序列长度
        min_length = get_func_length(x_train, x_test, func=min)

        print(dataset_name, 'max', max_length, 'min', min_length)
        print()
        # continue

        x_train = transform_to_same_length(x_train, n_var, max_length)  # ⭐ 统一训练集长度
        x_test = transform_to_same_length(x_test, n_var, max_length)  # ⭐ 统一测试集长度

        # save them
        np.save(out_dir + 'x_train.npy', x_train)
        np.save(out_dir + 'y_train.npy', y_train)
        np.save(out_dir + 'x_test.npy', x_test)
        np.save(out_dir + 'y_test.npy', y_test)

        print('Done')


def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    """
    计算分类模型的各项评估指标并返回结果DataFrame。

    Args:
        y_true (array-like): 真实标签数组
        y_pred (array-like): 预测标签数组
        duration (float): 模型运行时间
        y_true_val (array-like, optional): 验证集真实标签. Defaults to None.
        y_pred_val (array-like, optional): 验证集预测标签. Defaults to None.

    Returns:
        pd.DataFrame: 包含precision/accuracy/recall/duration指标的DataFrame，
                      若有验证集会额外包含accuracy_val指标
    """
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')  # ⭐ 计算宏平均精确率
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def save_test_duration(file_name, test_duration):
    """
    将测试持续时间保存到指定的CSV文件中。

    Args:
        file_name (str): 要保存的CSV文件路径。
        test_duration (float): 需要保存的测试持续时间值。

    Returns:
        None: 该函数没有返回值，直接将结果写入CSV文件。
    """
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration  # ⭐ 将测试持续时间值赋给数据框
    res.to_csv(file_name, index=False)


def generate_results_csv(output_file_name, root_dir):
    """
    从多个分类器和数据集的实验结果中收集指标数据，生成汇总CSV文件并计算平均准确率。

    Args:
        output_file_name (str): 输出CSV文件的名称
        root_dir (str): 包含所有实验结果文件的根目录路径

    Returns:
        pd.DataFrame: 包含平均准确率的DataFrame（按分类器、存档和数据集分组）
    """
    res = pd.DataFrame(data=np.zeros((0, 7), dtype=float), index=[],
                       columns=['classifier_name', 'archive_name', 'dataset_name',
                                'precision', 'accuracy', 'recall', 'duration'])
    for classifier_name in CLASSIFIERS:
        for archive_name in ARCHIVE_NAMES:
            datasets_dict = read_all_datasets(root_dir, archive_name)
            for it in range(ITERATIONS):
                curr_archive_name = archive_name
                if it != 0:
                    curr_archive_name = curr_archive_name + '_itr_' + str(it)
                for dataset_name in datasets_dict.keys():
                    output_dir = root_dir + '/results/' + classifier_name + '/' \
                                 + curr_archive_name + '/' + dataset_name + '/' + 'df_metrics.csv'
                    if not os.path.exists(output_dir):
                        continue
                    df_metrics = pd.read_csv(output_dir)  # ⭐ 读取单个实验结果的CSV文件
                    df_metrics['classifier_name'] = classifier_name
                    df_metrics['archive_name'] = archive_name
                    df_metrics['dataset_name'] = dataset_name
                    res = pd.concat((res, df_metrics), axis=0, sort=False)

    res.to_csv(root_dir + output_file_name, index=False)
    # aggreagte the accuracy for iterations on same dataset
    res = pd.DataFrame({
        'accuracy': res.groupby(
            ['classifier_name', 'archive_name', 'dataset_name'])['accuracy'].mean()
    }).reset_index()  # ⭐ 计算相同数据集上多次迭代的平均准确率

    return res


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def save_logs_t_leNet(output_directory, hist, y_pred, y_true, duration):
    """
    保存LeNet模型训练过程中的日志信息到指定目录。

    Args:
        output_directory (str): 输出文件保存目录路径
        hist (keras.callbacks.History): 训练历史记录对象
        y_pred (numpy.ndarray): 模型预测结果
        y_true (numpy.ndarray): 真实标签
        duration (float): 训练耗时

    功能:
        1. 保存训练历史到history.csv
        2. 计算并保存评估指标到df_metrics.csv
        3. 保存最佳模型参数到df_best_model.csv
        4. 绘制并保存损失曲线图
    """
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)  # ⭐ 保存训练历史记录到CSV文件

    df_metrics = calculate_metrics(y_true, y_pred, duration)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['acc']
    df_best_model['best_model_val_acc'] = row_best_model['val_acc']
    df_best_model['best_model_nb_epoch'] = index_best_model  # ⭐ 记录最佳模型的epoch数

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')


def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    """
    保存模型训练日志到指定目录，包括训练历史、评估指标和最佳模型参数。

    Args:
        output_directory (str): 输出文件保存目录路径
        hist (keras.callbacks.History): 训练历史对象
        y_pred (numpy.ndarray): 测试集预测结果
        y_true (numpy.ndarray): 测试集真实标签
        duration (float): 训练耗时
        lr (bool, optional): 是否记录学习率. 默认为True
        y_true_val (numpy.ndarray, optional): 验证集真实标签. 默认为None
        y_pred_val (numpy.ndarray, optional): 验证集预测结果. 默认为None

    Returns:
        pandas.DataFrame: 包含评估指标的DataFrame
    """
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)  # ⭐ 保存训练历史到CSV文件

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()  # ⭐ 找到损失最小的最佳模型索引
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics

import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns

def plot_epochs_metric_torch(hist, file_name, metric='accuracy'):
    plt.figure()
    plt.plot(hist['train_accuracy'])
    plt.plot(hist['val_accuracy'])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'])
    plt.xticks(range(0, len(hist['train_accuracy']), len(hist['train_accuracy']) // 10))
    plt.yticks(np.arange(0, 1.1, 0.1))  # ⭐ 使用np.arange支持浮点数步长
    plt.grid(True)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def calculate_confusion_matrix_torch(y_true_val, y_pred_val, output_directory, figsize=None, labels=None): 
    """
    计算分类模型的各项评估指标并返回结果DataFrame。

    Args:
        y_true (array-like): 真实标签数组
        y_pred (array-like): 预测标签数组
        duration (float): 模型运行时间
        y_true_val (array-like, optional): 验证集真实标签. Defaults to None.
        y_pred_val (array-like, optional): 验证集预测标签. Defaults to None.

    Returns:
        pd.DataFrame: 包含precision/accuracy/recall/duration指标的DataFrame，
                      若有验证集会额外包含accuracy_val指标
    """
    # 输出y_true_val的唯一值和对应数量
    unique_values, counts = np.unique(y_true_val, return_counts=True)
    for i in range(len(unique_values)):
        print(f'{unique_values[i]}: {counts[i]}')
    # unique_values, counts = np.unique(y_pred_val, return_counts=True)
    # for i in range(len(unique_values)):
    #     print(f'{unique_values[i]}: {counts[i]}')
    
    _confusion_matrix = confusion_matrix(y_true_val, y_pred_val)
    if figsize is None:
        figsize = (len(np.unique(y_true_val)), len(np.unique(y_true_val)))
    plt.figure(figsize=figsize)
    sns.heatmap(_confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_directory, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()
    
    return _confusion_matrix
    

def save_logs_torch(output_directory, hist, duration, y_true_val, y_pred_val, y_labels_description=None):
    """
    保存模型训练日志到指定目录，包括训练历史、评估指标和最佳模型参数。

    Args:
        output_directory (str): 输出文件保存目录路径
        hist (keras.callbacks.History): 训练历史对象
        y_pred (numpy.ndarray): 测试集预测结果
        y_true (numpy.ndarray): 测试集真实标签
        duration (float): 训练耗时
        lr (bool, optional): 是否记录学习率. 默认为True
        y_true_val (numpy.ndarray, optional): 验证集真实标签. 默认为None
        y_pred_val (numpy.ndarray, optional): 验证集预测结果. 默认为None

    Returns:
        pandas.DataFrame: 包含评估指标的DataFrame
    """
    
    try:
        best_model_train_loss = hist['best_model_train_loss']
        best_model_train_accuracy = hist['best_model_train_accuracy']
        best_model_val_accuracy = hist['best_model_val_accuracy']
        
        with open(os.path.join(output_directory, 'etc.txt'), 'w') as f:
            f.write(f'best_model_train_loss: {best_model_train_loss}\n')
            f.write(f'best_model_train_accuracy: {best_model_train_accuracy}\n')
            f.write(f'best_model_val_accuracy: {best_model_val_accuracy}\n')
            f.write(f'duration: {duration}\n')
        
        for key in hist.keys():
            if key not in ['loss', 'val_loss', 'train_accuracy', 'val_accuracy', 'train_f1', 'val_f1']:
                hist.pop(key)
    except:
        pass
    
    try:
        hist_df = pd.DataFrame(hist)
        hist_df.to_csv(os.path.join(output_directory, 'history.csv'), index=False)  # ⭐ 保存训练历史到CSV文件
    except:
        print('Save history failed.')
        pass

    # plot losses
    plot_epochs_metric_torch(hist, os.path.join(output_directory, 'epochs_accuracy.png'))

    calculate_confusion_matrix_torch(y_true_val, y_pred_val, output_directory, labels=y_labels_description)

def save_logs_torch_regr(output_directory, hist, duration, y_true_val, y_pred_val, y_labels_description=None):
    """
    保存模型训练日志到指定目录，包括训练历史、评估指标和最佳模型参数。

    Args:
        output_directory (str): 输出文件保存目录路径
        hist (keras.callbacks.History): 训练历史对象
        y_pred (numpy.ndarray): 测试集预测结果
        y_true (numpy.ndarray): 测试集真实标签
        duration (float): 训练耗时
        lr (bool, optional): 是否记录学习率. 默认为True
        y_true_val (numpy.ndarray, optional): 验证集真实标签. 默认为None
        y_pred_val (numpy.ndarray, optional): 验证集预测结果. 默认为None

    Returns:
        pandas.DataFrame: 包含评估指标的DataFrame
    """
    
    try:
        for key in hist.keys():
            if key not in ['loss', 'val_loss', 'train_mae', 'val_mae', 'train_maxae', 'val_maxae']:
                hist.pop(key)
    except:
        pass
    
    try:
        hist_df = pd.DataFrame(hist)
        hist_df.to_csv(os.path.join(output_directory, 'history.csv'), index=False)  # ⭐ 保存训练历史到CSV文件
    except:
        print('Save history failed.')
        pass

    # plot losses
    # plot_epochs_metric_torch(hist, os.path.join(output_directory, 'epochs_accuracy.png'))

    # calculate_confusion_matrix_torch(y_true_val, y_pred_val, output_directory, labels=y_labels_description)

def visualize_filter(root_dir):
    """
    可视化神经网络第一层卷积滤波器对时间序列数据的作用效果。

    1. 加载预训练模型和UCR数据集
    2. 提取第一层卷积层的权重作为滤波器
    3. 绘制原始数据与卷积结果的对比图
    4. 保存可视化结果为PDF文件

    Args:
        root_dir (str): 项目根目录路径，用于定位模型和数据文件

    Returns:
        int: 固定返回1，表示执行完成
    """
    import tensorflow.keras as keras
    classifier = 'resnet'
    archive_name = 'UCRArchive_2018'
    dataset_name = 'GunPoint'
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    model = keras.models.load_model(
        root_dir + 'results/' + classifier + '/' + archive_name + '/' + dataset_name + '/best_model.hdf5')  # ⭐ 加载预训练模型

    # filters
    filters = model.layers[1].get_weights()[0]

    new_input_layer = model.inputs
    new_output_layer = [model.layers[1].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    classes = np.unique(y_train)

    colors = [(255 / 255, 160 / 255, 14 / 255), (181 / 255, 87 / 255, 181 / 255)]
    colors_conv = [(210 / 255, 0 / 255, 0 / 255), (27 / 255, 32 / 255, 101 / 255)]

    idx = 10
    idx_filter = 1

    filter = filters[:, 0, idx_filter]

    plt.figure(1)
    plt.plot(filter + 0.5, color='gray', label='filter')
    for c in classes:
        c_x_train = x_train[np.where(y_train == c)]
        convolved_filter_1 = new_feed_forward([c_x_train])[0]  # ⭐ 获取卷积层输出特征图

        idx_c = int(c) - 1

        plt.plot(c_x_train[idx], color=colors[idx_c], label='class' + str(idx_c) + '-raw')
        plt.plot(convolved_filter_1[idx, :, idx_filter], color=colors_conv[idx_c], label='class' + str(idx_c) + '-conv')
        plt.legend()

    plt.savefig(root_dir + 'convolution-' + dataset_name + '.pdf')  # ⭐ 保存可视化结果

    return 1


def viz_perf_themes(root_dir, df):
    """
    分析并可视化不同数据主题下各分类器的性能表现，生成主题性能统计表。

    Args:
        root_dir (str): 结果文件存储的根目录路径
        df (DataFrame): 包含各数据集在各分类器上性能指标的DataFrame

    Returns:
        None (结果直接保存为CSV文件)
    """
    df_themes = df.copy()
    themes_index = []
    # 添加主题索引
    for dataset_name in df.index:
        themes_index.append(utils.constants.dataset_types[dataset_name])

    themes_index = np.array(themes_index)
    themes, themes_counts = np.unique(themes_index, return_counts=True)
    df_themes.index = themes_index
    df_themes = df_themes.rank(axis=1, method='min', ascending=False)  # ⭐ 计算每个数据集中各分类器的排名
    df_themes = df_themes.where(df_themes.values == 1)
    df_themes = df_themes.groupby(level=0).sum(axis=1)
    df_themes['#'] = themes_counts

    for classifier in CLASSIFIERS:
        df_themes[classifier] = df_themes[classifier] / df_themes['#'] * 100  # ⭐ 计算每个主题中排名第一的比例
    df_themes = df_themes.round(decimals=1)
    df_themes.to_csv(root_dir + 'tab-perf-theme.csv')


def viz_perf_train_size(root_dir, df):
    """
    根据训练集大小可视化模型性能，计算不同规模训练集下的平均排名并保存结果。

    Args:
        root_dir (str): 数据集根目录路径
        df (DataFrame): 包含各数据集性能指标的DataFrame

    Returns:
        None: 结果直接输出到控制台并保存为CSV文件
    """
    df_size = df.copy()
    train_sizes = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCR_TS_Archive_2015')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        train_size = len(datasets_dict[dataset_name][0])  # ⭐ 获取每个数据集的训练集样本数量
        train_sizes.append(train_size)

    train_sizes = np.array(train_sizes)
    bins = np.array([0, 100, 400, 800, 99999])
    train_size_index = np.digitize(train_sizes, bins)  # ⭐ 将训练集大小分配到预定义的区间
    train_size_index = bins[train_size_index]

    df_size.index = train_size_index
    df_size = df_size.rank(axis=1, method='min', ascending=False)
    df_size = df_size.groupby(level=0, axis=0).mean()  # ⭐ 按训练集大小分组计算平均排名
    df_size = df_size.round(decimals=2)

    print(df_size.to_string())
    df_size.to_csv(root_dir + 'tab-perf-train-size.csv')


def viz_perf_classes(root_dir, df):
    """
    分析并可视化不同类别数量下的模型性能表现，输出分类统计结果并保存为CSV文件。

    Args:
        root_dir (str): 数据集根目录路径
        df (DataFrame): 包含模型性能指标的DataFrame

    Returns:
        None: 结果直接输出到控制台并保存为CSV文件
    """
    df_classes = df.copy()
    class_numbers = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCR_TS_Archive_2015')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        train_size = len(np.unique(datasets_dict[dataset_name][1]))
        class_numbers.append(train_size)  # ⭐ 获取每个数据集的类别数量

    class_numbers = np.array(class_numbers)
    bins = np.array([0, 3, 4, 6, 8, 13, 9999])
    class_numbers_index = np.digitize(class_numbers, bins)
    class_numbers_index = bins[class_numbers_index]

    df_classes.index = class_numbers_index
    df_classes = df_classes.rank(axis=1, method='min', ascending=False)  # ⭐ 计算每行数据的排名
    df_classes = df_classes.groupby(level=0, axis=0).mean()  # ⭐ 按类别分组计算均值
    df_classes = df_classes.round(decimals=2)

    print(df_classes.to_string())
    df_classes.to_csv(root_dir + 'tab-perf-classes.csv')


def viz_perf_length(root_dir, df):
    """
    根据时间序列长度分析模型性能表现，按长度分组计算平均排名并输出结果。

    Args:
        root_dir (str): 数据集根目录路径
        df (DataFrame): 包含各数据集性能指标的DataFrame

    Returns:
        None: 结果直接输出到控制台和CSV文件
    """
    df_lengths = df.copy()
    lengths = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCR_TS_Archive_2015')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        length = datasets_dict[dataset_name][0].shape[1]  # ⭐ 获取每个数据集的时间序列长度
        lengths.append(length)

    lengths = np.array(lengths)
    bins = np.array([0, 81, 251, 451, 700, 1001, 9999])
    lengths_index = np.digitize(lengths, bins)
    lengths_index = bins[lengths_index]

    df_lengths.index = lengths_index
    df_lengths = df_lengths.rank(axis=1, method='min', ascending=False)  # ⭐ 计算各方法在不同长度组的排名
    df_lengths = df_lengths.groupby(level=0, axis=0).mean()
    df_lengths = df_lengths.round(decimals=2)

    print(df_lengths.to_string())
    df_lengths.to_csv(root_dir + 'tab-perf-lengths.csv')


def viz_plot(root_dir, df):
    """
    生成并保存FCN和ResNet模型在不同时间序列数据集上准确率的对比散点图

    Args:
        root_dir (str): 根目录路径，用于读取数据集和保存结果图
        df (pd.DataFrame): 包含各数据集分类准确率的DataFrame，索引为数据集名称

    Returns:
        None: 结果直接保存为PDF文件
    """
    df_lengths = df.copy()
    lengths = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCR_TS_Archive_2015')  # ⭐ 读取UCR时间序列数据集
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')  # ⭐ 读取多元时间序列数据集
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        length = datasets_dict[dataset_name][0].shape[1]  # ⭐ 获取每个数据集的时间序列长度
        lengths.append(length)

    lengths_index = np.array(lengths)

    df_lengths.index = lengths_index

    plt.scatter(x=df_lengths['fcn'], y=df_lengths['resnet'])  # ⭐ 生成FCN vs ResNet准确率散点图
    plt.ylim(ymin=0, ymax=1.05)
    plt.xlim(xmin=0, xmax=1.05)
    # df_lengths['fcn']
    plt.savefig(root_dir + 'plot.pdf')


def viz_for_survey_paper(root_dir, filename='results-ucr-mts.csv'):
    """
    处理并可视化时间序列数据集的调查结果，支持多种维度的性能展示。

    Args:
        root_dir (str): 结果文件所在的根目录路径。
        filename (str, optional): 要读取的结果文件名，默认为'results-ucr-mts.csv'。

    Returns:
        None: 该函数主要通过调用其他可视化函数来展示结果，不直接返回值。
    """
    df = pd.read_csv(root_dir + filename, index_col=0)  # ⭐ 读取CSV文件并设置第一列为索引
    df = df.T
    df = df.round(decimals=2)

    # get table performance per themes
    # viz_perf_themes(root_dir,df)

    # get table performance per train size
    # viz_perf_train_size(root_dir,df)

    # get table performance per classes
    # viz_perf_classes(root_dir,df)

    # get table performance per length
    # viz_perf_length(root_dir,df)

    # get plot
    viz_plot(root_dir, df)


def viz_cam(root_dir):
    """
    生成并保存类激活映射(CAM)可视化图像，展示模型在时间序列数据上的关键关注区域。

    Args:
        root_dir (str): 项目根目录路径，用于定位模型和数据文件。

    流程:
        1. 加载预训练模型和数据集
        2. 提取模型权重和中间层输出
        3. 计算每个时间点的类激活强度
        4. 生成带热力覆盖的时间序列图
        5. 按类别保存可视化结果
    """
    import tensorflow.keras as keras
    import sklearn
    classifier = 'resnet'
    archive_name = 'UCRArchive_2018'
    dataset_name = 'GunPoint'

    if dataset_name == 'Gun_Point':
        save_name = 'GunPoint'
    else:
        save_name = dataset_name
    max_length = 2000
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    y_test = datasets_dict[dataset_name][3]

    # transform to binary labels
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train_binary = enc.transform(y_train.reshape(-1, 1)).toarray()

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    model = keras.models.load_model(
        root_dir + 'results/' + classifier + '/' + archive_name + '/' + dataset_name + '/best_model.hdf5')  # ⭐ 加载预训练模型

    # filters
    w_k_c = model.layers[-1].get_weights()[0]  # ⭐ 获取最后一层的权重矩阵

    # the same input
    new_input_layer = model.inputs
    # output is both the original as well as the before last layer
    new_output_layer = [model.layers[-3].output, model.layers[-1].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    classes = np.unique(y_train)

    for c in classes:
        plt.figure()
        count = 0
        c_x_train = x_train[np.where(y_train == c)]
        for ts in c_x_train:
            ts = ts.reshape(1, -1, 1)
            [conv_out, predicted] = new_feed_forward([ts])  # ⭐ 获取卷积层输出和预测结果
            pred_label = np.argmax(predicted)
            orig_label = np.argmax(enc.transform([[c]]))
            if pred_label == orig_label:
                cas = np.zeros(dtype=float, shape=(conv_out.shape[1]))
                for k, w in enumerate(w_k_c[:, orig_label]):
                    cas += w * conv_out[0, :, k]  # ⭐ 计算类激活强度

                minimum = np.min(cas)

                cas = cas - minimum

                cas = cas / max(cas)
                cas = cas * 100

                x = np.linspace(0, ts.shape[1] - 1, max_length, endpoint=True)
                # linear interpolation to smooth
                f = interp1d(range(ts.shape[1]), ts[0, :, 0])
                y = f(x)
                # if (y < -2.2).any():
                #     continue
                f = interp1d(range(ts.shape[1]), cas)
                cas = f(x).astype(int)
                plt.scatter(x=x, y=y, c=cas, cmap='jet', marker='.', s=2, vmin=0, vmax=100, linewidths=0.0)
                if dataset_name == 'Gun_Point':
                    if c == 1:
                        plt.yticks([-1.0, 0.0, 1.0, 2.0])
                    else:
                        plt.yticks([-2, -1.0, 0.0, 1.0, 2.0])
                count += 1

        cbar = plt.colorbar()
        # cbar.ax.set_yticklabels([100,75,50,25,0])
        plt.savefig(root_dir + '/temp/' + classifier + '-cam-' + save_name + '-class-' + str(int(c)) + '.png',
                    bbox_inches='tight', dpi=1080)  # ⭐ 保存可视化结果
