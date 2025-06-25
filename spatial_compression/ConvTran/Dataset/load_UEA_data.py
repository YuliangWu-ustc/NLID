import os
import numpy as np
import logging
import random
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sktime.datasets import load_from_tsfile_to_dataframe


logger = logging.getLogger(__name__)


def load(config):
    """
    加载和预处理UEA时间序列数据集。

    Args:
        config (dict): 配置字典，包含以下关键参数：
            - data_dir: 数据目录路径
            - Norm: 是否进行标准化处理
            - val_ratio: 验证集划分比例

    Returns:
        dict: 包含预处理后数据的字典，包含以下键：
            - max_len: 最大序列长度
            - All_train_data: 全部训练数据
            - All_train_label: 全部训练标签
            - train_data: 训练集数据
            - train_label: 训练集标签
            - val_data: 验证集数据
            - val_label: 验证集标签
            - test_data: 测试集数据
            - test_label: 测试集标签
    """
    # Build data
    Data = {}
    problem = config['data_dir'].split('/')[-1]

    if os.path.exists(config['data_dir'] + '/' + problem + '.npy'):
        logger.info("Loading preprocessed data ...")
        Data_npy = np.load(config['data_dir'] + '/' + problem + '.npy', allow_pickle=True)  # ⭐ 加载预处理的.npy数据文件

        Data['max_len'] = Data_npy.item().get('max_len')
        Data['All_train_data'] = Data_npy.item().get('All_train_data')
        Data['All_train_label'] = Data_npy.item().get('All_train_label')
        Data['train_data'] = Data_npy.item().get('train_data')
        Data['train_label'] = Data_npy.item().get('train_label')
        Data['val_data'] = Data_npy.item().get('val_data')
        Data['val_label'] = Data_npy.item().get('val_label')
        Data['test_data'] = Data_npy.item().get('test_data')
        Data['test_label'] = Data_npy.item().get('test_label')

        logger.info("{} samples will be used for training".format(len(Data['train_label'])))
        logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
        logger.info("{} samples will be used for testing".format(len(Data['test_label'])))

    else:
        logger.info("Loading and preprocessing data ...")
        train_file = config['data_dir'] + "/" + problem + "_TRAIN.ts"
        test_file = config['data_dir'] + "/" + problem + "_TEST.ts"
        train_df, y_train = load_from_tsfile_to_dataframe(train_file)  # ⭐ 从.ts文件加载原始训练数据
        test_df, y_test = load_from_tsfile_to_dataframe(test_file)  # ⭐ 从.ts文件加载原始测试数据

        y_train = LabelEncoder().fit_transform(y_train)
        y_test = LabelEncoder().fit_transform(y_test)

        train_lengths = train_df.applymap(lambda x: len(x)).values
        test_lengths = test_df.applymap(lambda x: len(x)).values
        train_max_seq_len = int(np.max(train_lengths[:, 0]))
        test_max_seq_len = int(np.max(test_lengths[:, 0]))
        max_seq_len = np.max([train_max_seq_len, test_max_seq_len])

        X_train = process_ts_data(train_df, max_seq_len, normalise=False)
        X_test = process_ts_data(test_df, max_seq_len, normalise=False)

        if config['Norm']:
            mean, std = mean_std(X_train)
            mean = np.repeat(mean, max_seq_len).reshape(X_train.shape[1], max_seq_len)
            std = np.repeat(std, max_seq_len).reshape(X_train.shape[1], max_seq_len)
            X_train = mean_std_transform(X_train, mean, std)
            X_test = mean_std_transform(X_test, mean, std)

        Data['max_len'] = max_seq_len
        Data['All_train_data'] = X_train
        Data['All_train_label'] = y_train

        if config['val_ratio'] > 0:
            train_data, train_label, val_data, val_label = split_dataset(X_train, y_train, config['val_ratio'])  # ⭐ 划分训练集和验证集
        else:
            val_data, val_label = [None, None]

        logger.info("{} samples will be used for training".format(len(train_label)))
        logger.info("{} samples will be used for validation".format(len(val_label)))
        logger.info("{} samples will be used for testing".format(len(y_test)))

        Data['train_data'] = train_data
        Data['train_label'] = train_label
        Data['val_data'] = val_data
        Data['val_label'] = val_label
        Data['test_data'] = X_test
        Data['test_label'] = y_test

        np.save(config['data_dir'] + "/" + problem, Data, allow_pickle=True)  # ⭐ 保存预处理后的数据

    return Data


def split_dataset(data, label, validation_ratio):
    """
    将数据集分层划分为训练集和验证集，保持类别分布比例

    Args:
        data (np.array): 输入的特征数据
        label (np.array): 对应的类别标签
        validation_ratio (float): 验证集所占比例(0-1之间)

    Returns:
        tuple: 包含四个元素的元组 (训练集数据, 训练集标签, 验证集数据, 验证集标签)
    """
    splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=1234)  # ⭐ 创建分层抽样分割器，确保类别比例
    train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))
    train_data = data[train_indices]
    train_label = label[train_indices]
    val_data = data[val_indices]
    val_label = label[val_indices]
    return train_data, train_label, val_data, val_label


def fill_missing(x: np.array, max_len: int, vary_len: str = "suffix-noise", normalise: bool = True):
    """
    处理时间序列数据中的缺失值，并根据指定策略进行填充和标准化。

    Args:
        x (np.array): 包含可能缺失值的时间序列数据数组
        max_len (int): 时间序列的最大长度
        vary_len (str): 缺失值填充策略，可选：
            "zero" - 用零填充
            "prefix-suffix-noise" - 前后缀用随机噪声填充
            "uniform-scaling" - 均匀缩放填充
            其他值 - 默认用随机噪声填充
        normalise (bool): 是否对数据进行标准化

    Returns:
        np.array: 处理后的时间序列数据数组
    """
    if vary_len == "zero":
        if normalise:
            x = StandardScaler().fit_transform(x)  # ⭐ 对数据进行标准化处理
        x = np.nan_to_num(x)
    elif vary_len == 'prefix-suffix-noise':
        for i in range(len(x)):
            series = list()
            for a in x[i, :]:
                if np.isnan(a):
                    break
                series.append(a)
            series = np.array(series)
            seq_len = len(series)
            diff_len = int(0.5 * (max_len - seq_len))

            for j in range(diff_len):
                x[i, j] = random.random() / 1000  # ⭐ 前缀填充随机噪声

            for j in range(diff_len, seq_len):
                x[i, j] = series[j - seq_len]

            for j in range(seq_len, max_len):
                x[i, j] = random.random() / 1000  # ⭐ 后缀填充随机噪声

            if normalise:
                tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                x[i] = tmp[:, 0]
    elif vary_len == 'uniform-scaling':
        for i in range(len(x)):
            series = list()
            for a in x[i, :]:
                if np.isnan(a):
                    break
                series.append(a)
            series = np.array(series)
            seq_len = len(series)

            for j in range(max_len):
                scaling_factor = int(j * seq_len / max_len)  # ⭐ 计算均匀缩放因子
                x[i, j] = series[scaling_factor]
            if normalise:
                tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                x[i] = tmp[:, 0]
    else:
        for i in range(len(x)):
            for j in range(len(x[i])):
                if np.isnan(x[i, j]):
                    x[i, j] = random.random() / 1000  # ⭐ 默认随机噪声填充缺失值

            if normalise:
                tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                x[i] = tmp[:, 0]

    return x


def process_ts_data(x, max_len, vary_len: str = "suffix-noise", normalise: bool = False):
    """
    处理时间序列数据，将其从DataFrame转换为numpy数组，并进行长度统一化和预处理。

    Args:
        x (pd.DataFrame): 输入的时间序列数据，每列是一个特征维度
        max_len (int): 统一后的时间序列最大长度
        vary_len (str): 处理可变长度序列的方法，默认为"suffix-noise"
        normalise (bool): 是否对数据进行标准化，默认为False

    Returns:
        np.ndarray: 处理后的三维numpy数组，形状为(样本数, 特征维度, 时间步长)
    """
    num_instances, num_dim = x.shape
    columns = x.columns
    # max_len = np.max([len(X[columns[0]][i]) for i in range(num_instances)])
    output = np.zeros((num_instances, num_dim, max_len), dtype=np.float64)  # ⭐ 创建输出数组并初始化
    for i in range(num_dim):
        for j in range(num_instances):
            lengths = len(x[columns[i]][j].values)
            end = min(lengths, max_len)
            output[j, i, :end] = x[columns[i]][j].values
        output[:, i, :] = fill_missing(output[:, i, :], max_len, vary_len, normalise)
    return output


def mean_std(train_data):
    """
    计算训练数据在时间维度上的均值和标准差。

    Args:
        train_data (ndarray): 三维训练数据数组，形状为(样本数, 特征数, 时间步长)

    Returns:
        tuple: 包含两个元素的元组：
            - mean (ndarray): 每个特征在所有时间步上的均值
            - std (ndarray): 每个特征在所有时间步上的标准差的最大值
    """
    m_len = np.mean(train_data, axis=2)  # ⭐ 计算每个样本每个特征在时间维度上的均值
    mean = np.mean(m_len, axis=0)

    s_len = np.std(train_data, axis=2)  # ⭐ 计算每个样本每个特征在时间维度上的标准差
    std = np.max(s_len, axis=0)

    return mean, std


def mean_std_transform(train_data, mean, std):
    """
    使用给定的均值和标准差对数据进行标准化变换（Z-score标准化）。

    Args:
        train_data (ndarray): 需要标准化的原始数据
        mean (ndarray): 均值向量
        std (ndarray): 标准差向量

    Returns:
        ndarray: 标准化后的数据
    """
    return (train_data - mean) / std  # ⭐ 执行Z-score标准化计算
