import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

__author__ = "Chang Wei Tan and Navid Foumani"


def load_segmentation_data(file_path, data_type, norm=True, verbose=1):
    """
    加载并处理时间序列分割数据，包括数据过滤、标签重编码和标准化。

    Args:
        file_path (str): CSV文件路径，包含时间序列数据
        data_type (str): 数据类型("Clean"或其他)，决定过滤哪些类别
        norm (bool, optional): 是否对数据进行标准化. 默认为True
        verbose (int, optional): 控制打印信息的详细程度. 默认为1

    Returns:
        pd.DataFrame: 包含处理后的数据和标签的DataFrame，每行包含:
            - data: 形状为(seq_len, n_dim)的时间序列数据
            - label: 对应的时间点标签(0或1)
    """
    if verbose > 0:
        print("[Data_Loader] Loading data from {}".format(file_path))

    df = pd.read_csv(file_path)
    drive = [3, 11]
    if data_type == "Clean":
        # Drop other class data ------------------------------------------------------------------------------
        Other_Class = [0, 1, 2, 12]  # "X", "EyesCLOSEDneutral", "EyesOPENneutral", "LateBoredomLap"
        df = df.drop(np.squeeze(np.where(np.isin(df['label'], Other_Class))))
        distract = [4, 5, 6, 7, 8, 9, 10, 13, 14, 15]
        # -----------------------------------------------------------------------------------------------------
    else:
        distract = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
    all_series = df.series.unique()
    data = []
    for series in all_series:
        if verbose > 0:
            print("[Data_Loader] Processing series {}".format(series))
        this_series = df.loc[df.series == series].reset_index(drop=True)

        this_series.label = this_series.label.replace(distract, 0)  # ⭐ 将干扰类标签统一编码为0
        this_series.label = this_series.label.replace(drive, 1)     # ⭐ 将驾驶类标签统一编码为1

        series_labels = np.array(this_series.label)
        series_data = np.array(this_series.iloc[:, 3:])
        if norm:
            scaler = StandardScaler()
            series_data = scaler.fit_transform(series_data)  # ⭐ 对数据进行标准化处理
        data.append(pd.DataFrame({"data": [series_data],
                                  "label": [series_labels]}, index=[0]))
    data = pd.concat(data)
    data.reset_index(drop=True, inplace=True)
    return data


def load_activity(file_path, data_type, norm=True, verbose=1):

    column_names = ['series', 'label', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    df = pd.read_csv(file_path, header=None, names=column_names, comment=';')
    df.dropna(axis=0, how='any', inplace=True)
    LE = LabelEncoder()
    df['label'] = LE.fit_transform(df['label'])
    all_series = df.series.unique()
    train_series, test_series = train_test_split([x for x in range(len(all_series))], test_size=6, random_state=1)
    train_data = []
    print("[Data_Loader] Loading Train Data")
    for series in train_series:
        series = series + 1
        if verbose > 0:
            print("[Data_Loader] Processing series {}".format(series))
        this_series = df.loc[df.series == series].reset_index(drop=True)
        series_labels = np.array(this_series.label)
        series_data = np.array(this_series.iloc[:, 3:])
        if norm:
            scaler = StandardScaler()
            series_data = scaler.fit_transform(series_data)
        train_data.append(pd.DataFrame({"data": [series_data],
                                        "label": [series_labels]}, index=[0]))
    train_data = pd.concat(train_data)
    train_data.reset_index(drop=True, inplace=True)
    test_data = []
    print("[Data_Loader] Loading Test Data")
    for series in test_series:
        series = series + 1
        if verbose > 0:
            print("[Data_Loader] Processing series {}".format(series))
        this_series = df.loc[df.series == series].reset_index(drop=True)
        series_labels = np.array(this_series.label)
        series_data = np.array(this_series.iloc[:, 3:])
        if norm:
            scaler = StandardScaler()
            series_data = scaler.fit_transform(series_data)
        test_data.append(pd.DataFrame({"data": [series_data],
                                       "label": [series_labels]}, index=[0]))
    test_data = pd.concat(test_data)
    test_data.reset_index(drop=True, inplace=True)
    return train_data, test_data


def load_ford_data(file_path, data_type, norm=True, verbose=1):
    """
    加载并处理福特数据集，支持数据标准化和进度打印功能。

    Args:
        file_path (str): 福特数据集CSV文件路径
        data_type: 数据类型参数（未在函数内使用，可能为预留参数）
        norm (bool): 是否对数据进行标准化处理，默认为True
        verbose (int): 控制打印详细程度，>0时打印处理进度

    Returns:
        pd.DataFrame: 包含处理后的数据和标签的DataFrame，每行对应一个series的数据和标签
    """
    if verbose > 0:
        print("[Data_Loader] Loading data from {}".format(file_path))
    df = pd.read_csv(file_path)  # ⭐ 核心操作：读取原始CSV数据文件
    all_series = df.series.unique()
    data = []

    for series in all_series:
        if verbose > 0:
            print("[Data_Loader] Processing series {}".format(series))
        this_series = df.loc[df.series == series].reset_index(drop=True)
        series_labels = np.array(this_series.label)
        series_data = np.array(this_series.iloc[:, 3:])
        if norm:
            scaler = StandardScaler()
            series_data = scaler.fit_transform(series_data)  # ⭐ 核心操作：标准化数据
        data.append(pd.DataFrame({"data": [series_data],
                                  "label": [series_labels]}, index=[0]))
    data = pd.concat(data)
    data.reset_index(drop=True, inplace=True)

    return data