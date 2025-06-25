import os
import json
import torch
import numpy as np
import logging
import zipfile
import requests
from datetime import datetime
from torch.utils.data import Dataset

from Dataset import load_segment_data, load_UEA_data

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def Setup(args):
    """
    初始化实验配置并创建必要的目录结构。

    Args:
        args: 从argparse获取的参数对象

    Returns:
        dict: 包含完整配置信息的字典，包括创建的目录路径
    """
    config = args.__dict__  # ⭐ 将参数对象转换为配置字典
    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_dir = os.path.join(output_dir, config['data_path'], initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}' as a configuration.json".format(output_dir))

    return config


def create_dirs(dirs):
    """
    创建指定的多个目录（如果目录不存在）。

    Args:
        dirs (list): 需要创建的目录路径列表

    Returns:
        int: 操作状态码（0表示成功，-1表示失败）
    
    Raises:
        Exception: 当目录创建失败时抛出异常并打印错误信息
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)  # ⭐ 核心代码：实际创建目录的操作
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def Initialization(config):
    """
    初始化训练环境配置，包括随机种子和设备选择。

    Args:
        config (dict): 配置字典，包含以下键：
            - seed (int): 随机种子，为None时不设置
            - gpu (str): GPU设备编号，'-1'表示不使用GPU

    Returns:
        torch.device: 初始化后的计算设备对象
    """
    if config['seed'] is not None:
        torch.manual_seed(config['seed'])  # ⭐ 设置PyTorch的随机种子保证实验可复现
    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))
    return device


def Data_Loader(config):
    """
    根据配置信息加载相应类型的数据集

    Args:
        config (dict): 包含数据集路径等配置信息的字典，其中data_dir键指定数据路径
            - 当路径包含'Segmentation'时加载分割数据集
            - 否则加载UEA格式的时间序列数据

    Returns:
        object: 加载完成的数据集对象
    """
    if config['data_dir'].split('/')[1] == 'Segmentation':
        Data = load_segment_data.load(config)  # ⭐ 加载分割数据集（HAR WISDM V2和Ford数据集）
    else:
        Data = load_UEA_data.load(config)  # ⭐ 加载UEA格式的*.ts时间序列数据
    return Data


def Data_Verifier(config):
    """
    验证数据目录是否存在，若不存在则创建目录；对于UEA数据集自动下载并解压。

    Args:
        config (dict): 包含数据路径等配置信息的字典，其中data_path为关键字段。

    Returns:
        None: 直接修改传入的config字典中的data_path字段（当处理UEA数据集时）。
    """
    if not os.path.exists(config['data_path']):
        os.makedirs(os.path.join(os.getcwd(), config['data_path']))  # ⭐ 创建数据存储目录
    directories = [name for name in os.listdir(config['data_path']) if os.path.isdir(os.path.join(config['data_path'], name))]

    if directories:
        print(f"The {config['data_path'].split('/')[1]} data is already existed")
    else:
        if config['data_path'].split('/')[1] == 'UEA':
            file_url = 'http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip'
            Downloader(file_url, 'UEA')  # ⭐ 下载UEA数据集压缩包

    if config['data_path'].split('/')[1] == 'UEA':
        config['data_path'] = os.path.join(config['data_path'], 'Multivariate_ts')  # ⭐ 更新UEA数据集解压路径


def Downloader(file_url, problem):
    """
    从指定URL下载数据集文件并解压到本地目录

    Args:
        file_url (str): 要下载的文件URL地址
        problem (str): 数据集名称/问题名称，用于创建存储目录

    Returns:
        None
    """
    # Define the path to download
    path_to_download = os.path.join('Dataset/', problem)
    # Send a GET request to download the file
    response = requests.get(file_url, stream=True)
    # Check if the request was successful
    if response.status_code == 200:
        # Save the downloaded file
        file_path = os.path.join(path_to_download, 'Multivariate2018_ts.zip')
        with open(file_path, 'wb') as file:
            # Track the progress of the download
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024 * 100  # 1KB
            downloaded_size = 0

            for data in response.iter_content(block_size):
                file.write(data)  # ⭐ 核心代码：将下载的数据块写入本地文件
                downloaded_size += len(data)

                # Calculate the download progress percentage
                progress = (downloaded_size / total_size) * 100

                # Print the progress message
                print(f' Download in progress: {progress:.2f}%')

        # Extract the contents of the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(path_to_download)

        # Remove the downloaded zip file
        os.remove(file_path)

        print(f'{problem} Datasets downloaded and extracted successfully.')
    else:
        print(f'Failed to download the {problem} please update the file_url')
    return


class dataset_class(Dataset):

    def __init__(self, data, label):
        super(dataset_class, self).__init__()

        self.feature = data
        self.labels = label.astype(np.int32)

    def __getitem__(self, ind):

        x = self.feature[ind]
        x = x.astype(np.float32)

        y = self.labels[ind]  # (num_labels,) array

        data = torch.tensor(x)
        label = torch.tensor(y)

        return data, label, ind

    def __len__(self):
        return len(self.labels)



class dataset_class_regr(Dataset):

    def __init__(self, data, label):
        super(dataset_class_regr, self).__init__()

        self.feature = data
        self.labels = label.astype(np.float32)

    def __getitem__(self, ind):

        x = self.feature[ind]
        x = x.astype(np.float32)

        y = self.labels[ind]  # (num_labels,) array

        data = torch.tensor(x)
        label = torch.tensor(y)

        return data, label, ind

    def __len__(self):
        return len(self.labels)


