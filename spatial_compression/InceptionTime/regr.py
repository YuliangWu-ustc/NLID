from utils.dataloader.TSC_data_loader import TSC_data_loader, UCI_data_loader, MITdb_data_loader, my_TSC_data_loader
import os
from os.path import dirname
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 添加标签标准化（重要）
from sklearn.preprocessing import StandardScaler
# 修改导入路径（保持与修改后的训练器一致）
from Regressor.Easy_use_trainner_Regression import Easy_use_trainner  # 修改后的回归训练器
from Regressor.InceptionNet_Regression import InceptionNet  # 修改后的回归模型
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


device = "cuda:4"

# THIS IS FOR ROI EXP
SELECTED_AREAS = [
    # (530, 200, 810, 210), # y_division_val = 10, 1/54
    (530, 480, 810, 490), # y_division_val = 10, 1/54
    # (530, 460, 810, 540), # y_division_val = 80, 1/6.75
    (530, 270, 810, 540),  # y_division_val = 270, 1/2
    # (530, -1, 810, 600) # y_division_val = inf
    ]
DELTA_T_LIST = [
    1000, # 1/8000
    # 3000, # 1/2700
    10000, # 1/800
    # 30000, # 1/270
    # 100000 # 1/80
]

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

Result_log_path = './Results/' + dataset_name + '/'

###################################################################

# 初始化存储结果的矩阵
mse_matrix = np.zeros((len(SELECTED_AREAS), len(DELTA_T_LIST)))
mae_matrix = np.zeros((len(SELECTED_AREAS), len(DELTA_T_LIST)))
r2_matrix = np.zeros((len(SELECTED_AREAS), len(DELTA_T_LIST)))
maxae_matrix = np.zeros((len(SELECTED_AREAS), len(DELTA_T_LIST)))  # 新增MaxAE矩阵

for i, DELTA_T in enumerate(DELTA_T_LIST):
    for j, SELECTED_AREA in enumerate(SELECTED_AREAS):
        
        plt.figure(figsize=(6, 5))
        selected_area_list = [SELECTED_AREA]

        # 假设数据加载函数已修改为返回连续值标签
        X_train, y_train, X_test, y_test = my_TSC_data_loader(
            dataset_path, 
            folder_labels_dict,
            selected_area_list, 
            DELTA_T
        )

        # 在调用fit之前添加维度处理
        y_train = y_train.reshape(-1, 1)  # (440,) -> (440,1)
        y_test = y_test.reshape(-1, 1)    # (110,) -> (110,1)

        # # # # 修改后的标准化代码
        # y_scaler = StandardScaler()
        # y_train_scaled = y_scaler.fit_transform(y_train).flatten()  # 保持一维输入
        # y_test_scaled = y_scaler.transform(y_test).flatten()
        # # # print(f'X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}')
        # # print(f'y_train_scaled.shape: {y_train_scaled.shape}, y_test_scaled.shape: {y_test_scaled.shape}')

        # # # print(f'y_test: {y_test}, y_test.shape: {y_test.shape}')
        # # print(f'y_test_scaled: {y_test_scaled}, y_test_scaled.shape: {y_test_scaled.shape}')
        # y_train = y_train_scaled
        # y_test = y_test_scaled
        y_train = y_train.flatten()
        y_test = y_test.flatten()

        os.makedirs(Result_log_path, exist_ok=True)

        # 修改模型构建参数
        inceptionNet_model = InceptionNet(
            input_channle_size=1,
            output_dim=1  # 回归输出维度（根据需求修改）
        )

        # 修改训练器配置
        easy_use_trainner = Easy_use_trainner(
            Result_log_folder=Result_log_path,
            dataset_name=f'{dataset_name}_select_{SELECTED_AREA}_t{DELTA_T}',
            device=device,
            max_epoch=600,
            print_result_every_x_epoch=5
        )

        easy_use_trainner.get_model(inceptionNet_model)

        # 训练时自动处理标准化后的数据
        easy_use_trainner.fit(X_train, y_train, X_test, y_test)

        # 预测时获取回归结果
        y_predict_scaled = easy_use_trainner.predict(X_test)
        
        # 逆标准化预测结果
        # y_predict = y_scaler.inverse_transform(y_predict_scaled.reshape(-1, 1)).flatten()
        # y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_predict = y_predict_scaled
        y_test_orig = y_test
        print(f'y_test_orig: {y_test_orig}, y_test_orig.shape: {y_test_orig.shape}')
        print(f'y_predict: {y_predict}, y_predict.shape: {y_predict.shape}')
        # 输出y_predict的各指标
        print(f'y_predict的均值: {np.mean(y_predict)}')
        print(f'y_predict的方差: {np.var(y_predict)}')
        print(f'y_predict的最小值: {np.min(y_predict)}')
        print(f'y_predict的最大值: {np.max(y_predict)}')
        print(f'y_predict的中位数: {np.median(y_predict)}')
        print(f'y_predict的众数: {np.argmax(y_predict)}')
        
        # 计算回归指标
        mse = mean_squared_error(y_test_orig, y_predict)
        mae = mean_absolute_error(y_test_orig, y_predict)
        r2 = r2_score(y_test_orig, y_predict)
        maxae = np.max(np.abs(y_test_orig - y_predict))  # 计算MaxAE
        
        # 将结果存储到矩阵中
        mse_matrix[j, i] = mse
        mae_matrix[j, i] = mae
        r2_matrix[j, i] = r2
        maxae_matrix[j, i] = maxae  
        
        print('True values:', y_test_orig)
        print('Predictions:', y_predict)
        print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, MaxAE: {maxae:.4f}')

        # 修改结果记录方式
        result_dir = os.path.join(Result_log_path, f'{dataset_name}_select_{SELECTED_AREA}_t{DELTA_T}')
        os.makedirs(result_dir, exist_ok=True)
        
        # 保存完整预测结果
        np.savez(os.path.join(result_dir, 'predictions.npz'),
                 y_true=y_test_orig,
                 y_pred=y_predict)

        # 记录评估指标
        with open(os.path.join(result_dir, 'regression_metrics.txt'), 'w') as f:
            f.write(f'selected_area: {SELECTED_AREA}\n')
            f.write(f'delta_t: {DELTA_T}\n')
            f.write(f'MSE: {mse:.4f}\n')
            f.write(f'MAE: {mae:.4f}\n')
            f.write(f'R2: {r2:.4f}\n')
            f.write(f'MaxAE: {maxae:.4f}\n')  # 记录MaxAE

        # 可选：可视化预测结果
        plt.subplot(len(DELTA_T_LIST), len(SELECTED_AREAS), i*len(SELECTED_AREAS)+j+1)
        plt.suptitle(f'{dataset_name} - {SELECTED_AREA} - {DELTA_T}')
        plt.scatter(y_test_orig, y_predict, alpha=0.5)
        plt.plot([y_test_orig.min(), y_test_orig.max()], 
                [y_test_orig.min(), y_test_orig.max()], 'k--')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title(f'Regression Results (R2={r2:.2f})')
        
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.savefig(os.path.join(Result_log_path, f'prediction_plot_{SELECTED_AREA}_{DELTA_T}.png'))
        plt.show()


# 绘制MSE热力图
plt.figure(figsize=(10, 6))
sns.heatmap(mse_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=DELTA_T_LIST, yticklabels=SELECTED_AREAS)
plt.title('MSE Over different DELTA_T and SELECTED_AREA')
plt.xlabel('DELTA_T')
plt.ylabel('SELECTED_AREA')
plt.savefig(os.path.join(Result_log_path, f'mse_heatmap_{SELECTED_AREAS}_{DELTA_T_LIST}.png'))
plt.show()

# 绘制MAE热力图
plt.figure(figsize=(10, 6))
sns.heatmap(mae_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=DELTA_T_LIST, yticklabels=SELECTED_AREAS)
plt.title('MAE Over different DELTA_T and SELECTED_AREA')
plt.xlabel('DELTA_T')
plt.ylabel('SELECTED_AREA')
plt.savefig(os.path.join(Result_log_path, f'mae_heatmap_{SELECTED_AREAS}_{DELTA_T_LIST}.png'))
plt.show()

# 绘制R2热力图
plt.figure(figsize=(10, 6))
sns.heatmap(r2_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=DELTA_T_LIST, yticklabels=SELECTED_AREAS)
plt.title('R2 Over different DELTA_T and SELECTED_AREA')
plt.xlabel('DELTA_T')
plt.ylabel('SELECTED_AREA')
plt.savefig(os.path.join(Result_log_path, f'r2_heatmap_{SELECTED_AREAS}_{DELTA_T_LIST}.png'))
plt.show()

# 绘制MaxAE热力图
plt.figure(figsize=(10, 6))
sns.heatmap(maxae_matrix, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=DELTA_T_LIST, yticklabels=SELECTED_AREAS)
plt.title('MaxAE Over different DELTA_T and SELECTED_AREA')
plt.xlabel('DELTA_T')
plt.ylabel('SELECTED_AREA')
plt.savefig(os.path.join(Result_log_path, f'maxae_heatmap_{SELECTED_AREAS}_{DELTA_T_LIST}.png'))
plt.show()