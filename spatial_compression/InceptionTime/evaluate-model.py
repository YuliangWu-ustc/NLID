import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from utils.dataloader.TSC_data_loader import my_TSC_data_loader, simple_TSC_data_loader
from Classifier.InceptionNet_gradcam import InceptionNet
from Classifier.Easy_use_trainner import Easy_use_trainner

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

results_path = _dataset_info_dict[dataset_name]['Results_path']
    
# 遍历每个选择区域
Result_log_path = './Results/' + dataset_name + '/'
plt.figure(figsize=(30, 30))
for i, DELTA_T in enumerate(DELTA_T_LIST):
    for j, SELECTED_AREA in enumerate(SELECTED_AREAS):
        print(f'\n处理区域: {SELECTED_AREA}')
        
        # 加载数据
        selected_area_list = [SELECTED_AREA]
        _, y_train, X_test, y_test = my_TSC_data_loader(dataset_path, folder_labels_dict, selected_area_list, DELTA_T)
        
        # if folder_labels_dict_ex is not None:
        #     X_test_ex, y_test_ex = simple_TSC_data_loader(dataset_path, folder_labels_dict_ex, selected_area_list, DELTA_T)
            
        #     # print(f'X_test.shape: {X_test.shape}')
        #     # print(f'X_test_ex.shape: {X_test_ex.shape}')
        #     # print(f'y_test.shape: {y_test.shape}')
        #     # print(f'y_test_ex.shape: {y_test_ex.shape}')
        #     # 将所有ev_freq_series_list中的元素长度统一
        #     max_len = max(X_test.shape[1], X_test_ex.shape[1])
        #     if X_test.shape[1] < max_len:
        #         # 使用 np.pad 在 X_test 的末尾填充 0
        #         X_test = np.pad(X_test, ((0, 0), (0, max_len - X_test.shape[1])), mode='constant', constant_values=0)
        #         # print(f'X_test.shape: {X_test.shape}')
        #     if X_test_ex.shape[1] < max_len:
        #         # 使用 np.pad 在 X_test_ex 的末尾填充 0
        #         X_test_ex = np.pad(X_test_ex, ((0, 0), (0, max_len - X_test_ex.shape[1])), mode='constant', constant_values=0)
        #         # print(f'X_test_ex.shape: {X_test_ex.shape}')

        #     # print(f'y_test_ex: {y_test_ex}')
            
        #     # 合并 X_test 和 X_test_ex
        #     X_test = np.concatenate([X_test, X_test_ex], axis=0)
        #     y_test = np.concatenate([y_test, y_test_ex], axis=0)
            
        #     # print(f'X_test.shape: {X_test.shape}')
        #     # print(f'y_test.shape: {y_test.shape}')
        
        # # raise Exception('stop')
    
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
        
        # 进行预测
        y_pred = trainer.predict(X_test)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 绘制混淆矩阵
        plt.subplot(len(DELTA_T_LIST), len(SELECTED_AREAS), i*len(SELECTED_AREAS)+j+1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=list(folder_labels_dict.keys()),
                    yticklabels=list(folder_labels_dict.keys()))
        
        plt.title(f'Confusion Matrix for {SELECTED_AREA_DESCRIPTION[SELECTED_AREA]}|{DELTA_T}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        # plt.xticks(rotation=45)
        # plt.yticks(rotation=45)
        # # 修改这里：横轴标签水平显示（rotation=0），纵轴标签垂直显示（rotation=90）
        # plt.xticks(rotation=0, ha='center')  # ha='center' 确保标签居中对齐
        # plt.yticks(rotation=0, va='center')  # va='center' 确保标签居中对齐

        # 调整布局以确保所有标签都可见
        plt.tight_layout()

        
        # 保存混淆矩阵图
        
        # 计算每个类别的准确率
        class_acc = cm.diagonal() / cm.sum(axis=1)
        print("\n各类别准确率:")
        for label_name, acc in zip(folder_labels_dict.keys(), class_acc):
            print(f"{label_name}: {acc:.4f}")
        
        # 计算总体准确率
        total_acc = cm.diagonal().sum() / cm.sum()
        print(f"\n总体准确率: {total_acc:.4f}")
        
plt.savefig(Result_log_path + f'/eval-test.png')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
    