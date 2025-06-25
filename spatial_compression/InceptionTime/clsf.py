from utils.dataloader.TSC_data_loader import TSC_data_loader, UCI_data_loader, MITdb_data_loader, my_TSC_data_loader
import os
from os.path import dirname
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


device = "cuda:0"

# THIS IS FOR ROI EXP
SELECTED_AREAS = [
    # (530, 200, 810, 210), # y_division_val = 10, 1/54
    (530, 480, 810, 490), # y_division_val = 10, 1/54
    (530, 460, 810, 540), # y_division_val = 80, 1/6.75
    (530, 270, 810, 540),  # y_division_val = 270, 1/2
    (530, -1, 810, 600) # y_division_val = inf
    ]
DELTA_T_LIST = [
    1000, # 1/8000
    3000, # 1/2700
    10000, # 1/800
    30000, # 1/270
    # 100000 # 1/80
]

ARCHIVE_PATH = '/data/cls1-srv5-pool/Chem_liquid/'

# dataset_name = 'regression'
dataset_name = 'classification'
# dataset_name = 'all'

_dataset_info_dict = {
    'regression':{
        'dataset_path' : os.path.join(ARCHIVE_PATH, 'regression/processed/'),
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

for DELTA_T in DELTA_T_LIST:
    for SELECTED_AREA in SELECTED_AREAS:
        
        selected_area_list = [SELECTED_AREA]

        X_train, y_train, X_test, y_test = my_TSC_data_loader(dataset_path, folder_labels_dict, selected_area_list, DELTA_T)

        from Classifier.Easy_use_trainner import Easy_use_trainner
        from Classifier.InceptionNet import InceptionNet

        Result_log_path = './Results/' + dataset_name + '/'
        os.makedirs(Result_log_path, exist_ok=True)

        # build model
        inceptionNet_model = InceptionNet(input_channle_size = 1,nb_classes = max(y_train)+1)

        # build trainner
        easy_use_trainner = Easy_use_trainner(Result_log_folder = Result_log_path,
                                            dataset_name = f'{dataset_name}_select_{SELECTED_AREA}_t{DELTA_T}',
                                            device = device,
                                            max_epoch=600,
                                            print_result_every_x_epoch=5)

        # put model to trainner
        easy_use_trainner.get_model(inceptionNet_model)


        # fit data
        easy_use_trainner.fit(X_train, y_train, X_test, y_test)


        # test
        y_predict = easy_use_trainner.predict(X_test)

        print('correct:',y_test)
        print('predict:',y_predict)
        acc = accuracy_score(y_predict, y_test)
        print(acc)
        
        with open(Result_log_path + f'{dataset_name}_select_{SELECTED_AREA}_t{DELTA_T}/' + 'label-accuracy.txt', 'w+') as f:
            f.write(f'selected_area: {SELECTED_AREA}\n')
            f.write(f'delta_t: {DELTA_T}\n')
            f.write(f'  accuracy: {acc}\n')
            for i in range(max(y_train)+1):
                f.write(f'    label {i} accuracy: {accuracy_score(y_predict[y_test==i], y_test[y_test==i])}\n')
        
        with open(Result_log_path + f'{dataset_name}_select_{SELECTED_AREA}_t{DELTA_T}/' + 'accuracy.txt', 'w+') as f:
            f.write(f'selected_area: {SELECTED_AREA}\n')
            f.write(f'delta_t: {DELTA_T}\n')
            f.write(f'  accuracy: {acc}\n')

