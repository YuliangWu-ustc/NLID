import os

IS_LINUX = (os.name == 'posix')
if IS_LINUX:
    import sys
    print(sys.path)
    # Add /usr/lib/python3/dist-packages/ to PYTHONPATH if the output of print(sys.path) does not mention it.
    sys.path.append("/usr/lib/python3/dist-packages/") 
    sys.path.append("/tmp/prophesee/py3venv/lib/python3.9/site-packages")

from metavision_core.event_io import EventsIterator
import numpy as np
import time
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
    

# os.chdir(os.path.dirname(os.path.abspath(__file__)))


dataset_name = 'water-ethanol-black-background'
# dataset_name = 'water-OCC(O)CO'
# dataset_name = 'chem-liquid'
# dataset_name = 'life-liquid'

_dataset_info_dict = {
    'water-ethanol-black-background':{
        'raw_path' : '/data/cls1-srv5-pool/Chem_liquid/water-ethanol-black-background/H1/',
        'dataset_path' : '/data/cls1-srv5-pool/Chem_liquid/water-ethanol-black-background/H1/processed/',
        'Results_path' : './Results/water-ethanol-black-background/',
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
        }
    },
    
    # 'water-ethanol-black-background':{
    #     'dataset_path' : '/data/cls1-srv5-pool/Chem_liquid/water-ethanol-black-background/H1/processed/',
    #     'Results_path' : './Results/water-ethanol-black-background/',
    #     'folder_labels_dict' : {
    #         'water_H1-50': 0,
    #         'ethanol_20_H1-50': 1,
    #         'ethanol_40_H1-50': 2,
    #         'ethanol_60_H1-50': 3,
    #         'ethanol_80_H1-50': 4,
    #         'ethanol_100_H1-50': 5,
    #     }
    # },
    'water-OCC(O)CO':{
        'raw_path' : '/data/cls1-srv5-pool/Chem_liquid/water-OCC(O)CO/H1/',
        'dataset_path' : '/data/cls1-srv5-pool/Chem_liquid/water-OCC(O)CO/H1/processed/',
        'Results_path' : './Results/water-OCC(O)CO/',
        'RAW_FOLDERS': [
            # 'water_H1-50',
            'C3H8O3_10_H1-50',
            'C3H8O3_20_H1-50',
            # 'C3H8O3_20_H1-50-wrong-volume',
            'C3H8O3_30_H1-50',
            # 'C3H8O3_40_H1-50',
            'C3H8O3_40_H1-50-B',
        ],
        'folder_labels_dict' : {
            # 'water_H1-50': 0,
            'C3H8O3_10_H1-50': 1,
            'C3H8O3_20_H1-50': 2,
            'C3H8O3_30_H1-50': 3,
            'C3H8O3_40_H1-50-B': 4,
        }
    },
    'chem-liquid':{
        'raw_path' : '/data/cls1-srv5-pool/Chem_liquid/chem-liquid/H1/',
        'dataset_path' : '/data/cls1-srv5-pool/Chem_liquid/chem-liquid/H1/processed/',
        'Results_path' : './Results/chem-liquid/',
        'RAW_FOLDERS' : [
            'DMSO-50',
            'EtOAc_H1-50',
            'BnOH_H1-50',
            'iPrOH_H1-50',
            'C6H14_H1-50-spill-manual',
            'NaOH_1M_H1-50',
        ],
        'folder_labels_dict' : {            
            'DMSO-50':              0, 
            'EtOAc_H1-50':          1, 
            # 'CH3COOCH2CH3_H1-50':   15, 
            'BnOH_H1-50':           2, 
            # 'PhCO_H1-50':           17, 
            'iPrOH_H1-50':          3, 
            # 'CC(O)C_H1-50':         15, 
            'NaOH_1M_H1-50':        5,
            
        }
    },
    'life-liquid':{
        'raw_path' : '/data/cls1-srv5-pool/Chem_liquid/life-liquid/H1/',
        'dataset_path' : '/data/cls1-srv5-pool/Chem_liquid/life-liquid/H1/processed/',
        'Results_path' : './Results/life-liquid/',
        'RAW_FOLDERS' : [
            'alcohol_56v_H1-50',
            'alcohol_15v_H1-50',
            'alcohol_vodka_40v_H1-50',
            'vinegar_9_H1-50',
            'medical_alcohol_75v_H1-50',
            'coconut_water_H1-50',
            'NaCl_1M_H1-50',
            'NaCl_2M_H1-50',
            # 'NaCl_10_H1-50',
            'glucose_10_H1-50',
            'NaHCO3_0.5M_H1-50',
        ],
        'folder_labels_dict' : {
            # 'alcohol_56v_H1-50':    0, 
            # 'alcohol_15v_H1-50':    1, 
            # 'alcohol_vodka_40v_H1-50': 2, 
            'vinegar_9_H1-50': 3, # 醋
            # 'medical_alcohol_75v_H1-50': 4, 
            # 'coconut_water_H1-50':  5, 
            # 'NaCl_1M_H1-50':        6, 
            # 'NaCl_2M_H1-50':        7, 
            # # 'NaCl_10_H1-50',
            # 'glucose_10_H1-50':     8, 
            # 'NaHCO3_0.5M_H1-50':    9, 
        }
    }
}

dataset_path = _dataset_info_dict[dataset_name]['dataset_path']
folder_labels_dict = _dataset_info_dict[dataset_name]['folder_labels_dict']
results_path = _dataset_info_dict[dataset_name]['Results_path']
raw_path = _dataset_info_dict[dataset_name]['raw_path']

# SELECTED_AREAS = [
#     (530, 440, 810, 450),(530, 460, 810, 470),(530, 480, 810, 490),   
#     (530, 500, 810, 510),(530, 520, 810, 530)
#     ]

# THIS IS FOR ROI EXP
SELECTED_AREAS = [
    # (530, 200, 810, 210), # y_division_val = 10, 1/54
    (530, 460, 810, 470), # y_division_val = 10, 1/54
    # (530, 480, 810, 490), # y_division_val = 10, 1/54
    # (530, 460, 810, 540), # y_division_val = 80, 1/6.75
    # (530, 270, 810, 540),  # y_division_val = 270, 1/2
    # (530, -1, 810, 600) # y_division_val = inf
    ]
DELTA_T_LIST = [
    10000, # 1/800
    1000, # 1/8000
    # 3000, # 1/2700
    # 30000, # 1/270
    # 100000 # 1/80
]

PROCESSED_PATH = os.path.join(raw_path, 'processed')
os.makedirs(PROCESSED_PATH, exist_ok=True)

# 创建输出文件夹
raw_folders = folder_labels_dict.keys()
for folder in raw_folders:
    os.makedirs(os.path.join(PROCESSED_PATH, folder), exist_ok=True)

# 处理每个 .raw 文件
def load_raw_data_to_ev_freq(raw_file, select_area, delta_t, tag='', output_stream=False):
    mv_iterator = EventsIterator(input_path=raw_file, delta_t=delta_t)
    
    ev_freq_selected = []
    
    if output_stream:
        selected_x = []
        selected_y = []
        selected_t = []
        selected_p = []
    
    for i, evs in enumerate(mv_iterator):
        
        if i < 5e6 / delta_t: # 前5000帧数据不处理
            continue
        
        if not len(evs):
            ev_freq_selected.append(0)
            continue
        
        if (i + 1) % 5000 == 0:
            print(f'\tprocessing {i + 1} pkg, {len(evs)} events')
        
        x = evs['x']
        y = evs['y']
        t = evs['t']
        p = evs['p']
        
        ev_freq_selected.append(len(evs[(x > select_area[0]) & (x < select_area[2]) & (y > select_area[1]) & (y < select_area[3])]))
        
        if output_stream:
            selected_x.extend(x[(x > select_area[0]) & (x < select_area[2]) & (y > select_area[1]) & (y < select_area[3])])
            selected_y.extend(y[(x > select_area[0]) & (x < select_area[2]) & (y > select_area[1]) & (y < select_area[3])])
            selected_t.extend(t[(x > select_area[0]) & (x < select_area[2]) & (y > select_area[1]) & (y < select_area[3])])
            selected_p.extend(p[(x > select_area[0]) & (x < select_area[2]) & (y > select_area[1]) & (y < select_area[3])])
    
    if output_stream:
        ev_stream_selected = np.vstack((selected_x, selected_y, selected_t, selected_p))
        ev_stream_selected = ev_stream_selected.T
        print(f'\tev_stream_selected.shape: {ev_stream_selected.shape}')  
    print(f'\tev_freq_selected.len: {len(ev_freq_selected)}')
    
    # 获取输出文件夹路径
    raw_folder = os.path.basename(os.path.dirname(raw_file))
    output_folder = os.path.join(PROCESSED_PATH, raw_folder)
    
    # 保存处理后的数据
    # if output_stream:
    #     np.save(os.path.join(output_folder, f'ev_stream_select_{select_area}_t{delta_t}_{tag}.npy'), ev_stream_selected)
    # np.save(os.path.join(output_folder, f'ev_freq_select_{select_area}_t{delta_t}_{tag}.npy'), ev_freq_selected)

    return ev_freq_selected

for delta_t in DELTA_T_LIST:
    format_file = os.path.join(PROCESSED_PATH, f'.format.txt')
    with open(format_file, 'a+') as f:
        f.write(r'ev_stream_select_{select_area}_t{delta_t}_{tag}.npy')
        f.write('\n')
        f.write(r'ev_freq_select_{select_area}_t{delta_t}_{tag}.npy')
        f.write('\n')
        f.write(f'SELECTED_AREAS: {SELECTED_AREAS}')
        f.write('\n')
        f.write(f'DELTA_T: {delta_t}')
        f.write('\n')
        f.write(f'RAW_FOLDERS: {raw_folders}')
        f.write('\n')
        f.write(r"TAGS: os.path.basename(rawfile).split('.')[0]")
        f.write('\n')
    # exit()
    for SELECTED_AREA in SELECTED_AREAS:
        for folder in raw_folders:
            raw_files = [os.path.join(raw_path, folder, f'event{i}.raw') for i in range(1, 51)]
            ev_freq_series_list = []
            for rawfile in raw_files:
                print(f'processing {rawfile} // delta_t: {delta_t} // SELECTED_AREA: {SELECTED_AREA} // folder: {folder}')
                ev_freq_series_list.append(load_raw_data_to_ev_freq(rawfile, select_area=SELECTED_AREA, delta_t=delta_t, tag=os.path.basename(rawfile).split('.')[0]))
            
            ev_freq_series_len_list = [len(ev_freq) for ev_freq in ev_freq_series_list]
            if((max(ev_freq_series_len_list) - min(ev_freq_series_len_list) ) / np.mean(ev_freq_series_len_list) > 0.1):
                print(f'{folder} ev_freq_series_lengths vary too much')
                print(f'max: {max(ev_freq_series_len_list)}, min: {min(ev_freq_series_len_list)}, mean: {np.mean(ev_freq_series_len_list)}')
                raise ValueError('ev_freq_series_lengths vary too much')
            
            max_len = max(ev_freq_series_len_list)
            for i in range(len(ev_freq_series_list)):
                if len(ev_freq_series_list[i]) < max_len:
                    ev_freq_series_list[i].extend([0] * (max_len - len(ev_freq_series_list[i])))
            
            ev_freq_series_list = np.array(ev_freq_series_list)
            np.save(os.path.join(PROCESSED_PATH, f'ev_freq_series_list_{folder}_select_{SELECTED_AREA}_t{delta_t}.npy'), ev_freq_series_list)
            print(f'ev_freq_series_list.shape: {ev_freq_series_list.shape}')
            print(f'folder {folder} done')
            print('--------------------------------')
            # break
        print(f'SELECTED_AREA {SELECTED_AREA} done')
        print('#####################################')
    print(f'delta_t {delta_t} done')
    print('#####################################')

