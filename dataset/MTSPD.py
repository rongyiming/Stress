import os
import json
import pandas as pd
import pickle
import numpy as np

frequency = 10  # 采样频率10Hz

def preprocess():
    root_folder = 'C:/Users/12992/Desktop/实验室/stress/code/watchdata/data'
    dict = {}
    files = os.listdir(root_folder)
    data_dict = {}
    for file in files:
        filename = os.path.basename(file)
        name = filename[:-5]
        print(name)
        print("name:",name)
        if name == 'index': continue
        path = os.path.join(root_folder, filename)
        print(path)
        data = pd.read_json(path)
        dict = {
            'data': [],
            'label': []
        }
        if name[0] == 'X' : 
            exp = 'TSST'
        elif name[0] == 'C':
            exp = 'Base'
        elif name[0] == 'L':
            exp = 'Water'
        elif name[0] == 'W':
            exp = 'Base'
        elif name[1] == 'c':
            exp = 'Base'
        else:
            exp = 'TSST'
        eplist = ['T1', 'T2', 'Exp', 'T3', 'T4', 'T5', 'T6', 'End']
        for i in range(len(eplist)):
            if i == 5:
                break
            ppg = data['ppg'][data['epoch']==eplist[i]].tolist()
            dict['data'].append(ppg)
            l = 0
            if eplist[i] == 'Exp' and exp == 'TSST':
                l = 1
            elif eplist[i] == 'Exp' and exp == 'Water':
                l = 2
            dict['label'].append(l)
        data_dict[name] = dict
    return data_dict

def normalize(data):
    """
    标准化数据，使其均值为0，标准差为1
    
    参数:
        data: 输入数据，numpy数组
        
    返回:
        标准化后的数据
    """
    all_data = np.concatenate([np.concatenate(d['data']) for d in data.values()])
    mean = np.mean(all_data)
    std = np.std(all_data)
    normalized_data = {}
    for participant, d in data.items():
        normalized_data[participant] = {
            'data': [(x - mean) / std for x in d['data']],
            'label': d['label']
        }
    return normalized_data

if __name__ == "__main__":
    data_dict = preprocess()
    n_data_dict = normalize(data_dict)
    with open('./dataset/MTSPD.pkl', 'wb') as f:
        pickle.dump(n_data_dict, f)