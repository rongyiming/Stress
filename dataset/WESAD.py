import pandas as pd
import os
import numpy as np
import pickle
from collections import Counter

root = "D:\data\WESAD"

frequency = 64  # 采样频率64Hz

new_fs = 32

def interp_to_10(data):
    """
    将数据插值到10Hz采样率
    
    参数:
        data: 输入数据，numpy数组
        
    返回:
        插值后的数据
    """
    data_10hz = []
    for x in data:
        t_original = np.arange(len(x)) / frequency
        total_time = t_original[-1]  # 原始数据总时长（此处为10秒）
        t_target = np.arange(0, total_time, 1/new_fs)
        data_10hz.append(np.interp(t_target, t_original, x))
    return data_10hz

def list_files_in_folder(folder_path, recursive=False):
    """
    列出文件夹中的所有文件
    
    参数:
        folder_path: 文件夹路径
        recursive: 是否递归列出子文件夹中的文件，默认为False
    """
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return
    
    if not os.path.isdir(folder_path):
        print(f"错误: '{folder_path}' 不是一个文件夹")
        return
    
    file_list = []
    
    # 递归遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            if dir == "data":
                continue
            dir_path = os.path.join(root, dir)
            file_list.append(dir_path)

        # 如果不递归，只处理当前文件夹
        if not recursive:
            break
    
    return file_list

def preprocess():
    root_folder = root
    data_dict = {}
    files = list_files_in_folder(root_folder)
    print(files)
    for file in files:
        dict = {
            'data': [],
            'label': []
        }
        name = os.path.basename(file)
        print("name:",name)
        path = os.path.join(root, name, f'{name}.pkl')
        print("path:",path)
        with open(path, 'rb') as file:
            # 加载文件内容
            data = pickle.load(file, encoding='latin-1')
        print(len(data['signal']['wrist']['BVP'])/64)
        print(len(data['label'])/700)
        lastj = 0
        lasti = 0
        for i in range(len(data['label'])):
            j = i*64//700
            if data['label'][i] != data['label'][lasti]:
                newlabel = 0
                if data['label'][lasti] == 2:
                    newlabel = 1
                elif data['label'][lasti] >= 5 or data['label'][lasti] == 0:
                    lasti = i
                    lastj = j
                    continue
                dict['label'].append(newlabel)
                dict['data'].append([x[0] for x in data['signal']['wrist']['BVP'][lastj:j]])
                lasti = i
                lastj = j

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
            'data': interp_to_10([(x - mean) / std for x in d['data']]),
            'label': d['label']
        }
    return normalized_data

if __name__ == "__main__":
    data = preprocess()
    n_data = normalize(data)
    with open(f'./dataset/WESAD_{new_fs}Hz.pkl', 'wb') as f:
        pickle.dump(n_data, f)