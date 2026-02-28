import pandas as pd
import os
import numpy as np
import pickle

file = "D:\data\CLAS_Database\CLAS_Database\CLAS"

participants = [f"Part{x}" for x in range(1, 61)]

frequency = 256  # 采样频率256Hz

new_fs = 32

def interp_to_10(data):
    """
    将数据插值到new_fs采样率
    
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

def find_files_with_prefix(folder_path, prefix):
    """
    从指定文件夹中查找前缀为指定字符串的文件
    
    参数:
        folder_path: 要搜索的文件夹路径
        prefix: 文件名前缀，默认为"name"
        
    返回:
        符合条件的文件路径列表
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return []
    
    # 检查是否是文件夹
    if not os.path.isdir(folder_path):
        print(f"错误: '{folder_path}' 不是一个文件夹")
        return []
    
    # 存储找到的文件路径
    matched_files = []
    
    # 遍历文件夹中的所有项目
    for item in os.listdir(folder_path):
        # 构建完整路径
        item_path = os.path.join(folder_path, item)
        
        # 只处理文件，不处理子文件夹
        if os.path.isfile(item_path):
            # 检查文件名是否以指定前缀开头
            if item.startswith(prefix):
                matched_files.append(item_path)
    
    return matched_files

def load_data(participant, file_path=file):
    """
    读取指定参与者的PPG数据
    
    参数:
        participant: 参与者名称
        file_path: 要搜索的文件夹路径
        
    返回:
        符合条件的数据列表
    """
    # 读取Block_details文件，获取所有的EDA&PPG文件名
    name = os.path.join(file_path, "Block_details", f"{participant}_Block_Details.csv")
    if not os.path.exists(name):
        print(f"错误: 文件夹 '{name}' 不存在")
        return {}
    block_details = pd.read_csv(os.path.join(file_path, "Block_details", f"{participant}_Block_Details.csv"))
    # 遍历每个文件，读取数据
    dict = {
        'data': [],
        'label': []
    }
    for index, row in block_details.iterrows():
        block_file_name = row['EDA&PPG File']
        name = block_file_name.split('.')[0]
        
        block_folders = os.path.join(file_path, "Participants", participant, "by_block")
        block_file_list = find_files_with_prefix(block_folders, prefix=name)
        if len(block_file_list) == 0:
            print(f"错误: 文件夹 '{block_folders}' 中未找到前缀为 '{name}' 的文件")
            return {}
        block_file = block_file_list[0]
        data = pd.read_csv(block_file)

        
        # if row['Block Type'] in ['Math Test', 'Stroop Test', 'IQ Test']:
        if row['Block Type'] in ['Math Test', 'IQ Test']:
            dict['data'].append(data['ppg'].to_numpy())
            dict['label'].append(1)
        elif row['Block Type'] in ['Baseline', 'Neutral']:
            dict['data'].append(data['ppg'].to_numpy())
            dict['label'].append(0)

    return dict

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
    data_dict = {}
    for participant in participants:
        tmpdict = load_data(participant)
        if tmpdict:
            data_dict[participant] = tmpdict
    n_data_dict = normalize(data_dict)
    with open(f'./dataset/CLAS_{new_fs}Hz.pkl', 'wb') as f:
        pickle.dump(n_data_dict, f)
