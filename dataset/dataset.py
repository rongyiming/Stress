#CLAS:"C:\Users\12992\Desktop\实验室\stress_pretrain\dataset\CLAS.pkl"
#WESAD:"C:\Users\12992\Desktop\实验室\stress_pretrain\dataset\WESAD.pkl"
#MTSPD:"C:\Users\12992\Desktop\实验室\stress_pretrain\dataset\MTSPD.pkl"
import os
import pandas as pd
import numpy as np
import pickle
import torch
import random
from torch.utils.data import DataLoader, Dataset
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
from dataset.Feature import preprocess_ppg, calculate_ipa, calculate_sqi
from sklearn.model_selection import train_test_split
from dataset.individual import individual_feature_pipeline
from dataset.individual_test import new_individual
from numpy import polyfit

class NewDataset(Dataset):
    def __init__(self, data, labels, dlabels, finetune=False, individual=None):
        self.data = torch.tensor(data)          # 数据（如：图像路径列表/NumPy数组）
        if finetune:
            self.labels = torch.tensor(labels, dtype=torch.long)   # 分类标签
            self.individual = torch.tensor(individual, dtype=torch.float)
        else:
            self.labels = torch.tensor(labels, dtype=torch.float)   
            max = self.labels.max(dim=0, keepdim=True)
            min = self.labels.min(dim=0, keepdim=True)
            self.labels = (self.labels - min.values) / (max.values - min.values)
            self.individual = None
        self.dlabels = torch.tensor(dlabels, dtype=torch.float)    # next token标签

    def __len__(self):
        return len(self.data)     # 返回数据集总大小

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        dlabel = self.dlabels[idx]
        if self.individual is not None:
            individual = self.individual[idx]
            return sample, label, dlabel, individual
        else:
            individual = None
            return sample, label, dlabel
        

# 信号预处理（带通滤波：去除基线漂移和高频噪声）
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    """设计并应用巴特沃斯带通滤波器"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)  # 零相位滤波，避免相位偏移

def preprocess_ppg(ppg_data, fs=32):
    """
    PPG数据预处理：去除基线漂移（3阶多项式拟合）
    :param ppg_data: 输入的PPG原始数据，长度应为3840（2*60*32）
    :param fs: 采样率，固定为32Hz
    :return: 去基线后的PPG数据
    """
    # 生成时间轴
    ppg_data = np.array(ppg_data)
    ppg_data = butter_bandpass_filter(
        data=ppg_data,
        lowcut=0.5,
        highcut=4.0,
        fs=fs,
        order=2
    )
    t = np.arange(len(ppg_data)) / fs
    # 3阶多项式拟合基线
    baseline = polyfit(t, ppg_data, 3)
    baseline_curve = np.polyval(baseline, t)
    # 去除基线漂移
    ppg_clean = ppg_data - baseline_curve
    return ppg_clean


def calculate_hr_hrv(ppg_signal, fs = 10):
    """
    从10Hz PPG信号计算心率和HRV时间域指标
    
    参数:
        ppg_signal: 1D numpy数组，原始PPG信号（采样率10Hz）
        fs: 采样率，固定为10Hz（可省略）
    
    返回:
        以下指标（异常时为None）：
            - heart_rate: 心率（次/分）
            - sdnn: 所有PP间期的标准差（ms）
            - rmssd: 相邻PP间期差值的均方根（ms）
            - nn50: 相邻PP间期差值>50ms的数量
            - pnn50: NN50占总间期数的百分比（%）
    """
    # 初始化结果字典
    result = {
        'heart_rate': None,
        'sdnn': None,
        'rmssd': None,
        'nn50': None,
        'pnn50': None,
        'sqi': None
    }
    
    # 1. 检查输入信号有效性
    if len(ppg_signal) < fs * 10:  # 至少需要10秒数据（10Hz×10s=100个点）
        print("警告：信号长度不足10秒，结果可能不可靠")
    if not isinstance(ppg_signal, np.ndarray) or ppg_signal.ndim != 1:
        print("错误：输入必须是1D numpy数组")
        return result
    
    result['sqi'] = calculate_sqi(ppg_signal, fs, window_sec=len(ppg_signal)/fs)
    # 滤波参数：保留0.5-4Hz（对应心率30-240次/分，覆盖生理范围）
    ppg_filtered = ppg_signal
    # 3. 峰值检测（识别PP波峰，对应心跳）
    # 动态设置峰值高度阈值（信号均值的1.2倍，可根据实际信号调整）
    peak_height = np.mean(ppg_filtered) + 1 * np.std(ppg_filtered)
    # 最小峰间距：0.3秒（对应最大心率200次/分，避免过近的误检峰）
    min_peak_distance = int(0.3 * fs)
    
    # 检测峰值
    peaks, _ = find_peaks(
        x=ppg_filtered,
        height=peak_height,
        distance=min_peak_distance,
        prominence=0.1 * np.std(ppg_filtered)  # 突出度，过滤小波动
    )
    
    # 检查峰值数量（至少需要3个峰才能计算HRV）
    if len(peaks) < 3:
        return result
    
    # 4. 计算PP间期（相邻峰的时间差，单位：秒）
    peak_times = peaks / fs  # 峰值对应的时间（秒）
    pp_intervals = np.diff(peak_times)  # PP间期（秒）
    pp_intervals_ms = pp_intervals * 1000  # 转换为毫秒
    
    # 5. 计算心率（次/分）
    mean_pp_interval = np.mean(pp_intervals)
    result['heart_rate'] = round(len(ppg_signal) / fs / mean_pp_interval, 1)  # 保留1位小数
    if result['heart_rate'] < 30 / 60 * (len(ppg_signal) / fs) or result['heart_rate'] > 220 / 60 * (len(ppg_signal) / fs):
        return result  # 心率异常
    # 6. 计算HRV时间域指标
    # SDNN：所有PP间期的标准差（ms）
    result['sdnn'] = round(np.std(pp_intervals_ms), 1)
    
    # RMSSD：相邻PP间期差值的均方根（ms）
    diff_pp = np.diff(pp_intervals_ms)  # 相邻PP间期的差值
    result['rmssd'] = round(np.sqrt(np.mean(diff_pp **2)), 1)
    
    # NN50和pNN50：相邻差值>50ms的数量及占比
    nn50 = sum(abs(diff_pp) > 50)
    result['nn50'] = nn50
    result['pnn50'] = round((nn50 / len(diff_pp)) * 100, 1) if len(diff_pp) > 0 else None

    return result

def load_dataset(file_path):
    """
    加载数据集文件
    
    参数:
        file_path: 数据集文件路径
        
    返回:
        加载的数据集
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    return data

def pre_train_dataset(T = 10, frequency=32, overlap = 0.5, datasets=None):
    """
    加载所有数据集并打印基本信息
    """
    frequency = frequency  # 采样频率10Hz
    length = T * frequency  # 每个数据块的长度
    step = length * (1 - overlap)  # 步长
    clas_data = load_dataset(f"C:/Users/12992/Desktop/实验室/stress_pretrain/dataset/CLAS_{frequency}Hz.pkl")
    wesad_data = load_dataset(f"C:/Users/12992/Desktop/实验室/stress_pretrain/dataset/WESAD_{frequency}Hz.pkl")
    mtspd_data = load_dataset("C:/Users/12992/Desktop/实验室/stress_pretrain/dataset/MTSPD.pkl")
    new_data = []
    new_labels = []
    new_dlabels = []
    if 'CLAS' in datasets:
        cnt_sqi = 0
        cnt_ipa = 0
        for i in clas_data:
            # print(f"CLAS 参与者 {i} 的数据块数量: {len(clas_data[i]['data'])}")
            for j in clas_data[i]['data']:
                # print(f"数据块长度: {len(j)}")
                for k in range(0, len(j) - length - frequency + 1, int(step)):
                    segment = j[k:k + length]
                    segment = butter_bandpass_filter(
                        data=segment,
                        lowcut=0.5,
                        highcut=4.0,
                        fs=frequency,
                        order=2
                    )
                    # segment = preprocess_ppg(np.array(segment), fs=10)
                    label = calculate_hr_hrv(np.array(segment), fs=frequency)
                    dlabel = segment
                    if label['heart_rate'] is not None and label['heart_rate'] > 40 / 60 * T:
                        # print(f"CLAS 心率: {label['heart_rate']} 次/分")
                        new_data.append(segment)
                        new_labels.append(list(label.values()))
                        new_dlabels.append(dlabel)
                        if not np.isnan(label['sqi']):
                            cnt_sqi += 1
        print(f"CLAS 有效数据块数量: {len(new_data)}, SQI有效数量: {cnt_sqi}, IPA有效数量: {cnt_ipa}")
    print(f"总数据块数量: {len(new_data)}")
    print(f"总标签数量: {len(new_labels)}")
    if 'WESAD' in datasets:
        cnt_sqi = 0
        cnt_ipa = 0
        for i in wesad_data:
            # print(f"WESAD 参与者 {i} 的数据块数量: {len(wesad_data[i]['data'])}")
            for j in wesad_data[i]['data']:
                # print(f"数据块长度: {len(j)}")
                for k in range(0, len(j) - length - frequency + 1, int(step)):
                    segment = j[k:k + length]
                    segment = butter_bandpass_filter(
                        data=segment,
                        lowcut=0.5,
                        highcut=4.0,
                        fs=frequency,
                        order=2
                    )
                    # segment = preprocess_ppg(np.array(segment), fs=10)
                    label = calculate_hr_hrv(np.array(segment), fs=frequency)
                    dlabel = segment
                    if label['heart_rate'] is not None and label['heart_rate'] > 40 / 60 * T:
                        # print(f"WESAD 心率: {label['heart_rate']} 次/分")
                        new_data.append(segment)
                        new_labels.append(list(label.values()))
                        new_dlabels.append(dlabel)
                        if not np.isnan(label['sqi']):
                            cnt_sqi += 1
        print(f"WESAD 有效数据块数量: {len(new_data)}, SQI有效数量: {cnt_sqi}, IPA有效数量: {cnt_ipa}")
    print(f"总数据块数量: {len(new_data)}")
    print(f"总标签数量: {len(new_labels)}")
    cnt1 = 0
    cnt2 = 0
    if 'MTSPD' in datasets:
        for i in mtspd_data:
            # print(f"MTSPD 参与者 {i} 的数据块数量: {len(mtspd_data[i]['data'])}")
            for j in mtspd_data[i]['data']:
                # print(f"数据块长度: {len(j)}")
                for k in range(0, len(j) - length - frequency + 1, int(step)):
                    segment = j[k:k + length]
                    segment = butter_bandpass_filter(
                        data=segment,
                        lowcut=0.5,
                        highcut=4.0,
                        fs=frequency,
                        order=2
                    )
                    # segment = preprocess_ppg(np.array(segment), fs=10)
                    label = calculate_hr_hrv(np.array(segment), fs=frequency)
                    dlabel = segment
                    cnt1 += 1
                    if label['heart_rate'] is not None and label['heart_rate'] > 40 / 60 * T:
                        # print(f"MTSPD 心率: {label['heart_rate']} 次/分")
                        new_data.append(segment)
                        new_labels.append(list(label.values()))
                        new_dlabels.append(dlabel)
                        cnt2 += 1
    print(f"MTSPD 总数据块数量: {cnt1}, 有效数据块数量: {cnt2}")
    print(f"总数据块数量: {len(new_data)}")
    print(f"总标签数量: {len(new_labels)}")
    # for segment in new_data:
    #     label = calculate_hr_hrv(np.array(segment), fs=10)
    #     print(f"心率: {label['heart_rate']} 次/分, SDNN: {label['sdnn']} ms, RMSSD: {label['rmssd']} ms, NN50: {label['nn50']}, pNN50: {label['pnn50']} %")
    return new_data, new_labels, new_dlabels

def pretrain_create_dataloader(batch_size=32, T = 10, frequency = 32, overlap = 0.5, datasets=None, shuffle=True, num_workers=0, train_proportion=0.7, test_proportion=0.1, worker_init_fn=None):
    """
    创建DataLoader
    """
    data, labels, dlabels = pre_train_dataset(T=T, frequency=frequency, overlap=overlap, datasets=datasets)
    print(f"数据总量: {len(data)}, 标签总量: {len(labels)}")
    std_per_type = np.std(labels, axis=0, ddof=0)


    dataset = NewDataset(data, labels, dlabels)
    
    individual = calculate_individual(dlabels, frequency=frequency)
    dataset = NewDataset(data, individual, dlabels)

    train_size = int(len(data) * train_proportion)
    test_size = int(len(data) * test_proportion)
    val_size = len(data) - train_size - test_size

    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_init_fn)

    return train_loader, val_loader, test_loader, std_per_type

def finetune_per_dataset(T = 10, frequency=32, overlap = 0.5, datasets=None):
    frequency = frequency  # 采样频率10Hz
    length = T * frequency  # 每个数据块的长度
    step = length * (1 - overlap)  # 步长
    new_data = []
    if 'CLAS' in datasets:
        clas_data = load_dataset(f"C:/Users/12992/Desktop/实验室/stress_pretrain/dataset/CLAS_{frequency}Hz.pkl")
        cntl0 = 0
        clas = {
            'name': 'CLAS',
            'ID':[],
            'data': [],
            'label': [],
            'dlabel': [],
            'original_data': []
        }
        for i in clas_data:
            print(f"CLAS 参与者 {i} 的数据块数量: {len(clas_data[i]['data'])}")
            labelslength = [0, 0]
            for j in range(len(clas_data[i]['data'])):
                data = clas_data[i]['data'][j]
                labels = clas_data[i]['label'][j]
                labelslength[labels] += len(data)
            
            for j in range(len(clas_data[i]['data'])):
                data = clas_data[i]['data'][j]
                labels = clas_data[i]['label'][j]
                for k in range(0, len(data) - length - frequency + 1, int(step)):
                    segment = data[k:k + length]
                    segment = preprocess_ppg(
                        ppg_data=segment,
                        fs=frequency
                    )
                    label = calculate_hr_hrv(np.array(segment), fs=frequency)
                    dlabel = data[k + length: k + length + frequency]
                    if label['heart_rate'] is not None and label['heart_rate'] > 40 / 60 * T:
                        # print(f"CLAS 心率: {label['heart_rate']} 次/分")
                        clas['ID'].append(i)
                        clas['data'].append(segment)
                        clas['label'].append(labels)
                        clas['dlabel'].append(dlabel)
                        individual_data = clas_data[i]['data'][0][-min(3840, len(clas_data[i]['data'][0])):]
                        individual_data = preprocess_ppg(
                            ppg_data=individual_data,
                            fs=frequency
                        )
                        clas['original_data'].append(individual_data)
                        if labels == 0:
                            cntl0 += 1
        print(cntl0)
        print(len(clas['data']))
        new_data.append(clas)
    if 'WESAD' in datasets:
        wesad_data = load_dataset(f"C:/Users/12992/Desktop/实验室/stress_pretrain/dataset/WESAD_{frequency}Hz.pkl")
        cntl0 = 0
        wesad = {
            'name': 'WESAD',
            'ID':[],
            'data': [],
            'label': [],
            'dlabel': [],
            'original_data': []
        }
        for i in wesad_data:
            print(f"WESAD 参与者 {i} 的数据块数量: {len(wesad_data[i]['data'])}")
            for j in range(len(wesad_data[i]['data'])):
                data = wesad_data[i]['data'][j]
                labels = wesad_data[i]['label'][j]
                for k in range(0, len(data) - length - frequency + 1, int(step)):
                    segment = data[k:k + length]
                    segment = preprocess_ppg(
                        ppg_data=segment,
                        fs=frequency
                    )
                    label = calculate_hr_hrv(np.array(segment), fs=frequency)
                    dlabel = data[k + length: k + length + frequency]
                    if label['heart_rate'] is not None and label['heart_rate'] > 40 / 60 * T:
                        # print(f"WESAD 心率: {label['heart_rate']} 次/分")
                        wesad['ID'].append(i)
                        wesad['data'].append(segment)
                        wesad['label'].append(labels)
                        wesad['dlabel'].append(dlabel)
                        individual_data = wesad_data[i]['data'][0][-min(3840, len(wesad_data[i]['data'][0])):]
                        individual_data = preprocess_ppg(
                            ppg_data=individual_data,
                            fs=frequency
                        )
                        wesad['original_data'].append(individual_data)
                        if labels == 0:
                            cntl0 += 1
        print(cntl0)
        print(len(wesad['data']))
        new_data.append(wesad)
    cnt1 = 0
    cnt2 = 0
    if 'MTSPD' in datasets:
        mtspd_data = load_dataset(f"C:/Users/12992/Desktop/实验室/stress_pretrain/dataset/MTSPD_{frequency}Hz.pkl")
        mtspd = {
            'name': 'MTSPD',
            'data': [],
            'label': [],
            'dlabel': []
        }
        for i in mtspd_data:
            # print(f"MTSPD 参与者 {i} 的数据块数量: {len(mtspd_data[i]['data'])}")
            for j in range(len(mtspd_data[i]['data'])):
                data = mtspd_data[i]['data'][j]
                labels = mtspd_data[i]['label'][j]
                for k in range(0, len(data) - length - frequency + 1, int(step)):
                    segment = data[k:k + length]
                    segment = butter_bandpass_filter(
                        data=segment,
                        lowcut=0.5,
                        highcut=4.0,
                        fs=frequency,
                        order=2
                    )
                    label = calculate_hr_hrv(np.array(segment), fs=frequency)
                    dlabel = data[k + length: k + length + frequency]
                    cnt1 += 1
                    if label['heart_rate'] is not None and label['heart_rate'] > 40 / 60 * T:
                        # print(f"MTSPD 心率: {label['heart_rate']} 次/分")
                        mtspd['data'].append(segment)
                        mtspd['label'].append(labels)
                        mtspd['dlabel'].append(dlabel)
                        cnt2 += 1
        new_data.append(mtspd)
    print(f"MTSPD 总数据块数量: {cnt1}, 有效数据块数量: {cnt2}")
    # for segment in new_data:
    #     label = calculate_hr_hrv(np.array(segment), fs=10)
    #     print(f"心率: {label['heart_rate']} 次/分, SDNN: {label['sdnn']} ms, RMSSD: {label['rmssd']} ms, NN50: {label['nn50']}, pNN50: {label['pnn50']} %")
    return new_data

def build_dataset(df, ids):
    newdf = df[df['id'].isin(ids)]
    return NewDataset(newdf['data'].tolist(), newdf['labels'].tolist(), newdf['dlabels'].tolist(), finetune=True, individual=newdf['individual'].tolist())

def normalize(individuals):
    # 将每个特征维度的值缩放到0-1范围内
    all_features = np.array(individuals)
    min_vals = np.min(all_features, axis=0)
    max_vals = np.max(all_features, axis=0)
    for i in range(len(individuals)):
        for j in range(len(individuals[i])):
            if max_vals[j] - min_vals[j] > 0:
                individuals[i][j] = (individuals[i][j] - min_vals[j]) / (max_vals[j] - min_vals[j])
            else:
                individuals[i][j] = 0.0  # 如果所有值相同，直接设置为0
    return individuals

def calculate_individual(original_data, frequency=10):
    individuals = []
    for data in original_data:
        # individual, feature_names = individual_feature_pipeline(data, fs=frequency)
        individual = new_individual(data, fs=frequency)
        individuals.append(individual)
    individuals = normalize(individuals)
    return individuals

def finetune_create_dataset(batch_size=32, T=10, frequency=10, overlap=0.5, datasets=None, shuffle=True, num_workers=0, train_proportion=0.7, test_proportion=0.15, worker_init_fn=None):
    dataset_list = finetune_per_dataset(T=T, frequency=frequency, overlap=overlap, datasets=datasets)
    dataloader_list = []
    for dataset in dataset_list:
        ids = dataset['ID']
        unique_ids = list(dict.fromkeys(ids))
        data = dataset['data']
        labels = dataset['label']
        dlabels = dataset['dlabel']
        original_data = dataset['original_data']
        individual = calculate_individual(original_data, frequency=frequency)

        print(f"数据总量: {len(data)}, 标签总量: {len(labels)}")
        random.shuffle(unique_ids)
        train_size = int(len(data) * train_proportion)
        test_size = int(len(data) * test_proportion)
        val_size = len(data) - train_size - test_size
        print(train_size, test_size, val_size)

        train_idsize = int(len(unique_ids) * train_proportion)
        test_idsize = int(len(unique_ids) * test_proportion)
        val_idsize = len(unique_ids) - train_idsize - test_idsize
        print(train_idsize, test_idsize, val_idsize)

        train_ids = unique_ids[: train_idsize]
        test_ids = unique_ids[train_idsize: train_idsize + test_idsize]
        val_ids = unique_ids[train_idsize + test_idsize:]
        print(f"训练集ID: {train_ids}, 测试集ID: {test_ids}, 验证集ID: {val_ids}")
        df = pd.DataFrame(
            {
                'id': ids,
                'data': data,
                'labels': labels,
                'dlabels': dlabels,
                'individual': individual
            }
        )

        train_dataset = build_dataset(df, train_ids)
        test_dataset = build_dataset(df, test_ids)
        val_dataset = build_dataset(df, val_ids)

        # train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset_obj, [train_size, test_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_init_fn)

        dataloader_list.append((dataset['name'], train_loader, val_loader, test_loader))
    with open(f'./dataset/finetune_dataset.pkl', 'wb') as f:
        pickle.dump(dataloader_list, f)

def finetune_pd_create_dataloader(batch_size=32, T=10, frequency=10, overlap=0.5, datasets=None, shuffle=True, num_workers=0, train_proportion=0.7, test_proportion=0.15, worker_init_fn=None):
    with open(f'./dataset/finetune_dataset.pkl', "rb") as f:
        dataloader_list = pickle.load(f)
    return dataloader_list

if __name__ == "__main__":
    finetune_create_dataset(batch_size=64, T=10, frequency=32, overlap=0.5, datasets=['CLAS', 'WESAD'])
    # clas_data = load_dataset("C:/Users/12992/Desktop/实验室/stress_pretrain/dataset/CLAS.pkl")
    # wesad_data = load_dataset("C:/Users/12992/Desktop/实验室/stress_pretrain/dataset/WESAD.pkl")
    # mtspd_data = load_dataset("C:/Users/12992/Desktop/实验室/stress_pretrain/dataset/MTSPD.pkl")
    # pre_train_data = pre_train_dataset(T=10, overlap=0.5)