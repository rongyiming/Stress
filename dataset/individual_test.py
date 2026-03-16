import numpy as np
from scipy.signal import find_peaks, savgol_filter

def preprocess_ppg(ppg_data, fs=32):
    """
    PPG数据预处理：去除基线漂移（3阶多项式拟合）
    :param ppg_data: 输入的PPG原始数据（numpy数组），长度应为3840（2*60*32）
    :param fs: 采样率，固定为32Hz
    :return: 去基线后的PPG数据
    """
    # 生成时间轴
    t = np.arange(len(ppg_data)) / fs
    # 3阶多项式拟合基线
    baseline = np.polyfit(t, ppg_data, 3)
    baseline_curve = np.polyval(baseline, t)
    # 去除基线漂移
    ppg_clean = ppg_data - baseline_curve
    return ppg_clean

def detect_peaks(ppg_clean, fs=32):
    """
    检测PPG波峰（收缩峰），剔除异常峰，保证生理合理性
    :param ppg_clean: 去基线后的PPG数据
    :param fs: 采样率32Hz
    :return: 波峰位置索引数组、P-P间期序列（ms）、瞬时心率序列（bpm）
    """
    # 波峰检测：距离≥0.5s（16个采样点），排除过近伪峰；高度≥均值，排除低幅值噪声峰
    min_distance = int(0.5 * fs)  # 健康人脉搏周期≥0.5s
    peaks, _ = find_peaks(ppg_clean, distance=min_distance, height=np.mean(ppg_clean))
    # print(peaks, ppg_clean)
    
    # 计算P-P间期（ms）：相邻峰的时间差×1000
    pp_intervals = np.diff(peaks) / fs * 1000
    
    # 剔除异常P-P间期（3σ原则），避免伪差影响特征
    mean_pp = np.mean(pp_intervals)
    std_pp = np.std(pp_intervals) + 1  # 避免除以0
    # print(pp_intervals, mean_pp, std_pp)
    mask = (pp_intervals > mean_pp - 3 * std_pp) & (pp_intervals < mean_pp + 3 * std_pp)
    # print(mask)
    pp_intervals = pp_intervals[mask]
    peaks = peaks[:len(pp_intervals)+1]  # 同步裁剪波峰索引
    
    # 计算瞬时心率（bpm）
    hr_series = 60000 / pp_intervals
    
    numdown = len(ppg_clean) / fs / 60 * 40
    # print(f"ppg信号长度为{len(ppg_clean)}，检测到的有效脉搏数：{len(pp_intervals)}，建议至少{numdown}个以保证特征稳定性")

    # if len(pp_intervals) < numdown:
    #     print(ppg_clean)
    #     raise ValueError("有效脉搏数过少，可能是PPG数据质量问题或波峰检测失败")
    
    return peaks, pp_intervals, hr_series

def extract_single_wave_features(ppg_clean, peaks, fs=32):
    """
    提取单个脉搏波的时序+形态特征，最终返回所有波的平均值（个体特质）
    :param ppg_clean: 去基线后的PPG数据
    :param peaks: 波峰位置索引
    :param fs: 采样率32Hz
    :return: 单波特征平均值字典
    """
    # 采样步长（ms）：1/32*1000=31.25ms
    step_ms = 1000 / fs
    
    # 初始化特征列表
    t_up_list = []       # 上升时间（ms）
    dr_list = []         # 舒张期占比（%）
    t_d_list = []        # 重搏波峰延迟时间（ms）
    s_max_list = []      # 上升支最大斜率
    ddr_list = []        # 重搏波深度比（%）
    amp_list = []        # 收缩峰振幅（用于计算CVA）
    
    # 逐波处理
    for i in range(len(peaks)-1):
        # 分割单个脉搏波：从当前峰前10个点到下一个峰前10个点（避免边界截断）
        start_idx = max(0, peaks[i] - 10)
        end_idx = min(len(ppg_clean)-1, peaks[i+1] - 10)
        single_wave = ppg_clean[start_idx:end_idx]
        
        # 0-1归一化：消除振幅个体差异
        wave_min = np.min(single_wave)
        wave_max = np.max(single_wave)
        if wave_max - wave_min == 0:  # 避免除以0
            continue
        single_wave_norm = (single_wave - wave_min) / (wave_max - wave_min)
        
        # 1. 找到单波内的关键时间点
        # 收缩峰位置（归一化后最大值位置）
        p_idx = peaks[i] - start_idx  # 相对单波的位置
        # 起点（归一化后最小值位置，上升支起点）
        t0_idx = np.argmin(single_wave_norm[:p_idx])  # 只在收缩峰前找起点
        # 终点（单波最后一个点）
        t2_idx = len(single_wave_norm) - 1
        
        # 2. 计算时序特征
        # 上升时间（ms）：T0到P的时间差
        t_up = (p_idx - t0_idx) * step_ms
        t_up_list.append(t_up)
        
        # 舒张期占比（%）：(P到T2的时间)/(T0到T2的时间)×100
        t_total = (t2_idx - t0_idx) * step_ms
        t_diastole = (t2_idx - p_idx) * step_ms
        dr = (t_diastole / t_total) * 100 if t_total != 0 else 0
        dr_list.append(dr)
        
        # 3. 提取重搏波峰（D点）：下降支一阶导数的第二个零点
        # 对下降支求一阶导数
        descending_branch = single_wave_norm[p_idx:]
        if len(descending_branch) < 5:  # 下降支过短，跳过
            continue
        # 平滑后求导，减少噪声
        descending_smooth = savgol_filter(descending_branch, 5, 2)
        deriv = np.gradient(descending_smooth)
        # 找导数零点（符号变化的位置）
        zero_cross = np.where(np.diff(np.sign(deriv)))[0]
        # 重搏波峰是下降支的第二个零点（第一个是噪声，第二个是D点）
        if len(zero_cross) >= 2:
            d_idx_rel = zero_cross[1]  # 相对下降支的位置
            d_idx = p_idx + d_idx_rel  # 相对单波的位置
            # 重搏波峰延迟时间（ms）：P到D的时间差
            t_d = (d_idx - p_idx) * step_ms
            t_d_list.append(t_d)
            # 重搏波深度比（%）：(P振幅 - D振幅)/P振幅 ×100
            d_amp = single_wave_norm[d_idx]
            ddr = (1 - d_amp) * 100  # 归一化后P振幅=1
            ddr_list.append(ddr)
        
        # 4. 上升支最大斜率
        ascending_branch = single_wave_norm[t0_idx:p_idx+1]
        if len(ascending_branch) < 3:  # 上升支过短，跳过
            continue
        # 线性拟合上升支，求斜率
        x = np.arange(len(ascending_branch))
        slope, _ = np.polyfit(x, ascending_branch, 1)
        s_max_list.append(slope)
        
        # 记录收缩峰原始振幅（用于计算CVA）
        amp_list.append(single_wave[p_idx])
    
    # 计算所有特征的平均值（个体特质，消除单次波动）
    single_wave_features = {
        "mean_T_up": np.mean(t_up_list),          # 平均上升时间（ms）
        "mean_DR": np.mean(dr_list),              # 平均舒张期占比（%）
        "mean_T_d": np.mean(t_d_list) if t_d_list else 0,  # 平均重搏波延迟时间（ms）
        "mean_S_max": np.mean(s_max_list) if s_max_list else 0,  # 平均上升支最大斜率
        "mean_DDR": np.mean(ddr_list) if ddr_list else 0,  # 平均重搏波深度比（%）
        "CVA": (np.std(amp_list) / np.mean(amp_list)) * 100 if amp_list else 0  # 振幅变异系数（%）
    }
    return single_wave_features

def new_individual(ppg_data, fs=32):
    """
    主函数：计算所有PPG个人特质特征
    :param ppg_data: 输入的2分钟32Hz PPG数据（numpy数组，长度3840）
    :param fs: 采样率，固定32Hz
    :return: 所有特征的字典
    """
    # 步骤1：预处理（去基线）
    ppg_clean = preprocess_ppg(ppg_data, fs)
    # 步骤2：波峰检测，提取P-P间期和心率序列
    peaks, pp_intervals, hr_series = detect_peaks(ppg_clean, fs)
    
    # 步骤3：计算时域HRV特征
    mpp = np.mean(pp_intervals)                # 平均P-P间期（ms）
    mhr = np.mean(hr_series)                   # 平均心率（bpm）
    sdnn = np.std(pp_intervals, ddof=1)        # 心率标准差（ms）
    rmssd = np.sqrt(np.mean(np.square(np.diff(pp_intervals))))  # 相邻P-P差均方根（ms）
    
    # 步骤4：提取单波时序+形态特征
    single_wave_feat = extract_single_wave_features(ppg_clean, peaks, fs)
    
    # 整合所有特征
    all_features = {
        # 时域HRV特征
        "MPP (ms)": mpp,
        "MHR (bpm)": mhr,
        "SDNN (ms)": sdnn,
        "RMSSD (ms)": rmssd,
        # 脉搏波时序特征
        "mean_T_up (ms)": single_wave_feat["mean_T_up"],
        "mean_DR (%)": single_wave_feat["mean_DR"],
        "mean_T_d (ms)": single_wave_feat["mean_T_d"],
        # PPG形态特征
        "mean_S_max": single_wave_feat["mean_S_max"],
        "mean_DDR (%)": single_wave_feat["mean_DDR"],
        "CVA (%)": single_wave_feat["CVA"]
    }
    return list(all_features.values())