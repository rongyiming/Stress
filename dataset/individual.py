import numpy as np
import scipy.signal as signal
import pywt
from scipy.stats import variation, f_oneway
from sklearn.preprocessing import StandardScaler

# ===================== 1. 预处理模块（适配10Hz） =====================
def load_ppg_signal(signal_path=None, fs=10, duration=120):
    """
    加载/生成PPG信号（适配10Hz采样率、120秒时长）
    :param signal_path: 真实PPG数据路径（txt/csv），None则生成模拟信号
    :param fs: 采样率（Hz），默认10Hz
    :param duration: 信号时长（秒），默认120秒
    :return: ppg_signal (np.array), fs
    """
    if signal_path:
        # 读取真实PPG数据（示例：单列数据）
        ppg_signal = np.loadtxt(signal_path)
        # 确保数据长度匹配10Hz*120s=1200点（若真实数据长度不符，做裁剪/补零）
        if len(ppg_signal) > fs*duration:
            ppg_signal = ppg_signal[:fs*duration]
        elif len(ppg_signal) < fs*duration:
            ppg_signal = np.pad(ppg_signal, (0, fs*duration - len(ppg_signal)), 'constant')
    else:
        # 生成模拟静息PPG信号（10Hz、120秒，含基础心率+呼吸调制+轻微噪声）
        t = np.linspace(0, duration, fs*duration)
        hr = 70  # 静息心率70次/分
        f_hr = hr/60  # 心率频率
        f_resp = 0.25  # 呼吸频率0.25Hz（15次/分）
        
        # 基础PPG波形（正弦波模拟主波+重搏波）
        ppg_base = np.sin(2*np.pi*f_hr*t) + 0.2*np.sin(2*np.pi*2*f_hr*t)
        # 呼吸调制
        ppg_resp = ppg_base * (1 + 0.1*np.sin(2*np.pi*f_resp*t))
        # 加轻微噪声
        ppg_signal = ppg_resp + 0.05*np.random.randn(len(ppg_resp))
    return ppg_signal, fs

def preprocess_ppg(ppg_signal, fs):
    """
    PPG信号预处理：去基线漂移 + 低通滤波 + 伪影剔除（适配10Hz）
    :param ppg_signal: 原始PPG信号
    :param fs: 采样率（10Hz）
    :return: clean_ppg (去噪后信号), valid_segments (有效分段)
    """
    # 1. 去基线漂移（db4小波，适配10Hz的层数调整为4层，避免过分解）
    wavelet = 'db4'
    level = 4  # 10Hz、1200点数据，4层分解更合适（2^4=16点，远小于1200）
    coeffs = pywt.wavedec(ppg_signal, wavelet, level=level)
    # 置零低频系数（基线漂移主要在最高层）
    coeffs[-1] = np.zeros_like(coeffs[-1])
    coeffs[-2] = np.zeros_like(coeffs[-2])
    ppg_detrend = pywt.waverec(coeffs, wavelet)
    ppg_detrend = ppg_detrend[:len(ppg_signal)]  # 对齐长度
    
    # 2. 低通滤波（巴特沃斯，适配10Hz的截止频率）
    nyq = 0.5 * fs
    cutoff = 8 / nyq  # 8Hz截止频率（10Hz下nyq=5，cutoff=1.6）
    b, a = signal.butter(4, cutoff, btype='low')
    ppg_filtered = signal.filtfilt(b, a, ppg_detrend)
    
    # 3. 伪影剔除（按10秒分段，适配10Hz的平滑窗口）
    segment_len = 10 * fs  # 10秒/段 → 10*10=100点/段
    n_segments = len(ppg_filtered) // segment_len
    valid_segments = []
    for i in range(n_segments):
        seg = ppg_filtered[i*segment_len : (i+1)*segment_len]
        # 计算SNR（信号能量/噪声能量，噪声=信号-平滑信号）
        seg_smooth = signal.savgol_filter(seg, 15, 4)  
        noise = seg - seg_smooth
        snr = 10 * np.log10(np.sum(seg**2) / (np.sum(noise**2) + 1e-6))
        if snr > 10:
            valid_segments.append(seg)
    
    return ppg_filtered, valid_segments

# ===================== 2. 特征提取模块（适配10Hz） =====================
def detect_peaks_valleys(ppg_segment, fs):
    """
    检测PPG分段的波峰、波谷、重搏波（适配10Hz）
    :param ppg_segment: 单段PPG信号（10Hz下100点/段）
    :param fs: 采样率（10Hz）
    :return: peaks (波峰索引), valleys (波谷索引), dw_peaks (重搏波索引)
    """
    # 检测波峰（最小高度0.2，最小间距适配10Hz）
    # 最小间距：按最高心率100次/分 → 周期600ms → 10Hz下=6点
    min_distance = int(fs / (100/60))  
    mean = np.mean(ppg_segment)
    std = np.std(ppg_segment)
    threshold = mean + 0.5 * std
    peaks, _ = signal.find_peaks(ppg_segment, height=threshold, distance=min_distance)
    
    # 检测波谷（反转信号找波峰）
    threshold = mean - 0.5 * std
    valleys, _ = signal.find_peaks(-ppg_segment, height=-threshold, distance=min_distance)
    
    # 检测重搏波（下降支拐点，适配10Hz分段长度）
    dw_peaks = []
    for i in range(len(peaks)-1):
        # 取两个波峰之间的下降支
        start = peaks[i]
        valley_candidates = valleys[valleys > start]
        end = valley_candidates[0] if len(valley_candidates)>0 else peaks[i+1]
        desc_segment = ppg_segment[start:end]
        # 找下降支的极小值（重搏波谷），最小高度适配10Hz
        dw_valley, _ = signal.find_peaks(-desc_segment, height=0.1)
        if len(dw_valley) > 0:
            dw_peaks.append(start + dw_valley[0])
    
    return peaks, valleys, np.array(dw_peaks)

def extract_time_domain_features(ppg_segment, fs, peaks, valleys, dw_peaks):
    """提取时域特征（逻辑不变，适配10Hz的时间转换）"""
    features = {}
    # 1. 脉搏波传导时间（PTT）：模拟ECG R波（用PPG波峰前10ms替代）
    ptt = 0.010  # 10ms（示例值，实际需同步ECG）
    features['PTT'] = ptt
    
    # 2. 上升时间（RT）、下降时间（DT）
    if len(peaks) > 0 and len(valleys) > 0:
        # 取第一个完整周期
        peak = peaks[0]
        valley_prev = valleys[valleys < peak][-1] if len(valleys[valleys < peak])>0 else 0
        valley_next = valleys[valleys > peak][0] if len(valleys[valleys > peak])>0 else len(ppg_segment)-1
        # valley_next = valleys[valleys > peak][0] if len(valleys > peak)>0 else len(ppg_segment)-1
        rt = (peak - valley_prev) / fs  # 10Hz下，采样点转秒更精准
        dt = (valley_next - peak) / fs
        features['RT'] = rt
        features['DT'] = dt
    else:
        features['RT'] = 0
        features['DT'] = 0
    
    # 3. 重搏波深度（Dw）
    if len(peaks) > 0 and len(dw_peaks) > 0:
        peak_val = ppg_segment[peaks[0]]
        dw_val = ppg_segment[dw_peaks[0]]
        dw = (peak_val - dw_val) / (peak_val + 1e-6)  # 归一化深度
        features['Dw'] = dw
    else:
        features['Dw'] = 0
    
    # 4. HRV时域指标（SDNN、RMSSD、pNN50）
    if len(peaks) >= 2:
        nn_intervals = np.diff(peaks) / fs * 1000  # 采样点转毫秒（10Hz下1点=100ms）
        sdnn = np.std(nn_intervals)
        rmssd = np.sqrt(np.mean(np.square(np.diff(nn_intervals))))
        pnn50 = 100 * np.sum(np.abs(np.diff(nn_intervals)) > 50) / len(nn_intervals)
        features['SDNN'] = sdnn
        features['RMSSD'] = rmssd
        features['pNN50'] = pnn50
    else:
        features['SDNN'] = 0
        features['RMSSD'] = 0
        features['pNN50'] = 0
    
    # 5. 波形面积比（Ra）
    if len(peaks) > 0 and len(valleys) > 0:
        peak = peaks[0]
        valley_prev = valleys[valleys < peak][-1] if len(valleys[valleys < peak])>0 else 0
        # 上升支面积
        up_area = np.trapz(ppg_segment[valley_prev:peak])
        # 整个周期面积
        valley_next = valleys[valleys > peak][0] if len(valleys[valleys > peak])>0 else len(ppg_segment)-1
        total_area = np.trapz(ppg_segment[valley_prev:valley_next])
        features['Ra'] = up_area / (total_area + 1e-6)
    else:
        features['Ra'] = 0
    
    return features

def extract_freq_domain_features(ppg_segment, fs):
    """提取频域特征（适配10Hz的功率谱计算）"""
    features = {}
    # 1. 功率谱密度（Welch法，适配10Hz的分段长度）
    # 10Hz下每段100点，nperseg调整为64（小于100，且为2的幂）
    f, psd = signal.welch(ppg_segment, fs, nperseg=64, noverlap=32)
    
    # 2. 心率频率（fHR）和呼吸频率（fR）
    hr_band = (0.5, 3)  # 心率频率范围0.5-3Hz（30-180次/分）
    resp_band = (0.15, 0.4)  # 呼吸频率范围0.15-0.4Hz
    
    # 心率峰值
    hr_idx = np.where((f >= hr_band[0]) & (f <= hr_band[1]))[0]
    if len(hr_idx) > 0:
        f_hr = f[hr_idx][np.argmax(psd[hr_idx])]
        features['fHR'] = f_hr
    else:
        features['fHR'] = 0
    
    # 呼吸峰值
    resp_idx = np.where((f >= resp_band[0]) & (f <= resp_band[1]))[0]
    if len(resp_idx) > 0:
        f_r = f[resp_idx][np.argmax(psd[resp_idx])]
        features['fR'] = f_r
    else:
        features['fR'] = 0
    
    # 3. LF/HF功率比
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)
    lf_idx = np.where((f >= lf_band[0]) & (f <= lf_band[1]))[0]
    hf_idx = np.where((f >= hf_band[0]) & (f <= hf_band[1]))[0]
    lf_power = np.sum(psd[lf_idx]) if len(lf_idx) > 0 else 1e-6
    hf_power = np.sum(psd[hf_idx]) if len(hf_idx) > 0 else 1e-6
    features['LF/HF'] = lf_power / hf_power
    
    # 4. 谱峰宽度（PW）
    if features['fHR'] > 0:
        hr_peak_pos = np.where(f == features['fHR'])[0]
        if len(hr_peak_pos) > 0:
            hr_peak_val = psd[hr_peak_pos[0]]
            half_max = hr_peak_val / 2
            # 找半高宽
            peak_idx = np.argmax(psd[hr_idx])
            left_idx = np.where(psd[hr_idx][:peak_idx] <= half_max)[0]
            left = hr_idx[left_idx[-1]] if len(left_idx) > 0 else hr_idx[0]
            right_idx = np.where(psd[hr_idx][peak_idx:] >= half_max)[0]
            right = hr_idx[right_idx[-1]] if len(right_idx) > 0 else hr_idx[-1]
            features['PW'] = f[right] - f[left]
        else:
            features['PW'] = 0
    else:
        features['PW'] = 0
    
    return features

def extract_morphology_features(ppg_segment, fs, peaks, valleys, dw_peaks):
    """提取形态学特征（适配10Hz的时间/斜率计算）"""
    features = {}
    # 1. 波峰斜率（k_up）
    if len(peaks) > 0 and len(valleys) > 0:
        peak = peaks[-1]
        valley_prev = valleys[valleys < peak][-1] if len(valleys[valleys < peak])>0 else 0
        up_segment = ppg_segment[valley_prev:peak]
        # 10Hz下梯度转换为每秒斜率（*fs）
        k_up = np.max(np.gradient(up_segment)) * fs
        features['k_up'] = k_up
    else:
        features['k_up'] = 0
    
    # 2. 重搏波位置（Pw）
    if len(peaks) > 0 and len(dw_peaks) > 0:
        dt = features.get('DT', 0)
        pw_time = (dw_peaks[0] - peaks[0]) / fs
        features['Pw'] = pw_time / (dt + 1e-6)
    else:
        features['Pw'] = 0
    
    # 3. 波形对称性（S）
    if len(peaks) > 0:
        peak = peaks[0]
        valley_prev = valleys[valleys < peak][-1] if len(valleys[valleys < peak])>0 else 0
        left_area = np.trapz(ppg_segment[valley_prev:peak])
        valley_next = valleys[valleys > peak][0] if len(valleys[valleys > peak])>0 else len(ppg_segment)-1
        right_area = np.trapz(ppg_segment[peak:valley_next])
        features['S'] = left_area / (right_area + 1e-6)
    else:
        features['S'] = 0
    
    # 4. 特征点个数
    features['n_peaks'] = len(peaks)
    features['n_valleys'] = len(valleys)
    features['n_dw'] = len(dw_peaks)
    
    return features

def extract_time_freq_features(ppg_segment):
    """提取时频域特征（小波能量熵，适配10Hz分段长度）"""
    features = {}
    # 1. 小波能量熵（db4，4层分解，适配10Hz的100点分段）
    wavelet = 'db4'
    level = 4  # 100点分段，4层分解（2^4=16点）更合适
    coeffs = pywt.wavedec(ppg_segment, wavelet, level=level)
    # 计算各层能量
    energies = [np.sum(np.square(coeff)) for coeff in coeffs]
    total_energy = np.sum(energies) + 1e-6
    # 能量熵
    p = energies / total_energy
    entropy = -np.sum(p * np.log2(p + 1e-6))
    features['wavelet_entropy'] = entropy
    
    return features

def extract_all_features(valid_segments, fs):
    """整合所有特征提取，返回特征矩阵（适配10Hz）"""
    all_features = []
    feature_names = []
    for seg in valid_segments:
        # 检测特征点
        peaks, valleys, dw_peaks = detect_peaks_valleys(seg, fs)
        # 提取各维度特征
        td_feat = extract_time_domain_features(seg, fs, peaks, valleys, dw_peaks)
        fd_feat = extract_freq_domain_features(seg, fs)
        morph_feat = extract_morphology_features(seg, fs, peaks, valleys, dw_peaks)
        tf_feat = extract_time_freq_features(seg)
        
        # 合并特征
        combined = {**td_feat, **fd_feat, **morph_feat, **tf_feat}
        if len(feature_names) == 0:
            feature_names = list(combined.keys())
        all_features.append([combined[k] for k in feature_names])
    
    return np.array(all_features), feature_names

# ===================== 3. 个性化特征筛选（逻辑不变） =====================
def filter_personalized_features(features_matrix, feature_names, subject_features_list=None):
    """
    个性化特征筛选：
    1. 个体内稳定性：保留变异系数CV<20%的特征
    2. 个体间差异性：保留ANOVA p<0.05的特征（需多个体数据）
    :param features_matrix: 单个个体的特征矩阵（n_segments × n_features）
    :param feature_names: 特征名称列表
    :param subject_features_list: 其他个体的特征矩阵列表（用于ANOVA）
    :return: selected_features (筛选后特征向量), selected_names (筛选后特征名)
    """
    # 1. 个体内稳定性筛选（CV<20%）
    cv_values = variation(features_matrix, axis=0)  # 计算每个特征的变异系数
    stable_mask = cv_values < 0.2  # CV<20%为稳定特征
    stable_features = features_matrix[:, stable_mask]
    stable_names = [feature_names[i] for i in range(len(feature_names)) if stable_mask[i]]
    
    # 2. 个体间差异性筛选（ANOVA p<0.05）
    if subject_features_list is not None and len(subject_features_list) > 0:
        anova_p_values = []
        for i in range(stable_features.shape[1]):
            # 提取该特征在所有个体中的数据
            feat_list = [stable_features[:, i]] + [sub_feat[:, stable_mask][:, i] for sub_feat in subject_features_list]
            # 执行ANOVA
            f_stat, p_val = f_oneway(*feat_list)
            anova_p_values.append(p_val)
        # 保留p<0.05的特征
        diff_mask = np.array(anova_p_values) < 0.05
        selected_features = stable_features[:, diff_mask]
        selected_names = [stable_names[i] for i in range(len(stable_names)) if diff_mask[i]]
    else:
        # 无其他个体数据时，仅保留稳定特征
        selected_features = stable_features
        selected_names = stable_names
    
    # 3. 特征归一化（Z-score）
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(selected_features)
    
    # 4. 构建个性化特征向量（取所有分段的均值）
    personalized_vector = np.mean(normalized_features, axis=0)
    
    return personalized_vector, selected_names

def individual_feature_pipeline(ppg_signal, fs, subject_features_list=None):
    """
    个体化特征提取完整流程（适配10Hz、120s）
    :param ppg_signal: 原始PPG信号
    :param fs: 采样率（10Hz）
    :param subject_features_list: 其他个体的特征矩阵列表（用于ANOVA）
    :return: personalized_vector (个性化特征向量), selected_names (特征名称)
    """
    # 1. 预处理
    _, valid_segments = preprocess_ppg(ppg_signal, fs)

    # 2. 提取所有特征
    features_matrix, feature_names = extract_all_features(valid_segments, fs)
    
    feature_list = np.mean(features_matrix, axis=0)

    return feature_list, feature_names
    # 3. 个性化筛选
    # personalized_vector, selected_names = filter_personalized_features(
    #     features_matrix, feature_names, subject_features_list
    # )
    
    # return personalized_vector, selected_names

# ===================== 4. 主函数（测试10Hz、120s数据） =====================
if __name__ == "__main__":
    # 步骤1：生成/加载PPG信号（10Hz、120秒，模拟3个个体）
    fs = 10  # 固定为10Hz
    duration = 120  # 固定为120秒
    n_subjects = 3  # 模拟3个个体
    subject_features_list = []
    
    # 生成多个个体的PPG数据（用于ANOVA筛选）
    for sub_id in range(n_subjects):
        # 生成不同个体的模拟PPG（10Hz、120秒，心率/呼吸频率略有差异）
        ppg_signal, fs = load_ppg_signal(fs=fs, duration=duration)
        
        # 步骤2：预处理
        clean_ppg, valid_segments = preprocess_ppg(ppg_signal, fs)
        
        # 步骤3：提取所有特征
        features_matrix, feature_names = extract_all_features(valid_segments, fs)
        subject_features_list.append(features_matrix)
    
    # 步骤4：筛选第一个个体的个性化特征向量
    target_subject = 0
    target_features = subject_features_list[target_subject]
    # 其他个体的特征列表（用于ANOVA）
    other_subjects = subject_features_list[1:]
    
    personalized_vector, selected_names = filter_personalized_features(
        target_features, feature_names, other_subjects
    )
    
    # 步骤5：输出结果
    print("="*50)
    print(f"个体{target_subject+1}的个性化特征向量（{len(personalized_vector)}维）：")
    print("="*50)
    for name, val in zip(selected_names, personalized_vector):
        print(f"{name}: {val:.4f}")