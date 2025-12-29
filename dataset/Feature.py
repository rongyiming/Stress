import numpy as np
from scipy import signal
from scipy.stats import skew


def preprocess_ppg(ppg_signal, fs=125, lowcut=0.5, highcut=4.0):
    """
    PPG信号预处理：带通滤波+去基线+归一化
    :param ppg_signal: 原始PPG信号（1D数组）
    :param fs: 采样率（Hz），默认125Hz
    :param lowcut: 带通滤波低频截止（Hz）
    :param highcut: 带通滤波高频截止（Hz）
    :return: 预处理后的信号
    """
    # 1. 4阶切比雪夫I型带通滤波（去除高频噪声和基线漂移）
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.cheby1(4, 0.1, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, ppg_signal)
    
    # 2. 去基线（减去滑动平均，窗口为1秒）
    window_size = int(fs * 1)  # 1秒窗口
    baseline = np.convolve(filtered, np.ones(window_size)/window_size, mode='same')
    detrended = filtered - baseline
    
    # 3. 归一化到[0,1]
    normalized = (detrended - np.min(detrended)) / (np.max(detrended) - np.min(detrended) + 1e-8)
    return normalized


def find_sys_and_notch(ppg, fs=125):
    """
    识别收缩峰（sys）和重搏切迹（notch）
    :param ppg: 预处理后的PPG信号
    :param fs: 采样率
    :return: sys（收缩峰索引）、notch（重搏切迹索引），未找到返回None
    """
    # 1. 找收缩峰（sys）：信号最大值点（允许±0.2秒内的峰值）
    peak_threshold = 0.5 * np.max(ppg)  # 峰值阈值（避免噪声）
    peaks, _ = signal.find_peaks(ppg, height=peak_threshold, distance=int(fs*0.2))
    if len(peaks) == 0:
        return None, None
    sys = peaks[0]  # 取第一个主要峰值（短片段通常含一个完整周期）
    
    # 2. 找重搏切迹（notch）：收缩峰后，一阶导数的极小值点（拐点）
    if sys >= len(ppg) - int(fs*0.5):  # 收缩峰过晚，无足够数据找切迹
        return sys, None
    # 截取收缩峰后0.1-0.5秒的信号（切迹通常在收缩峰后约0.1-0.3秒）
    post_sys = ppg[sys + int(fs*0.1) : sys + int(fs*0.5)]
    if len(post_sys) == 0:
        return sys, None
    # 一阶导数（找下降段的拐点）
    der = np.gradient(post_sys)
    # 找导数的极小值点（切迹处导数变化率最小）
    notches_in_post, _ = signal.find_peaks(-der)  # 极小值即-der的极大值
    if len(notches_in_post) == 0:
        return sys, None
    # 映射回原始信号索引
    notch = sys + int(fs*0.1) + notches_in_post[0]
    return sys, notch


def calculate_ipa(ppg, fs=125):
    """
    计算拐点面积比IPA
    :param ppg: 预处理后的PPG信号
    :param fs: 采样率
    :return: IPA值（未找到节点返回NaN）
    """
    sys, notch = find_sys_and_notch(ppg, fs)
    if sys is None or notch is None or notch >= len(ppg):
        return np.nan
    # 收缩期面积（0→sys）
    area_systolic = np.sum(ppg[:sys+1])  # 包含sys点
    # 舒张期面积（notch→终点）
    area_diastolic = np.sum(ppg[notch:])
    if area_diastolic < 1e-8:  # 避免除零
        return np.nan
    return area_systolic / area_diastolic


def calculate_sqi(ppg, fs=125, window_sec=5):
    """
    计算信号质量指数SQI
    :param ppg: 预处理后的PPG信号
    :param fs: 采样率
    :param window_sec: 滑动窗口时长（秒），默认5秒
    :return: SQI值（窗口不足时返回NaN）
    """
    window_size = int(fs * window_sec)
    n_windows = len(ppg) // window_size
    if n_windows < 1:
        return np.nan  # 信号过短，至少需要1个窗口
    # 分窗口计算偏度
    skewness_list = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window = ppg[start:end]
        # 计算偏度（直接用scipy的skew函数，与公式等价）
        s = skew(window)
        skewness_list.append(s)
    # 最后一个不完整窗口（可选保留）
    if len(ppg) % window_size != 0:
        window = ppg[n_windows*window_size:]
        s = skew(window)
        skewness_list.append(s)
    return np.mean(skewness_list)



# 示例使用
if __name__ == "__main__":
    # 生成示例PPG信号（实际应用中替换为真实信号）
    fs = 125  # 采样率125Hz
    t = np.linspace(0, 10, fs*10)  # 10秒信号
    # 模拟PPG波形（收缩峰+重搏切迹）
    ppg_raw = 0.5 * np.sin(2*np.pi*1*t) + 0.3 * np.exp(-(t-0.3)**2/0.01) + 0.1*np.random.randn(len(t))
    
    # 预处理
    ppg_processed = preprocess_ppg(ppg_raw, fs=fs)
    
    # 计算指标
    ipa = calculate_ipa(ppg_processed, fs=fs)
    sqi = calculate_sqi(ppg_processed, fs=fs)
    
    print(f"IPA: {ipa:.4f}")
    print(f"SQI: {sqi:.4f}")