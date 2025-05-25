from pathlib import Path
import numpy as np
import pandas as pd
import math
import csv
import pickle
import joblib # XGBoostTrainingPipeline 使用 joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
# from sklearn.feature_selection import SelectFromModel, mutual_info_classif # mutual_info_classif 可能不需要了
import xgboost as xgb
from scipy import stats
from scipy.signal import find_peaks
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 特徵計算函數 (從 baseline_moreFeatures.py 複製過來)
# ==============================================================================
def FFT(xreal, ximag):    
  n = 2
  while(n*2 <= len(xreal)):
      n *= 2
  
  p = int(math.log(n, 2))
  
  for i in range(0, n):
      a = i
      b = 0
      for j in range(0, p):
          b = int(b*2 + a%2)
          a = a/2
      if(b > i):
          xreal[i], xreal[b] = xreal[b], xreal[i]
          ximag[i], ximag[b] = ximag[b], ximag[i]
          
  wreal = []
  wimag = []
      
  arg = float(-2 * math.pi / n)
  treal = float(math.cos(arg))
  timag = float(math.sin(arg))
  
  wreal.append(float(1.0))
  wimag.append(float(0.0))
  
  for j in range(1, int(n/2)):
      wreal.append(wreal[-1] * treal - wimag[-1] * timag)
      wimag.append(wreal[-1] * timag + wimag[-1] * treal)
      
  m = 2
  while(m < n + 1):
      for k in range(0, n, m):
          for j in range(0, int(m/2), 1):
              index1 = k + j
              index2 = int(index1 + m / 2)
              t = int(n * j / m)
              treal = wreal[t] * xreal[index2] - wimag[t] * ximag[index2]
              timag = wreal[t] * ximag[index2] + wimag[t] * xreal[index2]
              ureal = xreal[index1]
              uimag = ximag[index1]
              xreal[index1] = ureal + treal
              ximag[index1] = uimag + timag
              xreal[index2] = ureal - treal
              ximag[index2] = uimag - timag
      m *= 2
      
  return n, xreal, ximag   
  
def FFT_data(input_data, swinging_times):   
  txtlength = swinging_times[-1] - swinging_times[0]
  a_mean = [0] * txtlength
  g_mean = [0] * txtlength
     
  for num in range(len(swinging_times)-1):
      a = []
      g = []
      for swing in range(swinging_times[num], swinging_times[num+1]):
          a.append(math.sqrt(math.pow((input_data[swing][0] + input_data[swing][1] + input_data[swing][2]), 2)))
          g.append(math.sqrt(math.pow((input_data[swing][3] + input_data[swing][4] + input_data[swing][5]), 2)))

      a_mean[num] = (sum(a) / len(a))
      g_mean[num] = (sum(a) / len(a))
  
  return a_mean, g_mean

def calculate_zero_crossing_rate(signal):
  """計算過零率"""
  zero_crossings = 0
  for i in range(1, len(signal)):
      if (signal[i-1] >= 0 and signal[i] < 0) or (signal[i-1] < 0 and signal[i] >= 0):
          zero_crossings += 1
  return zero_crossings / len(signal)

def calculate_activity_counts(signal, threshold=0.1):
  """計算活動計數（超過閾值的數據點數量）"""
  return sum(1 for x in signal if abs(x) > threshold) / len(signal)

def calculate_peak_to_peak(signal):
  """計算峰峰值"""
  return max(signal) - min(signal)

def calculate_quartiles(signal):
  """計算四分位數"""
  q1 = np.percentile(signal, 25)
  q2 = np.percentile(signal, 50)  # 中位數
  q3 = np.percentile(signal, 75)
  return q1, q2, q3

def calculate_percentiles(signal):
  """計算特定百分位數"""
  p10 = np.percentile(signal, 10)
  p90 = np.percentile(signal, 90)
  return p10, p90

def calculate_spectral_features(fft_real, fft_imag, sampling_rate=100):
  """計算頻域特徵"""
  # 計算功率譜
  power_spectrum = np.array([fft_real[i]**2 + fft_imag[i]**2 for i in range(len(fft_real))])
  frequencies = np.fft.fftfreq(len(power_spectrum), 1/sampling_rate)
  
  # 只取正頻率部分
  positive_freq_idx = frequencies >= 0
  power_spectrum = power_spectrum[positive_freq_idx]
  frequencies = frequencies[positive_freq_idx]
  
  if len(power_spectrum) == 0 or np.sum(power_spectrum) == 0:
      return 0, 0, 0, 0
  
  # 主頻率（最大功率對應的頻率）
  dominant_freq = frequencies[np.argmax(power_spectrum)]
  
  # 頻譜重心
  spectral_centroid = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
  
  # 頻譜滾降（85%能量所在的頻率點）
  cumulative_power = np.cumsum(power_spectrum)
  total_power = cumulative_power[-1]
  rolloff_idx = np.where(cumulative_power >= 0.85 * total_power)[0]
  spectral_rolloff = frequencies[rolloff_idx[0]] if len(rolloff_idx) > 0 else frequencies[-1]
  
  # 頻譜熵
  normalized_power = power_spectrum / np.sum(power_spectrum)
  spectral_entropy = -np.sum(normalized_power * np.log2(normalized_power + 1e-10))
  
  return dominant_freq, spectral_centroid, spectral_rolloff, spectral_entropy

def calculate_signal_magnitude_area(ax_data, ay_data, az_data):
  """計算信號幅度面積 (SMA)"""
  sma = sum(abs(ax_data[i]) + abs(ay_data[i]) + abs(az_data[i]) for i in range(len(ax_data)))
  return sma / len(ax_data)

def calculate_inter_axis_correlation(data1, data2, record_id=None, file_name=None):
    """
    計算軸間相關係數，出錯時直接顯示是哪筆資料
    
    Parameters:
    - data1, data2: 要計算相關係數的兩個數據序列
    - record_id: 當前處理的資料編號（可選）
    """
    
    try:
        # 轉換為 numpy 數組
        arr1 = np.asarray(data1, dtype=float)
        arr2 = np.asarray(data2, dtype=float)
        
        # 基本檢查
        if len(arr1) < 2 or len(arr2) < 2:
            if record_id is not None:
                print(f"❌ 第 {record_id} 筆資料錯誤：數據點不足 (arr1:{len(arr1)}, arr2:{len(arr2)})")
            return 0
        
        if len(arr1) != len(arr2):
            if record_id is not None:
                print(f"❌ 第 {record_id} 筆資料錯誤：長度不匹配 (arr1:{len(arr1)}, arr2:{len(arr2)})")
            return 0
        
        # 檢查 NaN
        if np.any(np.isnan(arr1)) or np.any(np.isnan(arr2)):
            if record_id is not None:
                print(f"❌ 第 {record_id} 筆資料錯誤：包含 NaN 值")
            return 0
        
        # 檢查無窮大
        if np.any(np.isinf(arr1)) or np.any(np.isinf(arr2)):
            if record_id is not None:
                print(f"❌ 第 {record_id} 筆資料錯誤：包含無窮大值")
            return 0
        
        # 檢查標準差
        std1, std2 = np.std(arr1), np.std(arr2)
        if std1 == 0 or std2 == 0:
            if record_id is not None and file_name is not None:
                print(f"❌ 文件 {file_name} 中的第 {record_id} 筆資料錯誤：標準差為零 (std1:{std1:.6f}, std2:{std2:.6f})")
            return 0
        
        # 計算相關係數
        correlation = np.corrcoef(arr1, arr2)[0, 1]
        
        # 檢查結果
        if np.isnan(correlation):
            if record_id is not None:
                print(f"❌ 第 {record_id} 筆資料錯誤：計算結果為 NaN")
            return 0
        
        return correlation
        
    except Exception as e:
        if record_id is not None:
            print(f"❌ 第 {record_id} 筆資料錯誤：{type(e).__name__} - {str(e)}")
        return 0

def calculate_trajectory_length(ax_data, ay_data, az_data):
  """計算軌跡長度"""
  trajectory_length = 0
  for i in range(1, len(ax_data)):
      dx = ax_data[i] - ax_data[i-1]
      dy = ay_data[i] - ay_data[i-1]
      dz = az_data[i] - az_data[i-1]
      trajectory_length += math.sqrt(dx**2 + dy**2 + dz**2)
  return trajectory_length


def feature(input_data, swinging_now, swinging_times, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer, file_name):
    # 注意：swinging_times 在 baseline_moreFeatures.py 中是 len(swing_index) - 1
    allsum = []
    mean = []
    var = []
    rms = []
    std = []
    XYZmean_a = 0
    a = []
    g = []
    a_s1 = 0
    a_s2 = 0
    g_s1 = 0
    g_s2 = 0
    a_k1 = 0
    a_k2 = 0
    g_k1 = 0
    g_k2 = 0
    
    # 分離各軸數據
    ax_data = [row[0] for row in input_data]
    ay_data = [row[1] for row in input_data]
    az_data = [row[2] for row in input_data]
    gx_data = [row[3] for row in input_data]
    gy_data = [row[4] for row in input_data]
    gz_data = [row[5] for row in input_data]
    
    for i in range(len(input_data)):
        if i==0:
            allsum = input_data[i]
            a.append(math.sqrt(math.pow((input_data[i][0] + input_data[i][1] + input_data[i][2]), 2)))
            g.append(math.sqrt(math.pow((input_data[i][3] + input_data[i][4] + input_data[i][5]), 2)))
            continue
        
        a.append(math.sqrt(math.pow((input_data[i][0] + input_data[i][1] + input_data[i][2]), 2)))
        g.append(math.sqrt(math.pow((input_data[i][3] + input_data[i][4] + input_data[i][5]), 2)))
        
        allsum = [allsum[feature_index] + input_data[i][feature_index] for feature_index in range(len(input_data[i]))]
        
    mean = [allsum[feature_index] / len(input_data) for feature_index in range(len(input_data[i]))]
    
    # 計算標準差
    std = [np.std([input_data[i][feature_index] for i in range(len(input_data))]) for feature_index in range(len(input_data[0]))]
    
    for i in range(len(input_data)):
        if i==0:
            var = input_data[i]
            rms = input_data[i]
            continue

        var = [var[feature_index] + math.pow((input_data[i][feature_index] - mean[feature_index]), 2) for feature_index in range(len(input_data[i]))]
        rms = [rms[feature_index] + math.pow(input_data[i][feature_index], 2) for feature_index in range(len(input_data[i]))]
        
    var = [math.sqrt((var[feature_index] / len(input_data))) for feature_index in range(len(input_data[i]))]
    rms = [math.sqrt((rms[feature_index] / len(input_data))) for feature_index in range(len(input_data[i]))]
    
    a_max = [max(a)]
    a_min = [min(a)]
    a_mean = [sum(a) / len(a)]
    g_max = [max(g)]
    g_min = [min(g)]
    g_mean = [sum(g) / len(g)]
    
    # 新增特徵計算
    # 峰峰值
    a_peak_to_peak = [calculate_peak_to_peak(a)]
    g_peak_to_peak = [calculate_peak_to_peak(g)]
    ax_peak_to_peak = [calculate_peak_to_peak(ax_data)]
    ay_peak_to_peak = [calculate_peak_to_peak(ay_data)]
    az_peak_to_peak = [calculate_peak_to_peak(az_data)]
    gx_peak_to_peak = [calculate_peak_to_peak(gx_data)]
    gy_peak_to_peak = [calculate_peak_to_peak(gy_data)]
    gz_peak_to_peak = [calculate_peak_to_peak(gz_data)]
    
    # 四分位數
    a_q1, a_q2, a_q3 = calculate_quartiles(a)
    g_q1, g_q2, g_q3 = calculate_quartiles(g)
    
    # 百分位數
    a_p10, a_p90 = calculate_percentiles(a)
    g_p10, g_p90 = calculate_percentiles(g)
    
    # 過零率
    a_zcr = [calculate_zero_crossing_rate(a)]
    g_zcr = [calculate_zero_crossing_rate(g)]
    ax_zcr = [calculate_zero_crossing_rate(ax_data)]
    ay_zcr = [calculate_zero_crossing_rate(ay_data)]
    az_zcr = [calculate_zero_crossing_rate(az_data)]
    gx_zcr = [calculate_zero_crossing_rate(gx_data)]
    gy_zcr = [calculate_zero_crossing_rate(gy_data)]
    gz_zcr = [calculate_zero_crossing_rate(gz_data)]
    
    # 活動計數
    a_activity = [calculate_activity_counts(a)]
    g_activity = [calculate_activity_counts(g)]
    
    # 信號幅度面積 (SMA)
    sma_a = [calculate_signal_magnitude_area(ax_data, ay_data, az_data)]
    sma_g = [calculate_signal_magnitude_area(gx_data, gy_data, gz_data)]
    
    #   # 軸間相關係數
    #   corr_xy_a = [calculate_inter_axis_correlation(ax_data, ay_data)]
    #   corr_xz_a = [calculate_inter_axis_correlation(ax_data, az_data)]
    #   corr_yz_a = [calculate_inter_axis_correlation(ay_data, az_data)]
    #   corr_xy_g = [calculate_inter_axis_correlation(gx_data, gy_data)]
    #   corr_xz_g = [calculate_inter_axis_correlation(gx_data, gz_data)]
    #   corr_yz_g = [calculate_inter_axis_correlation(gy_data, gz_data)]
        # 計算相關係數，傳入當前資料編號    
    # 假設 file_name 是當前處理的文件名
    corr_xy_a = [calculate_inter_axis_correlation(ax_data, ay_data, swinging_now, file_name)]
    corr_xz_a = [calculate_inter_axis_correlation(ax_data, az_data, swinging_now, file_name)]
    corr_yz_a = [calculate_inter_axis_correlation(ay_data, az_data, swinging_now, file_name)]
    
    corr_xy_g = [calculate_inter_axis_correlation(gx_data, gy_data, swinging_now, file_name)]
    corr_xz_g = [calculate_inter_axis_correlation(gx_data, gz_data, swinging_now, file_name)]
    corr_yz_g = [calculate_inter_axis_correlation(gy_data, gz_data, swinging_now, file_name)]
    # 軌跡長度
    traj_length_a = [calculate_trajectory_length(ax_data, ay_data, az_data)]
    traj_length_g = [calculate_trajectory_length(gx_data, gy_data, gz_data)]
    
    a_var = math.sqrt(math.pow((var[0] + var[1] + var[2]), 2))
    
    for i in range(len(input_data)):
        a_s1 = a_s1 + math.pow((a[i] - a_mean[0]), 4)
        a_s2 = a_s2 + math.pow((a[i] - a_mean[0]), 2)
        g_s1 = g_s1 + math.pow((g[i] - g_mean[0]), 4)
        g_s2 = g_s2 + math.pow((g[i] - g_mean[0]), 2)
        a_k1 = a_k1 + math.pow((a[i] - a_mean[0]), 3)
        g_k1 = g_k1 + math.pow((g[i] - g_mean[0]), 3)
    
    a_s1 = a_s1 / len(input_data)
    a_s2 = a_s2 / len(input_data)
    g_s1 = g_s1 / len(input_data)
    g_s2 = g_s2 / len(input_data)
    a_k2 = math.pow(a_s2, 1.5)
    g_k2 = math.pow(g_s2, 1.5)
    a_s2 = a_s2 * a_s2
    g_s2 = g_s2 * g_s2
    
    a_kurtosis = [a_s1 / a_s2] if a_s2 != 0 else [0]
    g_kurtosis = [g_s1 / g_s2] if g_s2 != 0 else [0]
    a_skewness = [a_k1 / a_k2] if a_k2 != 0 else [0]
    g_skewness = [g_k1 / g_k2] if g_k2 != 0 else [0]
    
    a_fft_mean = 0
    g_fft_mean = 0
    cut = int(n_fft / swinging_times)
    a_psd = []
    g_psd = []
    entropy_a = []
    entropy_g = []
    e1 = []
    e3 = []
    e2 = 0
    e4 = 0
    
    for i in range(cut * swinging_now, cut * (swinging_now + 1)):
        a_fft_mean += a_fft[i]
        g_fft_mean += g_fft[i]
        a_psd.append(math.pow(a_fft[i], 2) + math.pow(a_fft_imag[i], 2))
        g_psd.append(math.pow(g_fft[i], 2) + math.pow(g_fft_imag[i], 2))
        e1.append(math.pow(a_psd[-1], 0.5))
        e3.append(math.pow(g_psd[-1], 0.5))
        
    a_fft_mean = a_fft_mean / cut
    g_fft_mean = g_fft_mean / cut
    
    a_psd_mean = sum(a_psd) / len(a_psd)
    g_psd_mean = sum(g_psd) / len(g_psd)
    
    for i in range(cut):
        e2 += math.pow(a_psd[i], 0.5)
        e4 += math.pow(g_psd[i], 0.5)
    
    for i in range(cut):
        if e2 != 0 and e4 != 0:
            entropy_a.append((e1[i] / e2) * math.log(e1[i] / e2))
            entropy_g.append((e3[i] / e4) * math.log(e3[i] / e4))
        else:
            entropy_a.append(0)
            entropy_g.append(0)
    
    a_entropy_mean = sum(entropy_a) / len(entropy_a) if entropy_a else 0
    g_entropy_mean = sum(entropy_g) / len(entropy_g) if entropy_g else 0
    
    # 計算頻域特徵
    a_dom_freq, a_spec_centroid, a_spec_rolloff, a_spec_entropy = calculate_spectral_features(
        a_fft[cut * swinging_now:cut * (swinging_now + 1)], 
        a_fft_imag[cut * swinging_now:cut * (swinging_now + 1)]
    )
    g_dom_freq, g_spec_centroid, g_spec_rolloff, g_spec_entropy = calculate_spectral_features(
        g_fft[cut * swinging_now:cut * (swinging_now + 1)], 
        g_fft_imag[cut * swinging_now:cut * (swinging_now + 1)]
    )

    output = (mean + std + var + rms + 
            a_max + a_mean + a_min + g_max + g_mean + g_min +
            a_peak_to_peak + g_peak_to_peak + 
            ax_peak_to_peak + ay_peak_to_peak + az_peak_to_peak +
            gx_peak_to_peak + gy_peak_to_peak + gz_peak_to_peak +
            [a_q1, a_q2, a_q3, g_q1, g_q2, g_q3] +
            [a_p10, a_p90, g_p10, g_p90] +
            a_zcr + g_zcr + ax_zcr + ay_zcr + az_zcr + gx_zcr + gy_zcr + gz_zcr +
            a_activity + g_activity +
            sma_a + sma_g +
            corr_xy_a + corr_xz_a + corr_yz_a + corr_xy_g + corr_xz_g + corr_yz_g +
            traj_length_a + traj_length_g +
            [a_fft_mean] + [g_fft_mean] + [a_psd_mean] + [g_psd_mean] +
            [a_dom_freq, a_spec_centroid, a_spec_rolloff, a_spec_entropy] +
            [g_dom_freq, g_spec_centroid, g_spec_rolloff, g_spec_entropy] +
            a_kurtosis + g_kurtosis + a_skewness + g_skewness + 
            [a_entropy_mean] + [g_entropy_mean])
    writer.writerow(output)

# ==============================================================================
# XGBoost 訓練管道類 (修改後)
# ==============================================================================
class XGBoostTrainingPipeline:
    def __init__(self, random_state=42, train_data_info_path='AICUP_data/Training_Dataset/train_info.csv', raw_data_path='AICUP_data/train_data', features_output_dir='AICUP_data/tabular_data_train_xgb'):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.selected_features = {}
        self.training_info = {}
        
        self.train_data_info_path = Path(train_data_info_path)
        self.raw_data_path = Path(raw_data_path)
        self.features_output_dir = Path(features_output_dir)
        self.features_output_dir.mkdir(parents=True, exist_ok=True)

        self.model_dir = Path("./models_xgb") # 修改模型儲存目錄以區分
        self.model_dir.mkdir(exist_ok=True)
        
        print(f"🚀 XGBoost 訓練管道初始化完成")
        print(f"📁 模型儲存目錄: {self.model_dir.absolute()}")
        print(f"📁 特徵輸出目錄: {self.features_output_dir.absolute()}")

    def _get_feature_header(self):
        # 與 baseline_moreFeatures.py 中的 headerList 保持一致
        headerList = [
            'ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean',
            'ax_std', 'ay_std', 'az_std', 'gx_std', 'gy_std', 'gz_std',
            'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var', 
            'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms',
            'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min',
            'a_peak_to_peak', 'g_peak_to_peak', 
            'ax_peak_to_peak', 'ay_peak_to_peak', 'az_peak_to_peak',
            'gx_peak_to_peak', 'gy_peak_to_peak', 'gz_peak_to_peak',
            'a_q1', 'a_q2', 'a_q3', 'g_q1', 'g_q2', 'g_q3',
            'a_p10', 'a_p90', 'g_p10', 'g_p90',
            'a_zcr', 'g_zcr', 'ax_zcr', 'ay_zcr', 'az_zcr', 'gx_zcr', 'gy_zcr', 'gz_zcr',
            'a_activity', 'g_activity',
            'sma_a', 'sma_g',
            'corr_xy_a', 'corr_xz_a', 'corr_yz_a', 'corr_xy_g', 'corr_xz_g', 'corr_yz_g',
            'traj_length_a', 'traj_length_g',
            'a_fft', 'g_fft', 'a_psd', 'g_psd',
            'a_dom_freq', 'a_spec_centroid', 'a_spec_rolloff', 'a_spec_entropy',
            'g_dom_freq', 'g_spec_centroid', 'g_spec_rolloff', 'g_spec_entropy',
            'a_kurt', 'g_kurt', 'a_skewn', 'g_skewn',
            'a_entropy', 'g_entropy'
        ]
        return headerList

    def generate_features_from_raw_data(self):
        print("\n🔄 開始從原始數據生成特徵...")
        missing = [298, 459, 692, 813, 1092, 1182, 1214, 1304, 1320, 1419, 1426, 1785]
        pathlist_txt = list(self.raw_data_path.glob('**/*.txt'))
        feature_header = self._get_feature_header()
        self.feature_names = feature_header # 設定特徵名稱

        for file_path in tqdm(pathlist_txt, desc="生成特徵CSV文件"):
            file_num = int(file_path.stem)
            if file_num in missing:
                continue
            
            output_csv_path = self.features_output_dir / f"{file_path.stem}.csv"
            # 如果特徵文件已存在，可以選擇跳過以節省時間
            # if output_csv_path.exists():
            #     continue

            try:
                with open(file_path, 'r') as f_raw:
                    All_data = []
                    count = 0
                    for line in f_raw.readlines():
                        if line == '\n' or count == 0:
                            count += 1
                            continue
                        num = line.split(' ')
                        if len(num) > 5:
                            tmp_list = []
                            for i in range(6):
                                tmp_list.append(int(num[i]))
                            All_data.append(tmp_list)
                
                if not All_data:
                    print(f"警告：文件 {file_path.stem} 沒有數據")
                    continue

                swing_index = np.linspace(0, len(All_data), 28, dtype=int)

                with open(output_csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(feature_header)
                    
                    a_fft_raw, g_fft_raw = FFT_data(All_data, swing_index)
                    a_fft_imag_raw = [0] * len(a_fft_raw)
                    g_fft_imag_raw = [0] * len(g_fft_raw)
                    n_fft_val, a_fft_processed, a_fft_imag_processed = FFT(a_fft_raw, a_fft_imag_raw)
                    _, g_fft_processed, g_fft_imag_processed = FFT(g_fft_raw, g_fft_imag_raw)
                    
                    for i in range(len(swing_index)):
                        if i == 0:
                            continue
                        # 調用 feature 函數
                        feature(All_data[swing_index[i-1]: swing_index[i]], 
                                i - 1, # swinging_now
                                len(swing_index) - 1, # swinging_times_len (total number of swings)
                                n_fft_val, 
                                a_fft_processed, g_fft_processed, 
                                a_fft_imag_processed, g_fft_imag_processed, 
                                writer, 
                                file_path.stem)
            except Exception as e:
                print(f"❌ 生成特徵時出錯 ({file_path.stem}): {e}")
                continue
        print("✅ 特徵生成完成.")

    def load_and_prepare_data(self):
        print("\n📊 開始載入和準備數據...")
        info = pd.read_csv(self.train_data_info_path)
        unique_players = info['player_id'].unique()
        train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=self.random_state)
        
        print(f"   總玩家數: {len(unique_players)}")
        print(f"   訓練玩家數: {len(train_players)}")
        print(f"   測試玩家數: {len(test_players)}")
        
        datalist_csv = list(self.features_output_dir.glob('**/*.csv'))
        target_mask = ['gender', 'hold racket handed', 'play years', 'level']
        print(f"   找到 {len(datalist_csv)} 個特徵CSV文件")

        x_train_list = []
        y_train_list = []
        x_test_list = []
        y_test_list = []

        for file_csv in tqdm(datalist_csv, desc="加載特徵數據"):
            unique_id = int(Path(file_csv).stem)
            row = info[info['unique_id'] == unique_id]
            if row.empty:
                continue
            player_id = row['player_id'].iloc[0]
            data_df = pd.read_csv(file_csv)
            
            # 確保特徵名稱一致性
            if not self.feature_names:
                 self.feature_names = data_df.columns.tolist()
            elif list(data_df.columns) != self.feature_names:
                print(f"警告: 文件 {file_csv.stem} 的特徵名稱與預期不符，跳過。")
                continue

            target_values = row[target_mask]
            # 重複目標值以匹配特徵數據的行數
            target_repeated_df = pd.concat([target_values] * len(data_df), ignore_index=True)

            if player_id in train_players:
                x_train_list.append(data_df)
                y_train_list.append(target_repeated_df)
            elif player_id in test_players:
                x_test_list.append(data_df)
                y_test_list.append(target_repeated_df)
        
        if not x_train_list or not x_test_list:
            raise ValueError("訓練集或測試集為空，請檢查數據路徑和分割邏輯。")

        X_train = pd.concat(x_train_list, ignore_index=True)
        y_train = pd.concat(y_train_list, ignore_index=True)
        X_test = pd.concat(x_test_list, ignore_index=True)
        y_test = pd.concat(y_test_list, ignore_index=True)

        # 確保特徵名稱已設定
        if not self.feature_names and not X_train.empty:
            self.feature_names = X_train.columns.tolist()

        print(f"✅ 數據準備完成")
        print(f"   訓練集大小: {X_train.shape}")
        print(f"   測試集大小: {X_test.shape}")
        if self.feature_names:
             print(f"   特徵數量: {len(self.feature_names)}")
        
        return X_train, X_test, y_train, y_test

    def preprocess_data(self, X_train, X_test, y_train, y_test):
        print("\n🔧 開始數據預處理...")
        scaler = MinMaxScaler()
        
        # 確保使用 self.feature_names 進行標準化
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[self.feature_names]), columns=self.feature_names)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test[self.feature_names]), columns=self.feature_names)
        
        self.scalers['feature_scaler'] = scaler
        
        y_train_encoded = {}
        y_test_encoded = {}
        targets = ['gender', 'hold racket handed', 'play years', 'level']
        for target in targets:
            le = LabelEncoder()
            y_train_encoded[target] = le.fit_transform(y_train[target])
            # 處理測試集中可能出現的新標籤
            y_test_encoded[target] = le.transform(y_test[target]) 
            self.label_encoders[target] = le
            print(f"   {target}: {len(le.classes_)} 個類別 {le.classes_}")
        print("✅ 數據預處理完成")
        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded

    def feature_selection(self, X_train, y_train_encoded_target, target_name, top_k=50):
        print(f"\n🎯 進行特徵選擇 - {target_name}")
        # XGBoost 內建重要性
        num_class = len(self.label_encoders[target_name].classes_)
        if num_class == 2:
            model_fs = xgb.XGBClassifier(objective='binary:logistic', importance_type='gain', random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')
        else:
            model_fs = xgb.XGBClassifier(objective='multi:softprob', num_class=num_class, importance_type='gain', random_state=self.random_state, use_label_encoder=False, eval_metric='mlogloss')
        
        model_fs.fit(X_train, y_train_encoded_target)
        importance_scores = model_fs.feature_importances_
        
        # 使用 self.feature_names
        sorted_idx = np.argsort(importance_scores)[::-1]
        selected_features_names = [self.feature_names[i] for i in sorted_idx[:top_k]]
        
        print(f"   選擇了 {len(selected_features_names)} 個特徵")
        print(f"   前5個重要特徵: {selected_features_names[:5]}")
        self.selected_features[target_name] = selected_features_names
        return selected_features_names

    def train_model(self, X_train, y_train_encoded_target, target_name, selected_features_for_target):
        print(f"\n🚀 開始訓練模型 - {target_name}")
        X_train_selected = X_train[selected_features_for_target]
        
        num_class = len(self.label_encoders[target_name].classes_)
        if num_class == 2:
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                random_state=self.random_state, use_label_encoder=False, eval_metric='logloss'
            )
        else:
            model = xgb.XGBClassifier(
                objective='multi:softprob', num_class=num_class,
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                random_state=self.random_state, use_label_encoder=False, eval_metric='mlogloss'
            )
        
        model.fit(X_train_selected, y_train_encoded_target)
        # ... (cv_scores and saving logic remains similar) ...
        cv_scores = cross_val_score(
            model, X_train_selected, y_train_encoded_target, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='accuracy'
        )
        print(f"   交叉驗證分數: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        self.models[target_name] = model
        self.training_info[target_name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_features': len(selected_features_for_target),
            'feature_names': selected_features_for_target,
            'model_params': model.get_params()
        }
        return model

    def evaluate_model(self, X_test, y_test_encoded_target, target_name):
        print(f"\n📊 評估模型 - {target_name}")
        model = self.models[target_name]
        selected_features_for_target = self.selected_features[target_name]
        X_test_selected = X_test[selected_features_for_target]
        
        y_pred = model.predict(X_test_selected)
        y_pred_proba = model.predict_proba(X_test_selected)
        
        accuracy = accuracy_score(y_test_encoded_target, y_pred)
        num_class = len(self.label_encoders[target_name].classes_)

        if num_class == 2:
            # Ensure y_pred_proba has 2 columns for binary case, take prob of positive class
            auc = roc_auc_score(y_test_encoded_target, y_pred_proba[:, 1] if y_pred_proba.shape[1] == 2 else y_pred_proba)
        else:
            auc = roc_auc_score(y_test_encoded_target, y_pred_proba, multi_class='ovr', average='micro')
        
        print(f"   準確率: {accuracy:.4f}")
        print(f"   AUC: {auc:.4f}")
        # ... (classification_report and saving logic remains similar) ...
        class_names = self.label_encoders[target_name].classes_
        print(classification_report(y_test_encoded_target, y_pred, target_names=[str(c) for c in class_names]))
        self.training_info[target_name].update({
            'test_accuracy': accuracy,
            'test_auc': auc
        })
        return accuracy, auc

    def save_models_and_tools(self):
        print(f"\n💾 儲存模型和工具...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for target_name in self.models.keys():
            model_filename = f"xgb_model_{target_name.replace(' ', '_')}_{timestamp}.joblib"
            model_path = self.model_dir / model_filename
            joblib.dump(self.models[target_name], model_path)
            print(f"   ✅ 模型已儲存: {model_path}")
        
       # Save only the scalers
        scalers_filename = f"scalers_xgb_{timestamp}.pkl"
        scalers_path = self.model_dir / scalers_filename
        with open(scalers_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        print(f"   ✅ 標準化工具已儲存: {scalers_path}")
        print(f"   ✅ 預處理工具已儲存: {tools_path}")
        # ... (report saving logic remains similar) ...
        report_filename = f"training_report_xgb_{timestamp}.txt"
        report_path = self.model_dir / report_filename
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("XGBoost 模型訓練報告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"訓練時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"隨機種子: {self.random_state}\n")
            f.write(f"總特徵數: {len(self.feature_names)}\n\n")
            for target, info in self.training_info.items():
                f.write(f"目標: {target}\n")
                f.write("-" * 30 + "\n")
                f.write(f"交叉驗證分數: {info.get('cv_mean', 'N/A'):.4f} ± {info.get('cv_std', 'N/A'):.4f}\n")
                f.write(f"測試準確率: {info.get('test_accuracy', 'N/A'):.4f}\n")
                f.write(f"測試 AUC: {info.get('test_auc', 'N/A'):.4f}\n")
                f.write(f"使用特徵數: {info.get('n_features', 'N/A')}\n")
                f.write(f"前10個重要特徵: {info.get('feature_names', [])[:10]}\n\n")
        print(f"   ✅ 訓練報告已儲存: {report_path}")
        return {'models_saved': len(self.models), 'tools_path': tools_path, 'report_path': report_path, 'timestamp': timestamp}

    def run_complete_pipeline(self):
        print("🚀 開始執行完整的 XGBoost 訓練管道")
        print("=" * 60)
        try:
            # 0. (新增步驟) 從原始數據生成特徵CSV文件
            self.generate_features_from_raw_data()

            # 1. 載入和準備數據 (從生成的CSV加載)
            X_train, X_test, y_train, y_test = self.load_and_prepare_data()
            
            # 2. 數據預處理
            X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded = self.preprocess_data(
                X_train, X_test, y_train, y_test
            )
            
            targets = ['gender', 'hold racket handed', 'play years', 'level']
            results_summary = {}
            
            for target_name in targets:
                print(f"\n{'='*60}")
                print(f"處理目標: {target_name}")
                print('='*60)
                
                selected_features_for_target = self.feature_selection(X_train_scaled, y_train_encoded[target_name], target_name)
                self.train_model(X_train_scaled, y_train_encoded[target_name], target_name, selected_features_for_target)
                accuracy, auc = self.evaluate_model(X_test_scaled, y_test_encoded[target_name], target_name)
                results_summary[target_name] = {'accuracy': accuracy, 'auc': auc}
            
            save_info = self.save_models_and_tools()
            
            print(f"\n{'='*60}")
            print("🎉 訓練完成！總結報告")
            # ... (總結報告打印邏輯不變) ...
            print(f"{'目標':20s} | {'準確率':>8s} | {'AUC':>8s}")
            print("-" * 40)
            for target, result in results_summary.items():
                print(f"{target:20s} | {result['accuracy']:8.4f} | {result['auc']:8.4f}")
            print(f"\n📁 所有文件已儲存到: {self.model_dir.absolute()}")
            print(f"🕒 時間戳記: {save_info['timestamp']}")
            return results_summary, save_info
            
        except Exception as e:
            print(f"❌ 訓練過程中發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            return None, None

# ==============================================================================
# 主函數調用
# ==============================================================================
def main():
    print("🏓 AICUP 2025 桌球數據分析 - XGBoost 訓練管道 (基於 baseline_moreFeatures)")
    print("=" * 70)
    
    # 可以修改路徑參數如果需要
    pipeline = XGBoostTrainingPipeline(
        random_state=42,
        train_data_info_path='/Users/shuyuhsu/code_workspace/AICUP2025_tableTennis/AICUP_data/Training_Dataset/train_info.csv',
        raw_data_path='/Users/shuyuhsu/code_workspace/AICUP2025_tableTennis/AICUP_data/train_data',
        features_output_dir='/Users/shuyuhsu/code_workspace/AICUP2025_tableTennis/AICUP_data/tabular_data_train_xgb_generated'
    )
    
    results, save_info = pipeline.run_complete_pipeline()
    
    if results is not None:
        print("\n🎯 推薦後續步驟:")
        # ... (推薦步驟不變) ...
        print("1. 檢查儲存的模型文件")
        print("2. 使用儲存的預處理工具進行預測")
        print("3. 分析特徵重要性")
        print("4. 調整模型參數以提升性能")
        return pipeline, results, save_info
    else:
        print("❌ 訓練失敗，請檢查錯誤訊息")
        return None, None, None

if __name__ == "__main__":
    # 執行主程式
    trained_pipeline, final_results, final_save_info = main()
    # 你可以在這裡添加更多關於 trained_pipeline, final_results, final_save_info 的操作