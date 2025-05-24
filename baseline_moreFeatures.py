from pathlib import Path
import numpy as np
import pandas as pd
import math
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from scipy import stats
from scipy.signal import find_peaks
from tqdm import tqdm # 新增導入 tqdm

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

# def calculate_inter_axis_correlation(data1, data2):
#   breakpoint()  
#   """計算軸間相關係數"""
#   if len(data1) != len(data2) or len(data1) < 2:
#       return 0
#   correlation = np.corrcoef(data1, data2)[0, 1]
#   return correlation if not np.isnan(correlation) else 0

def calculate_inter_axis_correlation(data1, data2, record_id=None):
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
            if record_id is not None:
                print(f"❌ 第 {record_id} 筆資料錯誤：標準差為零 (std1:{std1:.6f}, std2:{std2:.6f})")
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

def feature(input_data, swinging_now, swinging_times, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer):
  allsum = []
  mean = []
  var = []
  rms = []
  std = []  # 新增標準差
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
  corr_xy_a = [calculate_inter_axis_correlation(ax_data, ay_data, swinging_now)]
  corr_xz_a = [calculate_inter_axis_correlation(ax_data, az_data, swinging_now)]
  corr_yz_a = [calculate_inter_axis_correlation(ay_data, az_data, swinging_now)]
    
  corr_xy_g = [calculate_inter_axis_correlation(gx_data, gy_data, swinging_now)]
  corr_xz_g = [calculate_inter_axis_correlation(gx_data, gz_data, swinging_now)]
  corr_yz_g = [calculate_inter_axis_correlation(gy_data, gz_data, swinging_now)]
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
  
  # 組合所有特徵
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

def data_generate():
  missing = [298, 459, 692, 813, 1092, 1182, 1214, 1304, 1320, 1419, 1426, 1785]
  datapath = 'AICUP_data/train_data'
  tar_dir = 'AICUP_data/tabular_data_train'
  pathlist_txt = list(Path(datapath).glob('**/*.txt')) # 轉換為列表以獲取總數

  # 使用 tqdm 包裹迭代器以顯示進度條
  for file in tqdm(pathlist_txt, desc="Processing files"):
      file_num = int(Path(file).stem)
      if file_num in missing:
          continue
      f = open(file)

      All_data = []

      count = 0
      for line in f.readlines():
          if line == '\n' or count == 0:
              count += 1
              continue
          num = line.split(' ')
          if len(num) > 5:
              tmp_list = []
              for i in range(6):
                  tmp_list.append(int(num[i]))
              All_data.append(tmp_list)
      
      f.close()
      # print(f"Checking data for file: {file}") # 這行可以選擇性保留或移除，因為進度條會顯示進度

      swing_index = np.linspace(0, len(All_data), 28, dtype = int)

      # 更新特徵列表
      headerList = [
          # 基礎統計特徵 (24個)
          'ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean',
          'ax_std', 'ay_std', 'az_std', 'gx_std', 'gy_std', 'gz_std',
          'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var', 
          'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms',
          # 合成向量特徵 (6個)
          'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min',
          # 峰峰值特徵 (8個)
          'a_peak_to_peak', 'g_peak_to_peak', 
          'ax_peak_to_peak', 'ay_peak_to_peak', 'az_peak_to_peak',
          'gx_peak_to_peak', 'gy_peak_to_peak', 'gz_peak_to_peak',
          # 四分位數特徵 (6個)
          'a_q1', 'a_q2', 'a_q3', 'g_q1', 'g_q2', 'g_q3',
          # 百分位數特徵 (4個)
          'a_p10', 'a_p90', 'g_p10', 'g_p90',
          # 過零率特徵 (8個)
          'a_zcr', 'g_zcr', 'ax_zcr', 'ay_zcr', 'az_zcr', 'gx_zcr', 'gy_zcr', 'gz_zcr',
          # 活動計數特徵 (2個)
          'a_activity', 'g_activity',
          # 信號幅度面積 (2個)
          'sma_a', 'sma_g',
          # 軸間相關係數 (6個)
          'corr_xy_a', 'corr_xz_a', 'corr_yz_a', 'corr_xy_g', 'corr_xz_g', 'corr_yz_g',
          # 軌跡長度 (2個)
          'traj_length_a', 'traj_length_g',
          # 原有頻域特徵 (4個)
          'a_fft', 'g_fft', 'a_psd', 'g_psd',
          # 新增頻域特徵 (8個)
          'a_dom_freq', 'a_spec_centroid', 'a_spec_rolloff', 'a_spec_entropy',
          'g_dom_freq', 'g_spec_centroid', 'g_spec_rolloff', 'g_spec_entropy',
          # 高階統計特徵 (4個)
          'a_kurt', 'g_kurt', 'a_skewn', 'g_skewn',
          # 信息熵特徵 (2個)
          'a_entropy', 'g_entropy'
      ]

      with open('./{dir}/{fname}.csv'.format(dir = tar_dir, fname = Path(file).stem), 'w', newline = '') as csvfile:
          writer = csv.writer(csvfile)
          writer.writerow(headerList)
          try:
              a_fft, g_fft = FFT_data(All_data, swing_index)
              a_fft_imag = [0] * len(a_fft)
              g_fft_imag = [0] * len(g_fft)
              n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
              n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)
              for i in range(len(swing_index)):
                  if i==0:
                      continue
                  feature(All_data[swing_index[i-1]: swing_index[i]], i - 1, len(swing_index) - 1, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer)
          except Exception as e:
              print(f"Error processing {Path(file).stem}: {e}")
              continue

def main():
    # 若尚未產生特徵，請先執行 data_generate() 生成特徵 CSV 檔案
    data_generate()
    
    # 讀取訓練資訊，根據 player_id 將資料分成 80% 訓練、20% 測試
    info = pd.read_csv('/Users/shuyuhsu/code_workspace/AICUP2025_tableTennis/AICUP_data/Training_Dataset/train_info.csv')
    unique_players = info['player_id'].unique()
    train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)
    
    # 讀取特徵 CSV 檔（位於指定資料夾）
    datapath = '/Users/shuyuhsu/code_workspace/AICUP2025_tableTennis/AICUP_data/tabular_data_train'
    datalist = list(Path(datapath).glob('**/*.csv'))
    target_mask = ['gender', 'hold racket handed', 'play years', 'level']
    
    # 根據 test_players 分組資料
    x_train = pd.DataFrame()
    y_train = pd.DataFrame(columns=target_mask)
    x_test = pd.DataFrame()
    y_test = pd.DataFrame(columns=target_mask)
    
    for file in datalist:
        unique_id = int(Path(file).stem)
        row = info[info['unique_id'] == unique_id]
        if row.empty:
            continue
        player_id = row['player_id'].iloc[0]
        data = pd.read_csv(file)
        target = row[target_mask]
        target_repeated = pd.concat([target] * len(data))
        if player_id in train_players:
            x_train = pd.concat([x_train, data], ignore_index=True)
            y_train = pd.concat([y_train, target_repeated], ignore_index=True)
        elif player_id in test_players:
            x_test = pd.concat([x_test, data], ignore_index=True)
            y_test = pd.concat([y_test, target_repeated], ignore_index=True)
    
    # 標準化特徵
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    # 檢查資料集大小
    print(f"訓練集大小: {X_train_scaled.shape}")
    print(f"測試集大小: {X_test_scaled.shape}")
    print(f"特徵數量: {X_train_scaled.shape[1]}")
    
    # 為每個目標變數訓練模型並評估
    results = {}
    
    for target in target_mask:
        print(f"\n=== 訓練 {target} 分類器 ===")
        
        # 編碼標籤
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train[target])
        y_test_encoded = le.transform(y_test[target])
        
        # 訓練隨機森林分類器
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train_scaled, y_train_encoded)
        
        # 預測
        y_pred = rf.predict(X_test_scaled)
        y_pred_proba = rf.predict_proba(X_test_scaled)
        
        # 計算準確率
        accuracy = (y_pred == y_test_encoded).mean()
        # 計算 AUC
        if target in ['play years', 'level']:
            # 對於多分類目標，使用 predict_proba 的完整輸出
            auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr', average='micro')
            print(f"{target} Micro-avg ROC AUC: {auc:.4f}")
        elif target in ['gender', 'hold racket handed']:
            # 對於二元分類目標，通常取預測為正類的機率
            # 假設 y_pred_proba 的第二列是正類的機率
            if y_pred_proba.shape[1] > 1: # 確保至少有兩類
                auc = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
                print(f"{target} ROC AUC: {auc:.4f}")
            else: # 如果模型只預測一種類別（雖然不太可能，但作為防禦性程式碼）
                print(f"Warning: Model for {target} only predicts one class. AUC cannot be calculated meaningfully.")
                auc = float('nan') # 或者設定為 0 或其他合適的值
        else:
            # 其他情況（如果有的話）
            print(f"Warning: AUC calculation not specifically defined for target '{target}'.")
            auc = float('nan')
        print(f"AUC (ovr): {auc:.4f}")
        
        print(f"準確率: {accuracy:.4f}")
        # 特徵重要性分析
        feature_importance = rf.feature_importances_
        feature_names = x_train.columns
        top_features_idx = np.argsort(feature_importance)[-10:]  # 前10個重要特徵
        
        print(f"準確率: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"類別: {le.classes_}")
        print("前10個重要特徵:")
        for idx in reversed(top_features_idx):
            print(f"  {feature_names[idx]}: {feature_importance[idx]:.4f}")
        
        # 儲存結果
        results[target] = {
            'accuracy': accuracy,
            'auc': auc,
            'model': rf,
            'label_encoder': le,
            'feature_importance': feature_importance,
            'top_features': [(feature_names[idx], feature_importance[idx]) 
                           for idx in reversed(top_features_idx)]
        }
    
    # 總結報告
    print("\n" + "="*50)
    print("總結報告")
    print("="*50)
    
    for target, result in results.items():
        print(f"{target:20s} | 準確率: {result['accuracy']:.4f} | AUC: {result['auc']:.4f}")
    
    # 儲存模型和結果
    import pickle
    
    model_save_path = '/Users/shuyuhsu/code_workspace/AICUP2025_tableTennis/models'
    Path(model_save_path).mkdir(exist_ok=True)
    
    for target, result in results.items():
        # 儲存模型
        with open(f'{model_save_path}/{target}_model.pkl', 'wb') as f:
            pickle.dump({
                'model': result['model'],
                'label_encoder': result['label_encoder'],
                'scaler': scaler,
                'feature_names': list(x_train.columns)
            }, f)
        
        print(f"已儲存 {target} 模型至 {model_save_path}/{target}_model.pkl")
    
    # 特徵重要性綜合分析
    print("\n" + "="*50)
    print("特徵重要性綜合分析")
    print("="*50)
    
    # 計算所有目標的平均特徵重要性
    # 確保 X_scaled_df.columns 或等效的特徵名稱列表可用
    # 假設 feature_names_overall 是所有特徵的名稱列表
    # 如果 X_scaled 是一個 numpy array，你需要一個預先定義好的 feature_names_overall 列表
    # 如果 X_scaled 是一個 DataFrame，則 feature_names_overall = X_scaled.columns
    # 這裡假設 X_train.columns 可以代表全局的特徵名稱
    num_features = x_train.shape[1]
    all_importance = np.zeros(num_features)
    count_models_for_importance = 0
    for target_name, result in results.items():
        if 'feature_importance' in result and len(result['feature_importance']) == num_features:
            all_importance += result['feature_importance']
            count_models_for_importance += 1
    
    if count_models_for_importance > 0:
        avg_importance = all_importance / count_models_for_importance
        top_overall_idx = np.argsort(avg_importance)[-20:]
        
        print("整體最重要的15個特徵:")
        # 確保 feature_names 在這裡仍然可用且正確
        # 如果 x_train 是 DataFrame，x_train.columns 應該是正確的
        current_feature_names = x_train.columns if hasattr(x_train, 'columns') else [f'feature_{i}' for i in range(num_features)]
        for idx in reversed(top_overall_idx):
            print(f"  {current_feature_names[idx]:25s}: {avg_importance[idx]:.4f}")
    else:
        print("未能計算整體特徵重要性，因為沒有模型提供了特徵重要性數據或特徵數量不匹配。")

    return results, scaler, x_train.columns

# 執行主程式
if __name__ == "__main__":
    results, scaler, feature_names = main()
