import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import math
import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import kurtosis, skew
import xgboost as xgb  # 新增 XGBoost 導入


def FFT(xreal, ximag):
    n = 2
    while (n * 2 <= len(xreal)):
        n *= 2

    p = int(math.log(n, 2))

    for i in range(0, n):
        a = i
        b = 0
        for j in range(0, p):
            b = int(b * 2 + a % 2)
            a = a / 2
        if (b > i):
            xreal[i], xreal[b] = xreal[b], xreal[i]
            ximag[i], ximag[b] = ximag[b], ximag[i]

    wreal = []
    wimag = []

    arg = float(-2 * math.pi / n)
    treal = float(math.cos(arg))
    timag = float(math.sin(arg))

    wreal.append(float(1.0))
    wimag.append(float(0.0))

    for j in range(1, int(n / 2)):
        wreal.append(wreal[-1] * treal - wimag[-1] * timag)
        wimag.append(wreal[-1] * timag + wimag[-1] * treal)

    m = 2
    while (m < n + 1):
        for k in range(0, n, m):
            for j in range(0, int(m / 2), 1):
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

    for num in range(len(swinging_times) - 1):
        a = []
        g = []
        for swing in range(swinging_times[num], swinging_times[num + 1]):
            a.append(math.sqrt(math.pow((input_data[swing][0] + input_data[swing][1] + input_data[swing][2]), 2)))
            g.append(math.sqrt(math.pow((input_data[swing][3] + input_data[swing][4] + input_data[swing][5]), 2)))

        a_mean[num] = (sum(a) / len(a))
        g_mean[num] = (sum(a) / len(a))

    return a_mean, g_mean


def feature(input_data, swinging_now, swinging_times, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer):
    input_data = np.array(input_data)
    ax, ay, az, gx, gy, gz = input_data[:, 0], input_data[:, 1], input_data[:, 2], input_data[:, 3], input_data[:, 4], input_data[:, 5]

    # 合成向量
    a = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    g = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    # 1. 基礎統計特徵 (24 個)
    mean = np.mean(input_data, axis=0)
    std = np.std(input_data, axis=0)
    var = np.var(input_data, axis=0)
    rms = np.sqrt(np.mean(input_data ** 2, axis=0))

    # 2. 合成向量特徵 (6 個)
    a_max = np.max(a)
    a_mean = np.mean(a)
    a_min = np.min(a)
    g_max = np.max(g)
    g_mean = np.mean(g)
    g_min = np.min(g)

    # 3. 峰峰值特徵 (8 個)
    a_peak_to_peak = a_max - a_min
    g_peak_to_peak = g_max - g_min
    ax_peak_to_peak = np.max(ax) - np.min(ax)
    ay_peak_to_peak = np.max(ay) - np.min(ay)
    az_peak_to_peak = np.max(az) - np.min(az)
    gx_peak_to_peak = np.max(gx) - np.min(gx)
    gy_peak_to_peak = np.max(gy) - np.min(gy)
    gz_peak_to_peak = np.max(gz) - np.min(gz)

    # 4. 四分位數特徵 (6 個)
    a_q1, a_q2, a_q3 = np.percentile(a, [25, 50, 75])
    g_q1, g_q2, g_q3 = np.percentile(g, [25, 50, 75])

    # 5. 百分位數特徵 (4 個)
    a_p10, a_p90 = np.percentile(a, [10, 90])
    g_p10, g_p90 = np.percentile(g, [10, 90])

    # 6. 過零率特徵 (8 個)
    def zcr(data):
        return np.sum(np.abs(np.diff(np.sign(data))) / 2) / len(data)

    a_zcr = zcr(a)
    g_zcr = zcr(g)
    ax_zcr = zcr(ax)
    ay_zcr = zcr(ay)
    az_zcr = zcr(az)
    gx_zcr = zcr(gx)
    gy_zcr = zcr(gy)
    gz_zcr = zcr(gz)

    # 7. 活動計數特徵 (2 個)
    def activity_count(data, threshold=0.1):
        return np.sum(np.abs(data) > threshold) / len(data)

    a_activity = activity_count(a)
    g_activity = activity_count(g)

    # 8. 信號幅度面積 (2 個)
    sma_a = np.sum(np.abs(input_data[:, :3]), axis=1).mean()
    sma_g = np.sum(np.abs(input_data[:, 3:]), axis=1).mean()

    # 9. 軸間相關係數 (6 個)
    def safe_corr(x, y):
        # 檢查標準差是否為零
        std_x = np.std(x)
        std_y = np.std(y)
        if std_x == 0 or std_y == 0:
            return 0  # 如果任一軸的標準差為零，返回相關係數為零
        else:
            return np.corrcoef(x, y)[0, 1]

    corr_xy_a = safe_corr(ax, ay)
    corr_xz_a = safe_corr(ax, az)
    corr_yz_a = safe_corr(ay, az)
    corr_xy_g = safe_corr(gx, gy)
    corr_xz_g = safe_corr(gx, gz)
    corr_yz_g = safe_corr(gy, gz)

    # 10. 軌跡長度 (2 個)
    def traj_length(data):
        diff = np.diff(data, axis=0)
        return np.sum(np.sqrt(np.sum(diff ** 2, axis=1)))

    traj_length_a = traj_length(input_data[:, :3])
    traj_length_g = traj_length(input_data[:, 3:])

    # 11. 原有頻域特徵 (4 個)
    cut = int(n_fft / swinging_times)
    a_fft_mean = np.mean(a_fft[cut * swinging_now: cut * (swinging_now + 1)])
    g_fft_mean = np.mean(g_fft[cut * swinging_now: cut * (swinging_now + 1)])
    a_psd = np.array([math.pow(a_fft[i], 2) + math.pow(a_fft_imag[i], 2) for i in range(cut * swinging_now, cut * (swinging_now + 1))])
    g_psd = np.array([math.pow(g_fft[i], 2) + math.pow(g_fft_imag[i], 2) for i in range(cut * swinging_now, cut * (swinging_now + 1))])
    a_psd_mean = np.mean(a_psd)
    g_psd_mean = np.mean(g_psd)

    # 12. 新增頻域特徵 (8 個)
    def get_dom_freq(psd, fs):
        freqs = np.fft.fftfreq(len(psd), 1 / fs)
        return freqs[np.argmax(psd)]

    def get_spec_centroid(psd, fs):
        # 避免除以零
        if np.sum(psd) == 0:
            return 0
        freqs = np.fft.fftfreq(len(psd), 1 / fs)
        return np.sum(freqs * psd) / np.sum(psd)

    def get_spec_rolloff(psd, fs, rolloff=0.85):
        # 避免除以零
        if np.sum(psd) == 0:
            return 0
        freqs = np.fft.fftfreq(len(psd), 1 / fs)
        cumsum = np.cumsum(psd)
        total_energy = cumsum[-1]
        if total_energy == 0:
            return 0
        return freqs[np.argmax(cumsum >= rolloff * total_energy)]

    def get_spec_entropy(psd):
        # 避免除以零和對數運算中的零值
        psd_sum = np.sum(psd)
        if psd_sum == 0:
            return 0
        psd_norm = psd / psd_sum
        # 過濾掉零值，避免 log2(0)
        psd_norm = psd_norm[psd_norm > 0]
        return -np.sum(psd_norm * np.log2(psd_norm))

    fs = 85  # 設定采樣頻率為每秒85筆
    a_dom_freq = get_dom_freq(a_psd, fs)
    a_spec_centroid = get_spec_centroid(a_psd, fs)
    a_spec_rolloff = get_spec_rolloff(a_psd, fs)
    a_spec_entropy = get_spec_entropy(a_psd)
    g_dom_freq = get_dom_freq(g_psd, fs)
    g_spec_centroid = get_spec_centroid(g_psd, fs)
    g_spec_rolloff = get_spec_rolloff(g_psd, fs)
    g_spec_entropy = get_spec_entropy(g_psd)

    # 13. 高階統計特徵 (4 個)
    # 使用 nan-safe 版本的 kurtosis 和 skew，忽略 NaN 值
    a_kurt = kurtosis(a, nan_policy='omit')
    g_kurt = kurtosis(g, nan_policy='omit')
    a_skewn = skew(a, nan_policy='omit')
    g_skewn = skew(g, nan_policy='omit')

    # 14. 信息熵特徵 (2 個)
    def entropy(data):
        # 避免除以零和對數運算中的零值
        psd = np.abs(np.fft.fft(data)) ** 2
        psd_sum = np.sum(psd)
        if psd_sum == 0:
            return 0
        psd_norm = psd / psd_sum
        # 過濾掉零值，避免 log2(0)
        psd_norm = psd_norm[psd_norm > 0]
        return -np.sum(psd_norm * np.log2(psd_norm))

    a_entropy = entropy(a)
    g_entropy = entropy(g)

    output = list(mean) + list(std) + list(var) + list(rms) + \
             [a_max, a_mean, a_min, g_max, g_mean, g_min] + \
             [a_peak_to_peak, g_peak_to_peak, ax_peak_to_peak, ay_peak_to_peak, az_peak_to_peak, gx_peak_to_peak, gy_peak_to_peak, gz_peak_to_peak] + \
             [a_q1, a_q2, a_q3, g_q1, g_q2, g_q3] + \
             [a_p10, a_p90, g_p10, g_p90] + \
             [a_zcr, g_zcr, ax_zcr, ay_zcr, az_zcr, gx_zcr, gy_zcr, gz_zcr] + \
             [a_activity, g_activity] + \
             [sma_a, sma_g] + \
             [corr_xy_a, corr_xz_a, corr_yz_a, corr_xy_g, corr_xz_g, corr_yz_g] + \
             [traj_length_a, traj_length_g] + \
             [a_fft_mean, g_fft_mean, a_psd_mean, g_psd_mean] + \
             [a_dom_freq, a_spec_centroid, a_spec_rolloff, a_spec_entropy, g_dom_freq, g_spec_centroid, g_spec_rolloff, g_spec_entropy] + \
             [a_kurt, g_kurt, a_skewn, g_skewn] + \
             [a_entropy, g_entropy]

    writer.writerow(output)


def update_progress(progress, total, file_name=None):
    """更新進度條"""
    bar_length = 50
    filled_length = int(round(bar_length * progress / total))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    percentage = round(100.0 * progress / total, 1)

    if file_name:
        sys.stdout.write(f'\r進度: [{bar}] {percentage}% | 正在處理: {file_name}')
    else:
        sys.stdout.write(f'\r進度: [{bar}] {percentage}%')
    sys.stdout.flush()


def data_generate():
    missing = [298, 459, 692, 813, 1092, 1182, 1214, 1304, 1320, 1419, 1426, 1785]
    datapath = 'AICUP_data/train_data'
    tar_dir = 'AICUP_data/tabular_data_train'
    pathlist_txt = list(Path(datapath).glob('**/*.txt'))
    total_files = len(pathlist_txt)

    print(f"開始處理 {total_files} 個文件...")

    for i, file in enumerate(pathlist_txt):
        file_num = int(Path(file).stem)
        file_name = Path(file).name

        # 更新進度條
        update_progress(i, total_files, file_name)

        if file_num in missing:
            continue

        try:
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

            swing_index = np.linspace(0, len(All_data), 28, dtype=int)

            headerList = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean',
                          'ax_std', 'ay_std', 'az_std', 'gx_std', 'gy_std', 'gz_std',
                          'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var',
                          'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms',
                          'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min',
                          'a_peak_to_peak', 'g_peak_to_peak', 'ax_peak_to_peak', 'ay_peak_to_peak', 'az_peak_to_peak', 'gx_peak_to_peak', 'gy_peak_to_peak', 'gz_peak_to_peak',
                          'a_q1', 'a_q2', 'a_q3', 'g_q1', 'g_q2', 'g_q3',
                          'a_p10', 'a_p90', 'g_p10', 'g_p90',
                          'a_zcr', 'g_zcr', 'ax_zcr', 'ay_zcr', 'az_zcr', 'gx_zcr', 'gy_zcr', 'gz_zcr',
                          'a_activity', 'g_activity',
                          'sma_a', 'sma_g',
                          'corr_xy_a', 'corr_xz_a', 'corr_yz_a', 'corr_xy_g', 'corr_xz_g', 'corr_yz_g',
                          'traj_length_a', 'traj_length_g',
                          'a_fft', 'g_fft', 'a_psd', 'g_psd',
                          'a_dom_freq', 'a_spec_centroid', 'a_spec_rolloff', 'a_spec_entropy', 'g_dom_freq', 'g_spec_centroid', 'g_spec_rolloff', 'g_spec_entropy',
                          'a_kurt', 'g_kurt', 'a_skewn', 'g_skewn',
                          'a_entropy', 'g_entropy']

            with open('./{dir}/{fname}.csv'.format(dir=tar_dir, fname=Path(file).stem), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headerList)

                a_fft, g_fft = FFT_data(All_data, swing_index)
                a_fft_imag = [0] * len(a_fft)
                g_fft_imag = [0] * len(g_fft)
                n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
                n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)

                for i in range(len(swing_index)):
                    if i == 0:
                        continue
                    feature(All_data[swing_index[i - 1]: swing_index[i]], i - 1, len(swing_index) - 1, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer)
        except Exception as e:
            # 打印錯誤信息
            print(f"\n錯誤: 處理文件 {file_name} 時發生異常: {str(e)}")
            continue

    # 進度條完成
    update_progress(total_files, total_files)
    print("\n文件處理完成!")


import pickle  # 確保引入 pickle 模組

def main():
    # 若尚未產生特徵，請先執行 data_generate() 生成特徵 CSV 檔案
    data_generate()

    # 讀取訓練資訊，根據 player_id 將資料分成 80% 訓練、20% 測試
    info = pd.read_csv('AICUP_data/Training_Dataset/train_info.csv')
    unique_players = info['player_id'].unique()
    train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)

    # 讀取特徵 CSV 檔（位於指定資料夾）
    datapath = 'AICUP_data/tabular_data_train'
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
    # 保存到文件
    import os
    if not os.path.exists('./models_xgb'):
        os.makedirs('./models_xgb')
    
   
        
    le = LabelEncoder()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)
    with open('./models_xgb/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    group_size = 27

    def model_binary(X_train, y_train, X_test, y_test, model_name):
        # 替換為 XGBoost 二分類模型
        clf = xgb.XGBClassifier(
                # 基礎設定
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                
                # 核心參數
                n_estimators=300,           # 樹的數量
                max_depth=6,                # 樹的深度
                learning_rate=0.1,          # 學習率
                
                # 正則化參數
                reg_alpha=0.1,              # L1 正則化
                reg_lambda=1.0,             # L2 正則化
                
                # 隨機性參數（防止過擬合）
                subsample=0.8,              # 樣本隨機採樣比例
                colsample_bytree=0.8,       # 特徵隨機採樣比例
                colsample_bylevel=0.8,      # 每層特徵採樣比例
                
                # 葉子節點控制
                min_child_weight=3,         # 葉子節點最小權重
                gamma=0.1,                  # 最小分割損失
                
                # 性能優化
                n_jobs=-1,                  # 使用所有CPU核心
                verbosity=0,                # 減少輸出
            )

        clf.fit(X_train, y_train)

        predicted = clf.predict_proba(X_test)
        # 取出正類（index 0）的概率
        predicted = [predicted[i][0] for i in range(len(predicted))]

        num_groups = len(predicted) // group_size
        if sum(predicted[:group_size]) / group_size > 0.5:
            y_pred = [max(predicted[i * group_size: (i + 1) * group_size]) for i in range(num_groups)]
        else:
            y_pred = [min(predicted[i * group_size: (i + 1) * group_size]) for i in range(num_groups)]

        y_pred = [1 - x for x in y_pred]
        y_test_agg = [y_test[i * group_size] for i in range(num_groups)]

        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro')
        print(auc_score)

        # 使用 pickle 保存模型
        with open(f'./models_xgb/{model_name}.pkl', 'wb') as f:
            pickle.dump(clf, f)

    # 定義多類別分類評分函數 (例如 play years、level)
    def model_multiary(X_train, y_train, X_test, y_test, model_name):
        # 替換為 XGBoost 多分類模型
        clf = xgb.XGBClassifier(
                # 基礎設定
                objective='multi:softprob',
                eval_metric='mlogloss',
                random_state=42,
                
                # 核心參數
                n_estimators=300,           # 樹的數量
                max_depth=6,                # 樹的深度
                learning_rate=0.1,          # 學習率
                
                # 正則化參數
                reg_alpha=0.1,              # L1 正則化
                reg_lambda=1.0,             # L2 正則化
                
                # 隨機性參數（防止過擬合）
                subsample=0.8,              # 樣本隨機採樣比例
                colsample_bytree=0.8,       # 特徵隨機採樣比例
                colsample_bylevel=0.8,      # 每層特徵採樣比例
                
                # 性能優化
                n_jobs=-1,                  # 使用所有CPU核心
                verbosity=0,                # 減少輸出
                
                # 其他
                min_child_weight=3,         # 葉子節點最小權重
                gamma=0.1,                  # 最小分割損失
            )
        clf.fit(X_train, y_train)
        predicted = clf.predict_proba(X_test)
        num_groups = len(predicted) // group_size
        y_pred = []
        for i in range(num_groups):
            group_pred = predicted[i * group_size: (i + 1) * group_size]
            num_classes = len(np.unique(y_train))
            # 對每個類別計算該組內的總機率
            class_sums = [sum([group_pred[k][j] for k in range(group_size)]) for j in range(num_classes)]
            chosen_class = np.argmax(class_sums)
            candidate_probs = [group_pred[k][chosen_class] for k in range(group_size)]
            best_instance = np.argmax(candidate_probs)
            y_pred.append(group_pred[best_instance])

        y_test_agg = [y_test[i * group_size] for i in range(num_groups)]
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro', multi_class='ovr')
        print('Multiary AUC:', auc_score)

        # 使用 pickle 保存模型
        with open(f'./models_xgb/{model_name}.pkl', 'wb') as f:
            pickle.dump(clf, f)

    # 評分：針對各目標進行模型訓練與評分
    y_train_le_gender = le.fit_transform(y_train['gender'])
    y_test_le_gender = le.transform(y_test['gender'])
    model_binary(X_train_scaled, y_train_le_gender, X_test_scaled, y_test_le_gender, 'gender_model')

    y_train_le_hold = le.fit_transform(y_train['hold racket handed'])
    y_test_le_hold = le.transform(y_test['hold racket handed'])
    model_binary(X_train_scaled, y_train_le_hold, X_test_scaled, y_test_le_hold, 'hold_model')

    y_train_le_years = le.fit_transform(y_train['play years'])
    y_test_le_years = le.transform(y_test['play years'])
    model_multiary(X_train_scaled, y_train_le_years, X_test_scaled, y_test_le_years, 'years_model')

    y_train_le_level = le.fit_transform(y_train['level'])
    y_test_le_level = le.transform(y_test['level'])
    model_multiary(X_train_scaled, y_train_le_level, X_test_scaled, y_test_le_level, 'level_model')

    # AUC SCORE: 0.792(gender) + 0.998(hold) + 0.660(years) + 0.822(levels)

if __name__ == '__main__':
    main()
