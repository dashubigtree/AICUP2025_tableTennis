# AICUP2025 Table Tennis

本專案為AICUP2025桌球相關任務。
# 目標：
平均AUC分數要高於0.8197

# BaseLine 相關function 用途
## data generate
根據你提供的程式碼內容， data_generate 這個 function 的主要作用如下：

1. 讀取原始感測資料 它會從 datapath = '/Users/shuyuhsu/code_workspace/AICUP2025_tableTennis/AICUP_data/train_data' 這個資料夾下，逐一讀取所有 .txt 檔案。每個 txt 檔案記錄了一次測驗的連續感測數據（如加速度、角速度等六軸資料）。
2. 資料前處理與切割 針對每個檔案，會先將每一行資料轉成數字陣列，並依據資料長度自動切成 27 段（28 個切點），每段對應一次揮拍。
3. 特徵萃取 對每一段揮拍資料，呼叫 feature 函數進行特徵計算。特徵包括：
   
   - 各軸的平均值、變異數、RMS
   - 加速度與角速度的最大/最小/平均值
   - 頻域特徵（FFT、PSD、熵等）
   - 偏態、峰態等統計特徵
4. 儲存為 CSV 每個原始 txt 會對應產生一個 csv 檔，存到 tar_dir = '/Users/shuyuhsu/code_workspace/AICUP2025_tableTennis/AICUP_data/tabular_data_train' 。每個 csv 檔的每一列就是一段揮拍的特徵向量。
5. 用途 這些整理好的 csv 特徵檔，會被後續的機器學習模型（如 main 裡的訓練流程）讀取，作為訓練與預測的輸入資料

#### train data的txt files有缺少：缺少的編號： [298, 459, 692, 813, 1092, 1182, 1214, 1304, 1320, 1419, 1426, 1785]

# XGBoost第一次提交分數：0.78423
今天也將86個特徵值的數據計算結果和要預測的目標都儲存到processed_data folder當中，可以針對這份數據來做進一步的分析。