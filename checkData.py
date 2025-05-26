import pandas as pd
import numpy as np

def check_submission_file(file_path):
    """檢查提交文件中的機率總和是否為1，並檢查小數點位數"""
    # 讀取提交文件
    df = pd.read_csv(file_path)
    print(f"檔案形狀: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print("\n前5行數據:")
    print(df.head())
    
    # 檢查小數點位數
    decimal_places = {}
    for col in df.columns:
        if col != 'unique_id':  # 跳過ID列
            # 獲取每個值的小數點位數
            decimal_places[col] = df[col].apply(
                lambda x: len(str(float(x)).split('.')[-1]) if '.' in str(float(x)) else 0
            ).max()
    
    print("\n各列小數點位數:")
    for col, places in decimal_places.items():
        print(f"{col}: {places}")
    
    # 檢查多類別機率總和
    years_cols = [col for col in df.columns if col.startswith('play years_')]
    level_cols = [col for col in df.columns if col.startswith('level_')]
    
    # 檢查play years各類別機率總和
    years_sums = df[years_cols].sum(axis=1)
    print("\nPlay years機率總和統計:")
    print(f"最小值: {years_sums.min()}")
    print(f"最大值: {years_sums.max()}")
    print(f"平均值: {years_sums.mean()}")
    print(f"標準差: {years_sums.std()}")
    
    # 檢查level各類別機率總和
    level_sums = df[level_cols].sum(axis=1)
    print("\nLevel機率總和統計:")
    print(f"最小值: {level_sums.min()}")
    print(f"最大值: {level_sums.max()}")
    print(f"平均值: {level_sums.mean()}")
    print(f"標準差: {level_sums.std()}")
    
    # 檢查是否有總和不為1的行
    years_not_one = df[abs(years_sums - 1) > 0.0001].shape[0]
    level_not_one = df[abs(level_sums - 1) > 0.0001].shape[0]
    print(f"\nPlay years機率總和不為1的行數: {years_not_one}")
    print(f"Level機率總和不為1的行數: {level_not_one}")
    
    # 如果有總和不為1的行，顯示前幾個例子
    if years_not_one > 0:
        print("\nPlay years機率總和不為1的例子:")
        not_one_idx = df[abs(years_sums - 1) > 0.0001].index[:5]
        for idx in not_one_idx:
            print(f"ID {df.loc[idx, 'unique_id']}: {df.loc[idx, years_cols].tolist()}, 總和={years_sums[idx]}")
    
    if level_not_one > 0:
        print("\nLevel機率總和不為1的例子:")
        not_one_idx = df[abs(level_sums - 1) > 0.0001].index[:5]
        for idx in not_one_idx:
            print(f"ID {df.loc[idx, 'unique_id']}: {df.loc[idx, level_cols].tolist()}, 總和={level_sums[idx]}")

# 修改以下路徑為您的提交文件路徑
submission_path = './submission/xgboost_submission.csv'
check_submission_file(submission_path)