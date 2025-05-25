from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import joblib  # 新增：用於載入 XGBoost 模型
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

def load_trained_xgb_models():
  """載入已訓練好的 XGBoost 模型"""
  print("🚀 開始評估所有 XGBoost 模型...")
  
  models = {}
  
  # XGBoost 模型檔案列表（修正檔案路徑和命名）
  model_files = {
      'level': './models_xgb/xgb_model_level_20250525_142347.joblib',
      'gender': './models_xgb/xgb_model_gender_20250525_142347.joblib', 
      'play_years': './models_xgb/xgb_model_play_years_20250525_142347.joblib',
      'hold_racket_handed': './models_xgb/xgb_model_hold_racket_handed_20250525_142347.joblib'
  }
  
  # 載入預處理工具
  preprocessing_tools_path = "./models_xgb/preprocessing_tools_xgb_20250525_142347.pkl"
  try:
      with open(preprocessing_tools_path, 'rb') as f:
          preprocessing_tools = pickle.load(f)
      print(f"✅ 已載入預處理工具: {preprocessing_tools_path}")
  except Exception as e:
      print(f"❌ 載入預處理工具失敗: {e}")
      preprocessing_tools = None
  
  # 載入每個 XGBoost 模型
  for target_name, model_file in model_files.items():
      if Path(model_file).exists():
          try:
              # 載入 XGBoost 模型
              model = joblib.load(model_file)
              
              # 構建模型數據結構（保持與原始程式一致的介面）
              model_data = {
                  'model': model,
                  'label_encoder': None,
                  'scaler': None
              }
              
              # 從預處理工具中提取對應的編碼器和縮放器
              if preprocessing_tools:
                  if 'label_encoders' in preprocessing_tools and target_name in preprocessing_tools['label_encoders']:
                      model_data['label_encoder'] = preprocessing_tools['label_encoders'][target_name]
                  
                  if 'scalers' in preprocessing_tools and target_name in preprocessing_tools['scalers']:
                      model_data['scaler'] = preprocessing_tools['scalers'][target_name]
                  elif 'scaler' in preprocessing_tools:  # 如果是共用的縮放器
                      model_data['scaler'] = preprocessing_tools['scaler']
              
              models[target_name] = model_data
              print(f"✅ 已載入 {target_name} 模型從 {model_file}")
              print(f"   - 模型類型: {type(model).__name__}")
              
              if hasattr(model, 'n_estimators'):
                  print(f"   - 樹的數量: {model.n_estimators}")
              if hasattr(model, 'n_features_in_'):
                  print(f"   - 模型期望的特徵數量: {model.n_features_in_}")
              
              # 檢查預處理工具
              if model_data['label_encoder'] is None:
                  print(f"   ⚠️ 未找到 {target_name} 的 label_encoder")
              if model_data['scaler'] is None:
                  print(f"   ⚠️ 未找到 {target_name} 的 scaler")
                  
          except Exception as e:
              print(f"❌ 載入 {model_file} 時出錯: {e}")
      else:
          print(f"❌ 找不到模型文件: {model_file}")
  
  return models

def prepare_xgb_test_data():
  """準備 XGBoost 測試數據"""
  print("📊 準備 XGBoost 測試數據...")
  
  # 從生成的特徵檔案載入測試數據
  test_data_path = "/Users/shuyuhsu/code_workspace/AICUP2025_tableTennis/AICUP_data/tabular_data_train_xgb_generated"
  
  try:
      # 載入特徵數據
      feature_files = list(Path(test_data_path).glob("*.csv"))
      
      if not feature_files:
          print(f"❌ 在 {test_data_path} 中找不到 CSV 檔案")
          return None, None
      
      print(f"   從 {test_data_path} 加載特徵數據")
      
      # 讀取所有特徵檔案
      all_features = []
      all_ids = []
      
      for file_path in feature_files:
          try:
              # 從檔案名獲取 ID
              file_id = int(file_path.stem)
              
              # 讀取特徵
              features_df = pd.read_csv(file_path)
              
              if features_df.empty:
                  print(f"   ⚠️ 文件 {file_path.name} 為空，跳過")
                  continue
              
              # 假設每個檔案只有一行特徵
              if len(features_df) > 0:
                  all_features.append(features_df.iloc[0])  # 取第一行
                  all_ids.append(file_id)
                  
          except Exception as e:
              print(f"   ⚠️ 讀取文件 {file_path.name} 失敗: {e}")
              continue
      
      if not all_features:
          print("❌ 沒有成功載入任何特徵數據")
          return None, None
          
      # 轉換為 DataFrame
      x_test = pd.DataFrame(all_features).reset_index(drop=True)
      test_ids = np.array(all_ids)
      
      print(f"   - 測試集大小: {x_test.shape}")
      print(f"   - 測試樣本數量: {len(test_ids)}")
      
      # 載入標籤數據
      label_file = "/Users/shuyuhsu/code_workspace/AICUP2025_tableTennis/AICUP_data/train_label.csv"
      
      try:
          y_test_df = pd.read_csv(label_file)
          print(f"   - 標籤檔案: {label_file}")
          print(f"   - 標籤數據形狀: {y_test_df.shape}")
          print(f"   - 標籤列名: {list(y_test_df.columns)}")
          
          # 篩選出有對應特徵的標籤
          y_test_filtered = y_test_df[y_test_df['id'].isin(test_ids)].copy()
          y_test_filtered = y_test_filtered.sort_values('id').reset_index(drop=True)
          
          # 確保特徵和標籤的順序一致
          sorted_indices = []
          final_ids = []
          for test_id in y_test_filtered['id']:
              if test_id in test_ids:
                  idx = np.where(test_ids == test_id)[0][0]
                  sorted_indices.append(idx)
                  final_ids.append(test_id)
          
          x_test = x_test.iloc[sorted_indices].reset_index(drop=True)
          y_test_filtered = y_test_filtered[y_test_filtered['id'].isin(final_ids)].reset_index(drop=True)
          
          print(f"   - 匹配後的數據量: {len(x_test)}")
          
          return x_test, y_test_filtered
          
      except Exception as e:
          print(f"❌ 載入標籤數據失敗: {e}")
          return x_test, None
          
  except Exception as e:
      print(f"❌ 載入測試數據失敗: {e}")
      return None, None

def model_binary_evaluation_xgb(model_data, X_test, y_test, target_name, group_size=27):
  """使用 AICUP 方法評估 XGBoost 二元分類模型"""
  print(f"\n🔍 評估二元分類: {target_name}")
  
  model = model_data['model']
  label_encoder = model_data['label_encoder']
  scaler = model_data['scaler']
  
  # 處理特徵數據
  if scaler is not None:
      # 標準化測試數據
      X_test_processed = scaler.transform(X_test)
  else:
      print(f"   ⚠️ 未找到縮放器，使用原始特徵")
      X_test_processed = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
  
  # 檢查特徵數量
  expected_features = model.n_features_in_
  actual_features = X_test_processed.shape[1]
  
  if expected_features != actual_features:
      print(f"   ⚠️ 特徵數量不匹配: 期望 {expected_features}, 實際 {actual_features}")
      if actual_features > expected_features:
          X_test_processed = X_test_processed[:, :expected_features]
          print(f"   📝 截取前 {expected_features} 個特徵")
      else:
          print(f"   ❌ 特徵數量不足，無法進行預測")
          return 0, 0
  
  # 處理標籤
  if label_encoder is not None:
      y_test_encoded = label_encoder.transform(y_test)
  else:
      print(f"   ⚠️ 未找到標籤編碼器，使用原始標籤")
      y_test_encoded = y_test
  
  # 預測機率
  predicted_proba = model.predict_proba(X_test_processed)
  predicted_labels = model.predict(X_test_processed)
  
  # 計算基本準確率
  accuracy = accuracy_score(y_test_encoded, predicted_labels)
  print(f"   - 基本準確率: {accuracy:.4f}")
  
  # 取出正類的概率
  if predicted_proba.shape[1] > 1:
      predicted = predicted_proba[:, 1]  # 正類概率
  else:
      predicted = predicted_proba[:, 0]
  
  # 按組聚合預測結果 (AICUP 方法)
  num_groups = len(predicted) // group_size
  if num_groups == 0:
      print(f"   ⚠️ 測試數據不足以形成完整組別")
      return accuracy, 0
  
  # 根據第一組的平均值決定聚合策略
  first_group_avg = sum(predicted[:group_size]) / group_size
  if first_group_avg > 0.5:
      y_pred_agg = [max(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
  else:
      y_pred_agg = [min(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
  
  # 聚合真實標籤
  y_test_agg = [y_test_encoded[i*group_size] for i in range(num_groups)]
  
  try:
      auc_score = roc_auc_score(y_test_agg, y_pred_agg)
      print(f"   - AICUP AUC 分數: {auc_score:.4f}")
      print(f"   - 聚合策略: {'max' if first_group_avg > 0.5 else 'min'}")
      print(f"   - 組別數量: {num_groups}")
      return accuracy, auc_score
  except Exception as e:
      print(f"   ❌ 計算 AUC 時出錯: {e}")
      return accuracy, 0

def model_multiary_evaluation_xgb(model_data, X_test, y_test, target_name, group_size=27):
  """使用 AICUP 方法評估 XGBoost 多類別分類模型"""
  print(f"\n🔍 評估多類別分類: {target_name}")
  
  model = model_data['model']
  label_encoder = model_data['label_encoder']
  scaler = model_data['scaler']
  
  # 處理特徵數據
  if scaler is not None:
      X_test_processed = scaler.transform(X_test)
  else:
      print(f"   ⚠️ 未找到縮放器，使用原始特徵")
      X_test_processed = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
  
  # 檢查特徵數量
  expected_features = model.n_features_in_
  actual_features = X_test_processed.shape[1]
  
  if expected_features != actual_features:
      print(f"   ⚠️ 特徵數量不匹配: 期望 {expected_features}, 實際 {actual_features}")
      if actual_features > expected_features:
          X_test_processed = X_test_processed[:, :expected_features]
          print(f"   📝 截取前 {expected_features} 個特徵")
      else:
          print(f"   ❌ 特徵數量不足，無法進行預測")
          return 0, 0
  
  # 處理標籤
  if label_encoder is not None:
      y_test_encoded = label_encoder.transform(y_test)
      print(f"   - 類別: {label_encoder.classes_}")
  else:
      print(f"   ⚠️ 未找到標籤編碼器，使用原始標籤")
      y_test_encoded = y_test
  
  # 預測機率和標籤
  predicted_proba = model.predict_proba(X_test_processed)
  predicted_labels = model.predict(X_test_processed)
  
  # 計算基本準確率
  accuracy = accuracy_score(y_test_encoded, predicted_labels)
  print(f"   - 基本準確率: {accuracy:.4f}")
  print(f"   - 類別數量: {predicted_proba.shape[1]}")
  
  # 按組聚合預測結果 (AICUP 方法)
  num_groups = len(predicted_proba) // group_size
  if num_groups == 0:
      print(f"   ⚠️ 測試數據不足以形成完整組別")
      return accuracy, 0
  
  y_pred_agg = []
  for i in range(num_groups):
      group_pred = predicted_proba[i*group_size: (i+1)*group_size]
      num_classes = predicted_proba.shape[1]
      
      # 對每個類別計算該組內的總機率
      class_sums = [sum([group_pred[k][j] for k in range(group_size)]) for j in range(num_classes)]
      chosen_class = np.argmax(class_sums)
      
      # 在該類別中找到最高機率的實例
      candidate_probs = [group_pred[k][chosen_class] for k in range(group_size)]
      best_instance = np.argmax(candidate_probs)
      y_pred_agg.append(group_pred[best_instance])
  
  # 聚合真實標籤
  y_test_agg = [y_test_encoded[i*group_size] for i in range(num_groups)]
  
  try:
      auc_score = roc_auc_score(y_test_agg, y_pred_agg, average='micro', multi_class='ovr')
      print(f"   - AICUP AUC 分數: {auc_score:.4f}")
      print(f"   - 組別數量: {num_groups}")
      return accuracy, auc_score
  except Exception as e:
      print(f"   ❌ 計算 AUC 時出錯: {e}")
      return accuracy, 0

def evaluate_all_xgb_models():
  """評估所有 XGBoost 模型"""
  
  # 載入 XGBoost 模型
  models = load_trained_xgb_models()
  
  if not models:
      print("❌ 沒有成功載入任何模型")
      return
  
  # 準備測試數據
  try:
      x_test, y_test = prepare_xgb_test_data()
      if x_test is None or y_test is None:
          print("❌ 準備測試數據失敗")
          return
  except Exception as e:
      print(f"❌ 準備測試數據時出錯: {e}")
      return
  
  # 評估結果存儲
  results = {}
  
  # 定義分類類型（根據你的標籤調整）
  binary_targets = ['gender', 'hold_racket_handed']
  multi_targets = ['play_years', 'level']
  
  # 評估每個模型
  for target_name, model_data in models.items():
      if target_name not in y_test.columns:
          print(f"⚠️ 標籤數據中未找到 {target_name} 列，跳過")
          continue
          
      try:
          if target_name in binary_targets:
              accuracy, auc = model_binary_evaluation_xgb(
                  model_data, x_test, y_test[target_name], target_name
              )
          elif target_name in multi_targets:
              accuracy, auc = model_multiary_evaluation_xgb(
                  model_data, x_test, y_test[target_name], target_name
              )
          else:
              print(f"⚠️ 未知的目標類型: {target_name}")
              continue
              
          results[target_name] = {'accuracy': accuracy, 'auc': auc}
          print(f"✅ {target_name} 評估完成")
          
      except Exception as e:
          print(f"❌ 評估 {target_name} 時出錯: {e}")
          import traceback
          traceback.print_exc()
  
  # 輸出總結
  print("\n" + "="*60)
  print("📊 XGBoost 模型評估結果總結")
  print("="*60)
  
  if results:
      for target_name, result in results.items():
          print(f"{target_name:20}: 準確率 = {result['accuracy']:.4f}, AUC = {result['auc']:.4f}")
      
      avg_accuracy = np.mean([result['accuracy'] for result in results.values()])
      avg_auc = np.mean([result['auc'] for result in results.values()])
      print(f"\n🎯 平均準確率: {avg_accuracy:.4f}")
      print(f"🎯 平均 AUC: {avg_auc:.4f}")
      print(f"✅ 成功評估 {len(results)} 個模型")
  else:
      print("❌ 沒有模型被成功評估")
  
  print("="*60)

if __name__ == "__main__":
  evaluate_all_xgb_models()