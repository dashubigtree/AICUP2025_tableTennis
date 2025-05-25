from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

def load_trained_models(model_files):
  """載入已訓練好的模型"""
  models = {}
  
  for model_file in model_files:
      if Path(model_file).exists():
          try:
              with open(model_file, 'rb') as f:
                  model_data = pickle.load(f)
              
              # 從檔名提取目標名稱
              target_name = Path(model_file).stem.replace('_model', '')
              models[target_name] = model_data
              print(f"✅ 已載入 {target_name} 模型")
              
              # 顯示模型基本資訊
              if 'model' in model_data:
                  model = model_data['model']
                  print(f"   - 模型類型: {type(model).__name__}")
                  if hasattr(model, 'n_estimators'):
                      print(f"   - 樹的數量: {model.n_estimators}")
                  if hasattr(model, 'n_features_in_'):
                      print(f"   - 特徵數量: {model.n_features_in_}")
              
          except Exception as e:
              print(f"❌ 載入 {model_file} 時出錯: {e}")
      else:
          print(f"❌ 找不到模型文件: {model_file}")
  
  return models

def prepare_test_data():
  """準備測試數據"""
  print("📊 準備測試數據...")
  
  # 讀取訓練資訊
  info = pd.read_csv('/Users/shuyuhsu/code_workspace/AICUP2025_tableTennis/AICUP_data/Training_Dataset/train_info.csv')
  unique_players = info['player_id'].unique()
  
  # 使用相同的隨機種子分割數據
  train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)
  
  # 讀取特徵數據
  datapath = '/Users/shuyuhsu/code_workspace/AICUP2025_tableTennis/AICUP_data/tabular_data_train'
  datalist = list(Path(datapath).glob('**/*.csv'))
  target_mask = ['gender', 'hold racket handed', 'play years', 'level']
  
  # 準備測試數據
  x_test = pd.DataFrame()
  y_test = pd.DataFrame(columns=target_mask)
  
  for file in datalist:
      unique_id = int(Path(file).stem)
      row = info[info['unique_id'] == unique_id]
      if row.empty:
          continue
      player_id = row['player_id'].iloc[0]
      
      # 只使用測試玩家的數據
      if player_id in test_players:
          data = pd.read_csv(file)
          target = row[target_mask]
          target_repeated = pd.concat([target] * len(data))
          x_test = pd.concat([x_test, data], ignore_index=True)
          y_test = pd.concat([y_test, target_repeated], ignore_index=True)
  
  print(f"   - 測試集大小: {x_test.shape}")
  print(f"   - 測試玩家數量: {len(test_players)}")
  
  return x_test, y_test

def model_binary_evaluation(model_data, X_test, y_test, target_name, group_size=27):
  """使用 AICUP 方法評估二元分類模型"""
  print(f"\n🔍 評估二元分類: {target_name}")
  
  model = model_data['model']
  label_encoder = model_data['label_encoder']
  scaler = model_data['scaler']
  
  # 標準化測試數據
  X_test_scaled = scaler.transform(X_test)
  
  # 編碼標籤
  y_test_encoded = label_encoder.transform(y_test)
  
  # 預測機率
  predicted_proba = model.predict_proba(X_test_scaled)
  predicted_labels = model.predict(X_test_scaled)
  
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
      print(f"   ⚠️  測試數據不足以形成完整組別")
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

def model_multiary_evaluation(model_data, X_test, y_test, target_name, group_size=27):
  """使用 AICUP 方法評估多類別分類模型"""
  print(f"\n🔍 評估多類別分類: {target_name}")
  
  model = model_data['model']
  label_encoder = model_data['label_encoder']
  scaler = model_data['scaler']
  
  # 標準化測試數據
  X_test_scaled = scaler.transform(X_test)
  
  # 編碼標籤
  y_test_encoded = label_encoder.transform(y_test)
  
  # 預測機率和標籤
  predicted_proba = model.predict_proba(X_test_scaled)
  predicted_labels = model.predict(X_test_scaled)
  
  # 計算基本準確率
  accuracy = accuracy_score(y_test_encoded, predicted_labels)
  print(f"   - 基本準確率: {accuracy:.4f}")
  print(f"   - 類別數量: {len(label_encoder.classes_)}")
  print(f"   - 類別: {label_encoder.classes_}")
  
  # 按組聚合預測結果 (AICUP 方法)
  num_groups = len(predicted_proba) // group_size
  if num_groups == 0:
      print(f"   ⚠️  測試數據不足以形成完整組別")
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

def evaluate_all_models():
  """評估所有模型"""
  print("🚀 開始評估所有模型...")
  
  # 模型文件列表
  model_files = [
      './models/gender_model.pkl',
      './models/hold racket handed_model.pkl', 
      './models/play years_model.pkl',
      './models/level_model.pkl'
  ]
  
  # 載入模型
  models = load_trained_models(model_files)
  
  if not models:
      print("❌ 沒有成功載入任何模型")
      return
  
  # 準備測試數據
  try:
      x_test, y_test = prepare_test_data()
  except Exception as e:
      print(f"❌ 準備測試數據時出錯: {e}")
      return
  
  # 評估結果存儲
  results = {}
  
  # 定義分類類型
  binary_targets = ['gender', 'hold racket handed']
  multi_targets = ['play years', 'level']
  
  # 評估每個模型
  for target_name, model_data in models.items():
      try:
          if target_name in binary_targets:
              accuracy, auc = model_binary_evaluation(
                  model_data, x_test, y_test[target_name], target_name
              )
          elif target_name in multi_targets:
              accuracy, auc = model_multiary_evaluation(
                  model_data, x_test, y_test[target_name], target_name
              )
          else:
              print(f"⚠️  未知的目標類型: {target_name}")
              continue
              
          results[target_name] = {
              'accuracy': accuracy,
              'auc': auc,
              'type': 'binary' if target_name in binary_targets else 'multi'
          }
          
      except Exception as e:
          print(f"❌ 評估 {target_name} 時出錯: {e}")
          import traceback
          traceback.print_exc()
  
  # 顯示總結
  print("\n" + "="*60)
  print("📊 模型評估結果總結")
  print("="*60)
  
  total_auc = 0
  valid_models = 0
  
  for target, result in results.items():
      model_type = result['type']
      accuracy = result['accuracy']
      auc = result['auc']
      
      print(f"{target:20s} | 類型: {model_type:6s} | 準確率: {accuracy:.4f} | AUC: {auc:.4f}")
      
      if auc > 0:
          total_auc += auc
          valid_models += 1
  
  if valid_models > 0:
      avg_auc = total_auc / valid_models
      print(f"{'':20s} | {'':6s} {'':2s} | {'平均 AUC':8s}: {avg_auc:.4f}")
      print(f"{'':20s} | {'':6s} {'':2s} | {'總分':8s}: {total_auc:.4f}")
  
  print("="*60)
  
  return results

def main():
  """主函數"""
  try:
      results = evaluate_all_models()
      return results
  except Exception as e:
      print(f"❌ 程式執行時發生錯誤: {e}")
      import traceback
      traceback.print_exc()
      return None

if __name__ == "__main__":
  results = main()