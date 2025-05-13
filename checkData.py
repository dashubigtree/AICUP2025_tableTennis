from pathlib import Path

folder = Path("/Users/shuyuhsu/code_workspace/AICUP2025_tableTennis/AICUP_data/train_data")
all_files = [f.stem for f in folder.glob("*.txt")]
all_nums = sorted([int(f) for f in all_files if f.isdigit()])
missing = [i for i in range(all_nums[0], all_nums[-1]+1) if i not in all_nums]
print("缺少的編號：", missing)