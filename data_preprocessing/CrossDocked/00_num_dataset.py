import torch

split_path = "/Users/gohyixian/Downloads/Crossdocked_Pocket10/split_by_name.pt"
data_split = torch.load(split_path)

print(f"Train: {len(data_split['train'])}")
print(f"Test : {len(data_split['test'])}")

# (geoldm) (base) 🎃 gohyixian CrossDocked % python 00_num_dataset.py
# Train: 100,000
# Test : 100