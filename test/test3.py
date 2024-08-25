import os
from tqdm import tqdm

crossdocked = '/Users/gohyixian/Documents/Documents/3.2_FYP_1/data/CrossDocked/crossdocked_pocket10'

to_remove = ['.DS_Store', 'index.pkl']

all_complexes = []

all_folders = sorted([f for f in os.listdir(crossdocked) if f not in to_remove])

for f in tqdm(all_folders):
    cur = os.path.join(crossdocked, f)
    files = sorted([i for i in os.listdir(cur) if i not in to_remove])
    for i in files:
        split = i.split("_")[0]
        all_complexes.append(split)

all_complexes = sorted(list(set(all_complexes)))
print(all_complexes)

with open('/Users/gohyixian/Downloads/CrossDocked_stats.txt', 'w') as f:
    for i in all_complexes:
        print(i, file=f)


bindingmoad = '/Users/gohyixian/Documents/Documents/3.2_FYP_1/data/BindingMOAD/BindingMOAD_2020'

all_files = sorted([i.split(".")[0] for i in os.listdir(bindingmoad) if i not in to_remove])
all_files = sorted(list(set(all_files)))

with open('/Users/gohyixian/Downloads/BindingMOAD_stats.txt', 'w') as f:
    for i in all_files:
        print(i, file=f)

intersection = set(all_complexes) & set(all_files)
intersection = sorted(list(intersection))
print(intersection)
with open('/Users/gohyixian/Downloads/intersection_stats.txt', 'w') as f:
    for i in intersection:
        print(i, file=f)

print(f"len(crossdock):    {len(all_complexes)}")
print(f"len(bindingmoad):  {len(all_files)}")
print(f"len(intersection): {len(intersection)}")
