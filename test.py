import numpy as np

coords = [[0,   1.0, 0, 0, 0],       # 0
         [0,   1.0, 0, 0, 0],        # 0
         [0,   9.0, 0, 0, 0],   # 3    -1   2
         
         [1,   1.0, 1, 1, 1],        # 0         3
         [1,   1.0, 1, 1, 1],        # 0
         [1,   1.0, 1, 1, 1],        # 0
         [1,   9.0, 1, 1, 1],   # 4    -1   6
         
         [2,   1.0, 2, 2, 2],        # 0         7
         [2,   1.0, 2, 2, 2],        # 0
         [2,   1.0, 2, 2, 2],        # 0
         [2,   1.0, 2, 2, 2],        # 0
         [2,   9.0, 2, 2, 2],]  # 5

coords = np.array(coords)

mol_id = coords[:, 0].astype(int)
conformers = coords[:, 1:]
# Get ids corresponding to new molecules
split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
print(split_indices)     # [3, 7]
data_list = np.split(conformers, split_indices)


perm = np.random.permutation(len(data_list)).astype('int32')
print(perm)
data_list = [data_list[i] for i in perm]
[print(d) for d in data_list]


# print(data_list)