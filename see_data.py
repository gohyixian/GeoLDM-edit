import numpy as np

data_file = '/Users/gohyixian/Documents/GitHub/FYP/GeoLDM-edit/qm9/temp/qm9/test.npz'
f = np.load(data_file)
print(f.keys())

f.close()