import numpy as np

df = np.load("qm9/temp/qm9/train.npz")

print(df['positions'][0].shape)