import numpy as np

# Example 3D NumPy array
array = np.random.rand(3, 4, 5, 6)  # Shape (3, 4, 5)

# Convert the 3D array to a string
array_str = np.array2string(array, separator=',', formatter={'float_kind':lambda x: f'{x:.5f}'})

# Write to a text file
with open('array.txt', 'w') as f:
    f.write(array_str)
