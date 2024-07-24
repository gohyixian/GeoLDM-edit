import re

# Sample nvidia-smi output
output = """
+-----------------------------------------------------------------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   35C    P0    36W / 300W |    774MiB / 32768MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-SXM2...  Off  | 00000000:00:1F.0 Off |                    0 |
| N/A   40C    P0    45W / 300W |    1024MiB / 32768MiB|      5%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
"""

# Regular expressions to capture GPU ID and memory usage
gpu_id_pattern = r'^\|\s*(\d+)\s+'
memory_pattern = r'(\d+)MiB\s*/\s*(\d+)MiB'

# Dictionary to hold memory usage by GPU ID
gpu_memory_usage = {}

# Find all GPU IDs and memory usage
gpu_ids = re.findall(gpu_id_pattern, output, re.MULTILINE)
memories = re.findall(memory_pattern, output)

# Assuming GPU IDs and memories are in the same order
for gpu_id, (used_memory, total_memory) in zip(gpu_ids, memories):
    gpu_memory_usage[gpu_id] = {
        'Used Memory': used_memory,
        'Total Memory': total_memory
    }

# Print results
for gpu_id, memory_info in gpu_memory_usage.items():
    print(f"GPU {gpu_id}:")
    print(f"  Used Memory: {memory_info['Used Memory']} MiB")
    print(f"  Total Memory: {memory_info['Total Memory']} MiB")
