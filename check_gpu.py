import torch

def check_gpu_availability(gpu_ids):
    for gpu_id in gpu_ids:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            try:
                torch.zeros(1).to(device)
                print(f"GPU {gpu_id} is available.")
            except:
                print(f"GPU {gpu_id} is not available.")
        else:
            print(f"GPU {gpu_id} is not available.")

# Specify the GPU IDs to check
gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Check GPU availability
check_gpu_availability(gpu_ids)