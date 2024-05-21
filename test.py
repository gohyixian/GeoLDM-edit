import torch

# Create a tensor of shape [999, 512]
tensor = torch.randn(999, 512)

# Define the chunk size
chunk_size = 100

# Split the tensor into chunks
chunks = list(torch.split(tensor, chunk_size, dim=0))
print(len(chunks))

for i, c in enumerate(chunks):
    print(i, c.shape)
    tmp = torch.randn(c.shape)
    print(tmp == c)
    chunks[i] = tmp

# Combine the chunks back into a single tensor
combined_tensor = torch.cat(chunks, dim=0)

# Verify the shape of the combined tensor
print(f"Combined tensor shape: {combined_tensor.shape}")
print(combined_tensor == tensor)
