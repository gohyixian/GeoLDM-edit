import torch
import torch.nn as nn
import torch.optim as optim

class submodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(20, 20)
    
    def forward(self, x):
        x = self.model(x)
        return x

class Model(nn.Module):
    def __init__(self, submodel:submodel):
        super(Model, self).__init__()
        self.layerA = nn.Linear(10, 20)
        self.layersub = submodel.model
        self.layerB = nn.Linear(20, 30)
        self.layerC = nn.Linear(30, 40)
    
    def forward(self, x):
        x = self.layerA(x)
        x = self.layersub(x)
        x = self.layerB(x)
        x = self.layerC(x)
        return x

# Initialize the model
sub = submodel()
model = Model(sub)

# Set requires_grad=False for layers B and C
for param in model.layerB.parameters():
    param.requires_grad = False
for param in model.layerC.parameters():
    param.requires_grad = False

# Define a loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Dummy input and target tensors
input_tensor = torch.randn(5, 10)
target_tensor = torch.randn(5, 40)

# Forward pass
output = model(input_tensor)

# Compute loss
loss = criterion(output, target_tensor)

# Backward pass
loss.backward()

# Check gradients
print(model.layerA.weight.grad)  # Should have gradients
print(model.layersub.weight.grad)  # Should have gradients
print(model.layerB.weight.grad)  # Should be None
print(model.layerC.weight.grad)  # Should be None
