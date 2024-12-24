import torch
import torch.nn as nn
import torch.optim as optim

def zero_module(module: nn.Module):
    """
    :module: i.e. nn.Module
    Zero out the parameters of a module and return it.
    """
    for name, param in module.named_parameters():
        if 'bias' not in name:
            print(name)
            param.data.zero_()
    return module

# Define a simple model with two linear layers and a SiLU activation function
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layerA = nn.Linear(10, 20)
        self.act = nn.SiLU()
        self.layerA1 = nn.Linear(20, 20)
        self.layerA2 = nn.Linear(20, 20)
        self.layerA3 = nn.Linear(20, 20)
        # self.act = nn.Sigmoid()
        # self.act = nn.ReLU()
        # self.act = nn.Tanh()
        # self.act = nn.LeakyReLU()
        self.layerB = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.layerA(x)
        x = self.act(x)
        x = self.layerA1(x)
        x = self.act(x)
        x = self.layerA2(x)
        x = self.act(x)
        x = self.layerA3(x)
        x = self.act(x)
        x = self.layerB(x)
        return x

# Initialize the model
model = SimpleModel()
model = zero_module(model)

# Define a loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

ITERATIONS = 1000
CHECK_ITERATION = 999
for i in range(ITERATIONS):
    # Dummy input and target tensors
    input_tensor = torch.randn(5, 10)
    target_tensor = torch.randn(5, 5)

    # Forward pass
    output = model(input_tensor)

    # Compute loss
    loss = criterion(output, target_tensor)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Print gradients for all parameters that have requires_grad=True
    if i == CHECK_ITERATION:
        # for name, param in model.named_parameters():
        #     print(f"Gradients for {name}:")
        #     print(param.grad)
        
        print('\n\n\n\n')
        for name, param in model.named_parameters():
            print(name)
            print(param)

    optimizer.step()