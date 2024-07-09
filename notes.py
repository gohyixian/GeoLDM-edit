def zero_module(module):
    """
    :module: i.e. nn.Conv2d(), nn.linear()
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def low_vram_shift(self, is_diffusing):
    if is_diffusing:
        self.model = self.model.cuda()
        self.control_model = self.control_model.cuda()
        self.first_stage_model = self.first_stage_model.cpu()
        self.cond_stage_model = self.cond_stage_model.cpu()
    else:
        self.model = self.model.cpu()
        self.control_model = self.control_model.cpu()
        self.first_stage_model = self.first_stage_model.cuda()
        self.cond_stage_model = self.cond_stage_model.cuda()


from egnn.egnn_fusion import EGNN_Fusion
# Initialize the EGNN_Fusion model
model = EGNN_Fusion(in_node_nf=10, in_edge_nf=2, hidden_nf=256)

# Zero out all weights and biases
for param in model.parameters():
    param.data.zero_()

# Verify that all parameters are zero
all_zero = True
for name, param in model.named_parameters():
    print(f"Name: {name}, Sum of parameter values: {param.shape} {param.sum().item()} ")
    # print(f"Name: {name}, Sum of parameter values: {param} ")
    if param.sum().item() != 0.0:
        all_zero=False
print(all_zero)  # True
