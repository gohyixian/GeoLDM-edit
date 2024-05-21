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
