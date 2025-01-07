class InvalidInputError(Exception):
    """Exception raised if all input (i.e. PDB) files contain errors."""
    
    def __init__(self, message="Error reading all input files."):
        self.message = message
        super().__init__(self.message)


class ModelInitialisationError(Exception):
    """Exception raised for model initialisation error."""
    
    def __init__(self, message="Error during model initialisation."):
        self.message = message
        super().__init__(self.message)


class ModelGenerationError(Exception):
    """Exception raised for model generation error."""
    
    def __init__(self, message="Model error during generation."):
        self.message = message
        super().__init__(self.message)


class MetricError(Exception):
    """Exception raised if errors occur during metrics computation & docking analysis"""
    
    def __init__(self, message="Error during metrics computation / docking analysis."):
        self.message = message
        super().__init__(self.message)