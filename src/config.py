import numpy as np 
class Config:
    """Configuration class to hold simulation parameters."""
    def __init__(self, **kwargs):
        # Default parameters
        self.N = 150  # Number of vehicles
        self.L = 500  # System length
        self.kappa = 1.0  # Sensitivity parameter
        self.lambda_ = 0.5  # Sensitivity parameter
        self.dt = 0.1  # Time step
        self.T = 2000  # Total time steps
        self.xc = 2.0
        self.xc1 = 3.2
        self.xc2 = 4.0
        self.a = 5.0
        self.b = 1.0
        self.n_dec = 1  # Number of perturbation steps
        self.perturbation_vehicle = 0  # Index of vehicle to perturb
        self.output_file = 'vehicle_positions.npy'  # Output file name
        self.stochastic = False
        self.A = 0
        # Override defaults with provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")