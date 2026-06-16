import numpy as np
import tqdm
import random
from config import Config

def get_rand():
    return random.uniform(-0.5, 0.5)

class FVDSimulation:
    """Class to run the Full Velocity Difference (FVD) simulation."""
    def __init__(self, config, night_mode = True):
        self.config = config
        self.positions = None
        self.velocities = None
        self.x = None
        self.v = None
        self.night_mode = night_mode

        self.v_max = self.V_night(3.2)

    def V_normal(self, delta_x):
        """Normal optimal velocity function. (5)"""
        return np.tanh(delta_x - self.config.xc) + np.tanh(self.config.xc)

    def V_night(self, delta_x):
        """Night-time optimal velocity function. (6)"""
        if delta_x < self.config.xc1:
            return np.tanh(delta_x - self.config.xc) + np.tanh(self.config.xc)
        elif self.config.xc1 <= delta_x <= self.config.xc2:
            return self.config.a - delta_x
        else:
            return self.config.b

    def V(self, delta_x):
        """Select optimal velocity function based on mode."""
        return self.V_night(delta_x) if self.night_mode else self.V_normal(delta_x)

    def initialize(self, perturb=True):
        """Initialize vehicle positions and velocities."""
        density = self.config.N / self.config.L
        headway = self.config.L / self.config.N
        self.x = np.arange(self.config.N) * headway  # Uniform spacing
        self.v = np.ones(self.config.N) * self.V(headway)  # Initial velocity
        self.velocities = np.zeros((self.config.T, self.config.N))  # Initialize velocities array
        self.positions = np.zeros((self.config.T, self.config.N))
        self.positions[0] = self.x.copy()
        self.velocities[0] = self.v.copy()
        # Apply perturbation
        # if perturb:
        #     self.v[self.config.perturbation_vehicle] += -1.0 * self.config.dt * self.config.n_dec
        #     if self.v[self.config.perturbation_vehicle] < 0:
        #         self.v[self.config.perturbation_vehicle] = 0

    def run(self, perturb = True):
        """Run the simulation loop."""
        self.initialize(perturb)
        
        for t in range(1, self.config.T):
            x_old = self.x.copy()
            v_old = self.v.copy()
            
            # Compute acceleration for each vehicle
            for i in range(self.config.N):
                if perturb and t <= self.config.n_dec and i == self.config.perturbation_vehicle:
                    if v_old[i] <= 0:
                        self.v[i] = 0
                        self.x[i] = x_old[i]
                        continue
                    a = -1
                    v_tmp = v_old[i] - self.config.dt
                    self.v[i] = max(v_tmp, 0)
                    self.x[i] = x_old[i] + v_old[i] * self.config.dt + 0.5 * a * self.config.dt ** 2
                    self.x[i] = self.x[i] % self.config.L
                    continue
                # Periodic boundary conditions
                next_i = (i + 1) % self.config.N
                delta_x = (x_old[next_i] - x_old[i]) % self.config.L
                if delta_x < 0:
                    delta_x += self.config.L  # Ensure positive headway
                dv_dt = (self.config.kappa * (self.V(delta_x) - v_old[i]) +
                         self.config.lambda_ * (v_old[next_i] - v_old[i]))
                
                # Update velocity
                # (7) -> (9)
                if self.config.stochastic:
                    # Eq (7): Compute unconstrained intermediate velocity v*(t + dt)
                    v_star = v_old[i] + dv_dt * self.config.dt + get_rand() * self.config.A
                    # Eq (8): Clip velocity within safe bounds [0, v_max]
                    self.v[i] = min(max(0.0, v_star), self.v_max)
                    # Eq (9): Position update using trapezoidal average velocity rule
                    self.x[i] = x_old[i] + 0.5 * (v_old[i] + self.v[i]) * self.config.dt
                else:
                    # Eq (3): Standard Euler velocity step
                    self.v[i] = max(0.0, v_old[i] + dv_dt * self.config.dt)
                    # Eq (4): Position update via explicit second-order Taylor expansion
                    self.x[i] = x_old[i] + v_old[i] * self.config.dt + 0.5 * dv_dt * (self.config.dt ** 2)
                self.x[i] = self.x[i] % self.config.L  # Apply periodic boundary conditions
            
            self.positions[t] = self.x.copy()
            self.velocities[t] = self.v.copy()
        np.save(self.config.output_file, self.positions)
        return self.positions, self.velocities

if __name__ == "__main__":
    # Example usage with default parameters
    config = Config(N = 230, lambda_ = 0.1, kappa = 1.0, T = 5000, output_file='figure7_positions.npy')
    # config = Config()
    sim = FVDSimulation(config)
    sim.run()