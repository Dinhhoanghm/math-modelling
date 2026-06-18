import numpy as np
from config import Config


class FVDSimulation:
    """Full Velocity Difference (FVD) car-following simulation, vectorized with numpy.

    Implements Jiang & Wu, Physica A 375 (2007) 297-306:
      dv_{n+1}/dt = kappa*(V(dx) - v_{n+1}) + lambda*(v_n - v_{n+1})    (Eq. 1)
    integrated by the explicit Euler scheme (Eqs. 3-4), or by the stochastic
    scheme (Eqs. 7-9) when config.stochastic is set. Periodic boundary conditions.
    """

    def __init__(self, config, night_mode=True, seed=0):
        self.config = config
        self.night_mode = night_mode
        self.seed = seed
        self.positions = None
        self.velocities = None
        self.x = None
        self.v = None
        # Maximum velocity used by the stochastic clip (Eq. 8): v_max = V(3.2).
        self.v_max = self.V_night(3.2)

    # ---- scalar optimal-velocity functions (used for v_max / initial state) ----
    def V_normal(self, delta_x):
        """Normal optimal velocity function (Eq. 5)."""
        return np.tanh(delta_x - self.config.xc) + np.tanh(self.config.xc)

    def V_night(self, delta_x):
        """Night-time optimal velocity function (Eq. 6)."""
        if delta_x < self.config.xc1:
            return np.tanh(delta_x - self.config.xc) + np.tanh(self.config.xc)
        elif self.config.xc1 <= delta_x <= self.config.xc2:
            return self.config.a - delta_x
        else:
            return self.config.b

    def V(self, delta_x):
        """Select optimal velocity function based on mode (scalar)."""
        return self.V_night(delta_x) if self.night_mode else self.V_normal(delta_x)

    # ---- vectorized optimal-velocity function (used inside the time loop) ----
    def _V_vec(self, delta_x):
        """Vectorized optimal velocity for an array of headways.

        Boundary handling matches the scalar V_night exactly: at dx==xc1 and
        dx==xc2 the piecewise value is a-dx.
        """
        xc, xc1, xc2 = self.config.xc, self.config.xc1, self.config.xc2
        a, b = self.config.a, self.config.b
        tanh_branch = np.tanh(delta_x - xc) + np.tanh(xc)
        if not self.night_mode:
            return tanh_branch
        return np.where(
            delta_x < xc1,
            tanh_branch,
            np.where(delta_x <= xc2, a - delta_x, b),
        )

    def initialize(self):
        """Initialize uniformly spaced vehicles at the homogeneous velocity V(headway)."""
        headway = self.config.L / self.config.N
        self.x = np.arange(self.config.N, dtype=float) * headway
        self.v = np.ones(self.config.N) * self.V(headway)

    def run(self, perturb=True, save=True, ref10=True, store_steps=2000):
        """Run the simulation.

        Parameters
        - perturb (bool): apply the single-vehicle deceleration perturbation.
        - save (bool): write the position tail to config.output_file (disable for
          fundamental-diagram sweeps where the trajectory is not needed).
        - ref10 (bool): apply the Ref. [10] braking formula x += v^2/(2|a|) when a
          decelerating vehicle would otherwise cross zero velocity. Set False only
          to reproduce the legacy scalar behaviour for validation.
        - store_steps (int|None): how many trailing time steps to persist to disk
          (None = all). The full arrays are always returned in memory.
        """
        self.initialize()
        cfg = self.config
        N, L, dt, T = cfg.N, cfg.L, cfg.dt, cfg.T
        kappa, lambda_ = cfg.kappa, cfg.lambda_
        pv = cfg.perturbation_vehicle
        rng = np.random.default_rng(self.seed)

        positions = np.zeros((T, N))
        velocities = np.zeros((T, N))
        positions[0] = self.x
        velocities[0] = self.v

        x_old = self.x
        v_old = self.v
        for t in range(1, T):
            delta_x = (np.roll(x_old, -1) - x_old) % L          # leader is i+1
            v_lead = np.roll(v_old, -1)
            dv_dt = kappa * (self._V_vec(delta_x) - v_old) + lambda_ * (v_lead - v_old)

            if cfg.stochastic:
                # Eqs. (7)-(9): noisy velocity, clip to [0, v_max], trapezoidal position
                v_star = v_old + dv_dt * dt + rng.uniform(-0.5, 0.5, N) * cfg.A
                v_new = np.minimum(np.maximum(0.0, v_star), self.v_max)
                x_new = (x_old + 0.5 * (v_old + v_new) * dt) % L
            else:
                # Eqs. (3)-(4): explicit Euler velocity + second-order position
                v_new = np.maximum(0.0, v_old + dv_dt * dt)
                x_new = (x_old + v_old * dt + 0.5 * dv_dt * dt ** 2) % L

            # Single-vehicle deceleration perturbation for the first n_dec steps.
            if perturb and t <= cfg.n_dec:
                vo = v_old[pv]
                if vo <= 0.0:
                    # Already stopped: remain at rest (Eq. text: stays for n_dec - m_dec).
                    v_new[pv] = 0.0
                    x_new[pv] = x_old[pv]
                elif ref10 and vo < dt:
                    # Ref. [10]: decelerating at a=-1 would give v<0, so stop and
                    # advance by the braking distance v^2/(2|a|), with |a|=1.
                    v_new[pv] = 0.0
                    x_new[pv] = (x_old[pv] + vo * vo / 2.0) % L
                else:
                    # Constant deceleration a = -1 (a*dt = -dt since |a|=1).
                    v_new[pv] = max(vo - dt, 0.0)
                    x_new[pv] = (x_old[pv] + vo * dt - 0.5 * dt ** 2) % L

            positions[t] = x_new
            velocities[t] = v_new
            x_old = x_new
            v_old = v_new

        self.x = x_old
        self.v = v_old
        self.positions = positions
        self.velocities = velocities

        if save:
            tail = positions if (store_steps is None or T <= store_steps) else positions[-store_steps:]
            np.save(cfg.output_file, tail)
        return positions, velocities


if __name__ == "__main__":
    config = Config(N=230, lambda_=0.1, kappa=1.0, T=5000, output_file='figure7_positions.npy')
    sim = FVDSimulation(config)
    sim.run()
