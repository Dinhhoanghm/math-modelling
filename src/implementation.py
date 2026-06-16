import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
N = 150               # Number of vehicles
L = 500               # Length of the road
T = 100             # Number of time steps
dt = 0.1              # Time step size

# FVD parameters
k = 1.0               # Sensitivity to headway
lmbd = 0.5            # Sensitivity to relative velocity

# Night driving OV function parameters
xc = 2.0
xc1 = 3.2
xc2 = 4.0
a = 5.0
b = 1.0

# Perturbation
n_dec = 1             # Number of deceleration steps
perturb_vehicle = 0   # Vehicle index to perturb
a_dec = -1.0          # Constant deceleration

# --- Optimal velocity function (night) ---
def V(dx):
    if dx < xc1:
        return np.tanh(dx - xc) + np.tanh(xc)
    elif dx < xc2:
        return a - dx
    else:
        return b

# Vectorized OV function
V_vec = np.vectorize(V)

# --- Initialization ---
x = np.linspace(0, L - L/N, N)  # Initial positions, equally spaced
v = np.full(N, V(L/N))          # Initial velocities

# Storage for space-time plot
x_hist = np.zeros((T, N))

# Deceleration flag for perturbation
decelerating = True

for t in range(T):
    x_hist[t] = x.copy()

    dv = np.zeros(N)

    for i in range(N):
        lead = (i - 1) % N
        dx = (x[lead] - x[i]) % L

        dv[i] = k * (V(dx) - v[i]) + lmbd * (v[lead] - v[i])

    # Apply perturbation
    if t < n_dec:
        v[perturb_vehicle] += a_dec * dt
        v[perturb_vehicle] = max(v[perturb_vehicle], 0.0)
    else:
        v += dv * dt

    # Position update
    x += v * dt + 0.5 * dv * dt**2
    x %= L  # Periodic boundary

# --- Visualization ---
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(range(T), x_hist[:, i], lw=0.5)
plt.xlabel('Time step')
plt.ylabel('Position on road')
plt.title('Space-Time Plot of Vehicles (Night Driving)')
plt.grid(True)
plt.show()
