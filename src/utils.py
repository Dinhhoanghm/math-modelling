import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from config import Config
from fvd_simulation import FVDSimulation


def compute_flow(N_values, kappa=1.0, lambda_=0.2, L=500, T=15000, dt=0.1, n_dec=1, save_file='flow_data.npz',
                 skip_if_exists=True):
    """Compute flow for different density values and save results.

    If skip_if_exists=True and save_file already exists, loads and returns cached data.
    """
    if skip_if_exists and os.path.exists(save_file):
        print(f"Loading cached data from {save_file} (pass skip_if_exists=False to recompute)")
        data = np.load(save_file)
        return (data['density_values'], data['normal_flows'],
                data['night_flows'], data['night_perturbed_flows'])

    normal_flows = []
    night_flows = []
    night_perturbed_flows = []
    density_values = []

    for N in tqdm.tqdm(N_values):
        config = Config(N=N, lambda_=lambda_, kappa=kappa, T=T, dt=dt, L=L, n_dec=n_dec)
        normal_config = Config(N=N, lambda_=lambda_, kappa=kappa, T=5000, dt=dt, L=L, n_dec=n_dec)
        density = N / L
        density_values.append(density)

        # Normal driving (no perturbation)
        sim_normal = FVDSimulation(normal_config, night_mode=False)
        _, velocities = sim_normal.run(perturb=False)
        avg_velocity = np.mean(velocities[-1000:])  # Average over last 1000 steps
        normal_flows.append(density * avg_velocity)

        # Night driving (no perturbation)
        sim_night = FVDSimulation(normal_config, night_mode=True)
        _, velocities = sim_night.run(perturb=False)
        avg_velocity = np.mean(velocities[-1000:])
        night_flows.append(density * avg_velocity)

        # Night driving with small perturbation
        sim_night_perturbed = FVDSimulation(config, night_mode=True)
        _, velocities = sim_night_perturbed.run(perturb=True)
        avg_velocity = np.mean(velocities[-1000:])
        night_perturbed_flows.append(density * avg_velocity)

    # Save arrays to a .npz file
    np.savez(save_file, 
             density_values=density_values, 
             normal_flows=normal_flows, 
             night_flows=night_flows, 
             night_perturbed_flows=night_perturbed_flows)
    print(f"Data saved to {save_file}")

    return density_values, normal_flows, night_flows, night_perturbed_flows

def compute_critical_densities(kappa, lambda_, xc=2.0, xc1=3.2, xc2=4.0):
    """
    Compute analytically-derived critical densities for the fundamental diagram.

    k_c2, k_c1: boundaries of the V'<0 region (from x_c2, x_c1).
    k_low, k_high: crossings where V'(Δx) = κ/2 + λ (FVD stability boundary).

    Returns list of (density, label) tuples, sorted by density ascending.
    """
    kc2 = 1.0 / xc2   # lower density boundary of V'<0 region (≈ 0.25)
    kc1 = 1.0 / xc1   # upper density boundary of V'<0 region (≈ 0.3125)

    threshold = kappa / 2.0 + lambda_
    vlines = [(kc2, r'$k_{c2}$'), (kc1, r'$k_{c1}$')]

    # sech²(0) = 1; threshold must be < 1 for crossings to exist
    if threshold < 1.0:
        u = np.arccosh(1.0 / np.sqrt(threshold))
        k_low  = 1.0 / (xc + u)   # low-density stability boundary (k_c3 for λ=0.2)
        k_high = 1.0 / (xc - u)   # high-density stability boundary (k_c4 for λ=0.2)
        vlines += [(k_low, r'$k_{c3}$'), (k_high, r'$k_{c4}$')]

    return sorted(vlines, key=lambda pair: pair[0])


def load_and_plot_fundamental_diagram(load_file='flow_data.npz', output_plot='fundamental_diagram.png',
                                      title='Fundamental Diagram (κ=1.0, λ=0.1)', figsize=(8, 6),
                                      vlines=None):
    """
    Load saved flow data from a .npz file and plot the fundamental diagram.

    Parameters:
    - load_file (str): Path to the .npz file containing saved data.
    - output_plot (str): Path to save the generated plot.
    - title (str): Title of the plot.
    - figsize (tuple): Figure size as (width, height) in inches.
    - vlines (list): Optional list of (density, label) tuples for vertical reference lines.
    """
    try:
        data = np.load(load_file)
        density_values = data['density_values']
        normal_flows = data['normal_flows']
        night_flows = data['night_flows']
        night_perturbed_flows = data['night_perturbed_flows']
        print(f"Data loaded from {load_file}")
    except FileNotFoundError:
        print(f"Error: File {load_file} not found.")
        return
    except KeyError as e:
        print(f"Error: Missing key {e} in {load_file}. Ensure all arrays are saved correctly.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(density_values, normal_flows, 'k-', label='Normal driving')
    ax.plot(density_values, night_flows, 'k--', label='Night driving')
    ax.plot(density_values, night_perturbed_flows, 'k:', label='Night driving (perturbed)')

    if vlines:
        y_min, y_max = ax.get_ylim()
        for density, label in vlines:
            ax.axvline(x=density, color='k', linestyle=':')
            ax.text(x=density, y=y_min + 0.02 * (y_max - y_min),
                    s=label, color='k', fontsize=9, ha='center', va='bottom')

    ax.set_xlabel('density')
    ax.set_ylabel('flow')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    try:
        plt.show()
    except Exception:
        pass

def plot_traffic_pattern(position_filename, output_name, N=150, kappa=1.0, lambda_=0.2, L=500, dt=0.1):
    """
    Plot periodic vehicle trajectories in a space-time diagram.
    X-axis: Position (wrapped)
    Y-axis: Time
    
    Parameters:
    - position_filename (str): Path to .npy file with position array of shape (T, N).
    - output_name (str): Path to save the plot.
    - N (int): Number of vehicles (default: 150).
    - kappa (float): Sensitivity parameter κ (default: 1.0).
    - lambda_ (float): Sensitivity parameter λ (default: 0.2).
    - L (float): System length (default: 500).
    - dt (float): Time step (default: 0.1).
    """
    # Load position array
    try:
        positions = np.load(position_filename)
        print(f"Loaded positions from {position_filename}, shape: {positions.shape}")
    except FileNotFoundError:
        print(f"Error: File {position_filename} not found.")
        return

    T, N_actual = positions.shape
    if N_actual != N:
        print(f"Warning: File contains {N_actual} vehicles, overriding N={N}")
        N = N_actual

    # Prepare time array
    time = np.arange(T) * dt

    # Plot setup
    plt.figure(figsize=(10, 6))

    for i in range(N):
        traj = positions[:, i].copy()
        delta = np.diff(traj)
        jumps = np.abs(delta) > L / 2
        traj_plot = traj.astype(float)
        traj_plot[1:][jumps] = np.nan
        # print(traj_plot.shape)
        plt.plot(traj_plot[-1000:], np.arange(T)[-1000:], 'k-', linewidth=0.5)
        # plt.plot(traj_plot, np.arange(T), 'k-', linewidth=0.5)
    plt.ylabel('Time (s)')
    plt.xlabel('Position (units)')
    plt.title(f'Space-Time Plot (N={N}, κ={kappa}, λ={lambda_})')
    plt.tight_layout()
    plt.savefig(output_name)
    print(f"Plot saved to {output_name}")
    try:
        plt.show()
    except:
        pass