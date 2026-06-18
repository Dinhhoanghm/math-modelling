import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from config import Config
from fvd_simulation import FVDSimulation


def homogeneous_flow(k_values, night_mode, xc=2.0, xc1=3.2, xc2=4.0, a=5.0, b=1.0):
    """Analytical homogeneous fundamental flow q(k) = k * V(1/k).

    The no-perturbation branches are deterministic equilibria of the FVD model, so
    they equal this closed form exactly. Computing them analytically (rather than by
    simulation) gives sharp piecewise bends that fall *exactly* on the critical
    densities k_c1=1/xc1 and k_c2=1/xc2, and recovers the true steep night segment
    q=a*k-1 on k_c2<k<k_c1 — which a simulation cannot hold because that branch is
    linearly unstable (V'<0) and clusters under any numerical noise.
    """
    k = np.asarray(k_values, dtype=float)
    headway = np.divide(1.0, k, out=np.full_like(k, np.inf), where=k > 0)
    tanh_branch = np.tanh(headway - xc) + np.tanh(xc)
    if not night_mode:
        V = tanh_branch
    else:
        V = np.where(headway < xc1, tanh_branch,
                     np.where(headway <= xc2, a - headway, b))
    return k * V


def compute_flow(N_values, kappa=1.0, lambda_=0.2, L=500, T=15000, dt=0.1, n_dec=1, save_file='flow_data.npz',
                 skip_if_exists=True):
    """Compute the fundamental-diagram data and save results.

    The normal (solid) and night (dashed) homogeneous branches are computed
    analytically on a fine density grid (``k_fine``); only the night perturbed
    (dotted) branch is simulated, on the ``N_values`` grid (``density_values``).

    If skip_if_exists=True and save_file already exists, loads and returns cached data.
    Returns (k_fine, normal_flows, night_flows, density_values, night_perturbed_flows).
    """
    if skip_if_exists and os.path.exists(save_file):
        print(f"Loading cached data from {save_file} (pass skip_if_exists=False to recompute)")
        data = np.load(save_file)
        return (data['k_fine'], data['normal_flows'], data['night_flows'],
                data['density_values'], data['night_perturbed_flows'])

    # Analytical homogeneous branches on a fine grid (exact bends at k_c1, k_c2).
    k_fine = np.linspace(0.002, 1.12, 600)
    normal_flows = homogeneous_flow(k_fine, night_mode=False)
    night_flows = homogeneous_flow(k_fine, night_mode=True)

    # Simulated night perturbed branch (small n_dec=1 or large n_dec=80).
    density_values = []
    night_perturbed_flows = []
    for N in tqdm.tqdm(N_values):
        config = Config(N=N, lambda_=lambda_, kappa=kappa, T=T, dt=dt, L=L, n_dec=n_dec)
        density = N / L
        density_values.append(density)
        sim = FVDSimulation(config, night_mode=True)
        _, velocities = sim.run(perturb=True, save=False)
        avg_velocity = np.mean(velocities[-1000:])  # steady-state mean over last 1000 steps
        night_perturbed_flows.append(density * avg_velocity)

    np.savez(save_file,
             k_fine=k_fine,
             normal_flows=normal_flows,
             night_flows=night_flows,
             density_values=np.asarray(density_values),
             night_perturbed_flows=np.asarray(night_perturbed_flows))
    print(f"Data saved to {save_file}")

    return k_fine, normal_flows, night_flows, np.asarray(density_values), np.asarray(night_perturbed_flows)

# Paper labels for the two V'(Δx)=κ/2+λ stability boundaries depend on λ:
# Fig. 4 (λ=0.2) names them k_c3/k_c4; Fig. 6 (λ=0.1) names them k_c5/k_c6.
_STABILITY_LABELS = {
    0.2: (r'$k_{c3}$', r'$k_{c4}$'),
    0.1: (r'$k_{c5}$', r'$k_{c6}$'),
}


def compute_critical_densities(kappa, lambda_, xc=2.0, xc1=3.2, xc2=4.0,
                               stability_labels=None):
    """
    Compute analytically-derived critical densities for the fundamental diagram.

    k_c2, k_c1: boundaries of the V'<0 region (from x_c2, x_c1).
    The two stability-boundary crossings where V'(Δx) = κ/2 + λ are labeled
    according to the case (k_c3/k_c4 for λ=0.2, k_c5/k_c6 for λ=0.1), or by the
    explicit ``stability_labels`` (low, high) tuple if given.

    Returns list of (density, label) tuples, sorted by density ascending.
    """
    kc2 = 1.0 / xc2   # lower density boundary of V'<0 region (≈ 0.25)
    kc1 = 1.0 / xc1   # upper density boundary of V'<0 region (≈ 0.3125)

    threshold = kappa / 2.0 + lambda_
    vlines = [(kc2, r'$k_{c2}$'), (kc1, r'$k_{c1}$')]

    # sech²(0) = 1; threshold must be < 1 for crossings to exist
    if threshold < 1.0:
        if stability_labels is None:
            stability_labels = _STABILITY_LABELS.get(
                round(lambda_, 3), (r'$k_{c3}$', r'$k_{c4}$'))
        low_label, high_label = stability_labels
        u = np.arccosh(1.0 / np.sqrt(threshold))
        k_low  = 1.0 / (xc + u)   # low-density stability boundary
        k_high = 1.0 / (xc - u)   # high-density stability boundary
        vlines += [(k_low, low_label), (k_high, high_label)]

    return sorted(vlines, key=lambda pair: pair[0])


def detect_perturbation_features(density_values, night_perturbed_flows,
                                 rel_tol=0.03, k_min=0.0):
    """
    Locate the kinks of the dotted (perturbed) branch — the densities where the
    simulated traffic enters, switches regime within, and leaves its unstable state.
    These are the critical densities the paper marks with vertical lines; they fall on
    the visible bends of the perturbed curve rather than on the linear-stability values
    (a finite n_dec perturbation makes the nonlinear region differ slightly from the
    infinitesimal V'=κ/2+λ prediction).

    The perturbation makes the perturbed flow depart from the analytical homogeneous
    night flow. Because the night OV function is non-monotonic, that departure is below
    the homogeneous branch at low density (one cluster / clustering) and above it at
    higher density (kink-antikink / unstable clusters). Restricting to ``k > k_min``
    (used to skip the separately-marked V'<0 clustering band between k_c2 and k_c1),
    we return the bends as a dict:
      * 'lower'   : lower edge where the perturbed curve first departs (k_c3 / k_c5);
      * 'crossing': below→above zero crossing of (q_pert - q_homog) inside the region —
                    the regime boundary (k_c7);
      * 'upper'   : upper edge where the perturbed curve rejoins the stable branch
                    (k_c4 / k_c6 / k_c7 / k_c8 / k_c9).
    Any feature that does not exist is omitted. Returns {} if nothing deviates.
    """
    k = np.asarray(density_values, dtype=float)
    qp = np.asarray(night_perturbed_flows, dtype=float)
    order = np.argsort(k)
    k, qp = k[order], qp[order]
    qh = homogeneous_flow(k, night_mode=True)  # analytical reference at the perturbed grid

    diff = qp - qh
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_dev = np.where(qh > 1e-9, np.abs(diff) / qh, 0.0)
    deviates = (rel_dev > rel_tol) & (k > k_min)
    if not np.any(deviates):
        return {}

    idx = np.where(deviates)[0]
    lo_idx, hi_idx = idx.min(), idx.max()

    # Place the edges on the actual corner samples of the perturbed curve: the bottom
    # corner of the initial drop (first deviating sample) and the top corner of the
    # final drop before it rejoins the stable branch (last deviating sample).
    features = {
        'lower': float(k[lo_idx]),
        'upper': float(k[hi_idx]),
    }

    # Interior regime boundary (large perturbation): the last sample still at/below the
    # homogeneous branch just before the below→above crossing — the one-cluster ↔
    # kink-antikink boundary (k_c7 for Figs. 12/14).
    for i in range(lo_idx, hi_idx):
        if diff[i] <= 0.0 < diff[i + 1]:
            features['crossing'] = float(k[i])
            break

    # Local peak of the perturbed flow inside the deviation region — the stable-cluster
    # regime gives maximum flow here, just before clusters turn unstable (k_c7 for Fig.6).
    features['peak'] = float(k[lo_idx + int(np.argmax(qp[lo_idx:hi_idx + 1]))])

    return features


def fundamental_vlines(density_values, night_perturbed_flows,
                       include_kc12=False, feature_labels=None, k_min=0.0,
                       xc1=3.2, xc2=4.0):
    """
    Assemble the (density, label) vertical-line list for a fundamental diagram.

    - include_kc12: prepend the analytical V'<0 boundaries k_c2=1/xc2, k_c1=1/xc1.
    - feature_labels: dict mapping detected-feature name ('lower'/'crossing'/'upper')
      to its LaTeX label, e.g. {'lower': r'$k_{c3}$', 'upper': r'$k_{c4}$'}.
    - k_min: lower density cutoff for feature detection (skip the clustering band).

    Returns a list of (density, label) tuples sorted by density.
    """
    vlines = []
    if include_kc12:
        vlines += [(1.0 / xc2, r'$k_{c2}$'), (1.0 / xc1, r'$k_{c1}$')]
    if feature_labels:
        feats = detect_perturbation_features(density_values, night_perturbed_flows, k_min=k_min)
        for name, label in feature_labels.items():
            if name in feats:
                vlines.append((feats[name], label))
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
        k_fine = data['k_fine']                       # fine grid for analytical branches
        normal_flows = data['normal_flows']
        night_flows = data['night_flows']
        density_values = data['density_values']       # simulation grid for perturbed branch
        night_perturbed_flows = data['night_perturbed_flows']
        print(f"Data loaded from {load_file}")
    except FileNotFoundError:
        print(f"Error: File {load_file} not found.")
        return
    except KeyError as e:
        print(f"Error: Missing key {e} in {load_file}. Ensure all arrays are saved correctly.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(k_fine, normal_flows, 'k-', label='Normal driving')
    ax.plot(k_fine, night_flows, 'k--', label='Night driving')
    ax.plot(density_values, night_perturbed_flows, 'k:', label='Night driving (perturbed)')

    ax.set_xlabel('density')
    ax.set_ylabel('flow')
    ax.set_title(title)
    # Fixed axes to match the paper's fundamental diagrams (Figs. 4/6/10/12/14).
    ax.set_xlim(0.0, 1.1)
    ax.set_ylim(0.0, 0.8)
    ax.legend()
    fig.tight_layout()

    if vlines:
        # Place labels just above the x-axis, relative to the fixed y-limits.
        y_label = 0.02 * 0.8
        for density, label in vlines:
            ax.axvline(x=density, color='k', linestyle=':', linewidth=0.8)
            ax.text(density, y_label, label,
                    ha='center', va='bottom', fontsize=9, color='k')

    fig.savefig(output_plot, bbox_inches='tight')
    print(f"Plot saved to {output_plot}")
    try:
        plt.show()
    except Exception:
        pass

def plot_traffic_pattern(position_filename, output_name, N=150, kappa=1.0, lambda_=0.2, L=500, dt=0.1,
                         title_suffix=''):
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
    - title_suffix (str): Extra text appended to the plot title (e.g. ', A=0.1').
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
    plt.title(f'Space-Time Plot (N={N}, κ={kappa}, λ={lambda_}{title_suffix})')
    plt.tight_layout()
    plt.savefig(output_name)
    print(f"Plot saved to {output_name}")
    try:
        plt.show()
    except:
        pass