import numpy as np
import matplotlib.pyplot as plt
from config import Config
from fvd_simulation import FVDSimulation
from utils import (plot_traffic_pattern, compute_flow, load_and_plot_fundamental_diagram,
                   fundamental_vlines)

# Lower density cutoff for empirical feature detection: skip the V'<0 clustering band
# (k_c2..k_c1) which is marked separately by the analytical k_c1/k_c2 lines.
_FEATURE_KMIN = 1.0 / 3.2 + 0.01


def plot_figure1():
    """V(Δx) for normal (solid) and night (dashed) OV functions — purely analytical, no simulation."""
    xc, xc1, xc2 = 2.0, 3.2, 4.0
    a, b = 5.0, 1.0
    delta_x = np.linspace(0, 10, 1000)

    v_normal = np.tanh(delta_x - xc) + np.tanh(xc)

    v_night = np.where(
        delta_x < xc1,
        np.tanh(delta_x - xc) + np.tanh(xc),
        np.where(delta_x <= xc2, a - delta_x, b)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(delta_x, v_normal, 'k-',  label='Normal OV (Eq. 5)')
    ax.plot(delta_x, v_night,  'k--', label='Night OV (Eq. 6)')
    ax.set_xlabel('headway')
    ax.set_ylabel('velocity')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.legend()
    fig.tight_layout()
    fig.savefig('figure1_replicated.png')
    print("Plot saved to figure1_replicated.png")
    try:
        plt.show()
    except Exception:
        pass


def plot_figure2():
    """V'(Δx) for normal (solid) and night (dashed) OV functions — purely analytical, no simulation."""
    xc, xc1, xc2 = 2.0, 3.2, 4.0
    delta_x = np.linspace(0, 10, 1000)

    v_prime_normal = 1.0 / np.cosh(delta_x - xc) ** 2

    v_prime_night = np.where(
        delta_x < xc1,
        1.0 / np.cosh(delta_x - xc) ** 2,
        np.where(delta_x <= xc2, -1.0, 0.0)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(delta_x, v_prime_normal, 'k-',  label='Normal OV (Eq. 5)')
    ax.plot(delta_x, v_prime_night,  'k--', label='Night OV (Eq. 6)')
    ax.axhline(y=0,   color='k', linewidth=0.5)
    ax.set_xlabel('headway')
    ax.set_ylabel("V'(Δx)")
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.2, 1.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig('figure2_replicated.png')
    print("Plot saved to figure2_replicated.png")
    try:
        plt.show()
    except Exception:
        pass


def _space_time(config):
    """Run simulation and save space-time plot. Output PNG name derived from .npy filename."""
    FVDSimulation(config).run()
    output_png = config.output_file.replace('_positions.npy', '_replicated.png')
    suffix = f', A={config.A}' if config.stochastic else ''
    plot_traffic_pattern(config.output_file, output_png, config.N, config.kappa, config.lambda_,
                         title_suffix=suffix)


def _fundamental(save_file, output_png, kappa, lambda_, n_dec=1,
                 include_kc12=False, feature_labels=None):
    """Compute (or load cached) flow data and plot the fundamental diagram.

    Critical-density lines are placed on the kinks of the simulated perturbed curve
    (see utils.detect_perturbation_features). ``include_kc12`` adds the analytical
    V'<0 boundaries k_c1/k_c2; ``feature_labels`` maps detected kinks to labels.
    """
    # Finer density grid for a smoother perturbed (dotted) curve, up to k≈1.1.
    _, _, _, density, night_pert = compute_flow(
        np.arange(5, 556, 5), kappa=kappa, lambda_=lambda_,
        L=500, T=25000, dt=0.1, n_dec=n_dec, save_file=save_file)
    vlines = fundamental_vlines(density, night_pert, include_kc12=include_kc12,
                                feature_labels=feature_labels, k_min=_FEATURE_KMIN)
    load_and_plot_fundamental_diagram(save_file, output_png,
                                      f'Fundamental Diagram (κ={kappa}, λ={lambda_})',
                                      vlines=vlines)


# ── Small perturbations (n_dec=1) ────────────────────────────────────────────

def plot_figure3():
    """Fig 3: N=150, κ=1.0, λ=0.5 — clustering in the V'<0 range (xc1<Δx<xc2)."""
    _space_time(Config(T=15000, N=150, kappa=1.0, lambda_=0.5,
                       output_file='figure3_positions.npy'))


def plot_figures_4_5():
    """κ=1.0, λ=0.2 — fundamental diagram (Fig4), kink-antikink (Fig5)"""
    kappa, lambda_ = 1.0, 0.2
    # k_c2, k_c1 analytical (V'<0 band); k_c3, k_c4 = kinks of the kink-antikink region.
    _fundamental('flow_data4.npz', 'figure4_replicated.png', kappa, lambda_,
                 include_kc12=True,
                 feature_labels={'lower': r'$k_{c3}$', 'upper': r'$k_{c4}$'})
    _space_time(Config(T=15000, N=250, kappa=kappa, lambda_=lambda_, output_file='figure5_positions.npy'))


def plot_figures_6_7_8_9():
    """κ=1.0, λ=0.1 — fundamental (Fig6), stable clusters (Fig7), unstable clusters (Fig8), stochastic (Fig9)"""
    kappa, lambda_ = 1.0, 0.1
    # k_c2, k_c1 analytical; k_c5 lower edge, k_c7 = perturbed-flow peak (stable→unstable
    # cluster transition), k_c6 upper edge.
    _fundamental('flow_data6.npz', 'figure6_replicated.png', kappa, lambda_,
                 include_kc12=True,
                 feature_labels={'lower': r'$k_{c5}$', 'peak': r'$k_{c7}$', 'upper': r'$k_{c6}$'})
    _space_time(Config(T=15000, N=230, kappa=kappa, lambda_=lambda_, output_file='figure7_positions.npy'))
    # Fig 8a: FVD model
    _space_time(Config(T=15000, N=300, kappa=kappa, lambda_=lambda_, output_file='figure8a_positions.npy'))
    # Fig 8b: OV model (λ=0, κ=1.2)
    _space_time(Config(T=15000, N=300, kappa=1.2, lambda_=0, output_file='figure8b_positions.npy'))
    # Fig 9: stochastic effect at three noise magnitudes. A long run (T=60000) is needed
    # for A=0.1 so the macroscopic high-density region has time to nucleate (Fig 9c).
    for A, suffix in [(0.01, 'a'), (0.05, 'b'), (0.1, 'c')]:
        _space_time(Config(T=60000, N=300, kappa=kappa, lambda_=lambda_,
                           stochastic=True, A=A, output_file=f'figure9{suffix}_positions.npy'))


# ── Large perturbations (n_dec=80) ───────────────────────────────────────────

def plot_figures_10_11():
    """κ=1.0, λ=0.5, large pert — fundamental (Fig10), one cluster no density wave (Fig11)"""
    kappa, lambda_, n_dec = 1.0, 0.5, 80
    _fundamental('flow_data10.npz', 'figure10_replicated.png', kappa, lambda_, n_dec=n_dec,
                 feature_labels={'upper': r'$k_{c7}$'})
    _space_time(Config(T=14000, N=220, kappa=kappa, lambda_=lambda_, n_dec=n_dec,
                       output_file='figure11_positions.npy'))


def plot_figures_12_13():
    """κ=1.0, λ=0.2, large pert — fundamental (Fig12), one cluster + density wave (Fig13)"""
    kappa, lambda_, n_dec = 1.0, 0.2, 80
    _fundamental('flow_data12.npz', 'figure12_replicated.png', kappa, lambda_, n_dec=n_dec,
                 feature_labels={'crossing': r'$k_{c7}$', 'upper': r'$k_{c8}$'})
    _space_time(Config(T=14000, N=220, kappa=kappa, lambda_=lambda_, n_dec=n_dec,
                       output_file='figure13_positions.npy'))


def plot_figure14():
    """κ=1.0, λ=0.1, large pert — fundamental diagram only"""
    _fundamental('flow_data14.npz', 'figure14_replicated.png', kappa=1.0, lambda_=0.1, n_dec=80,
                 feature_labels={'crossing': r'$k_{c7}$', 'upper': r'$k_{c9}$'})


def main():
    plot_figure1()
    plot_figure2()
    plot_figure3()
    plot_figures_4_5()
    plot_figures_6_7_8_9()
    plot_figures_10_11()
    plot_figures_12_13()
    plot_figure14()


if __name__ == "__main__":
    main()
