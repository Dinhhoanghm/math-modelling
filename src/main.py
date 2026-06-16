import numpy as np
import matplotlib.pyplot as plt
from config import Config
from fvd_simulation import FVDSimulation
from utils import plot_traffic_pattern, compute_flow, load_and_plot_fundamental_diagram, compute_critical_densities


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
    ax.axvline(x=xc1, color='k', linestyle=':')
    ax.axvline(x=xc2, color='k', linestyle=':')
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
    ax.axvline(x=xc1, color='k', linestyle=':')
    ax.axvline(x=xc2, color='k', linestyle=':')
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
    plot_traffic_pattern(config.output_file, output_png, config.N, config.kappa, config.lambda_)


def _fundamental(save_file, output_png, kappa, lambda_, n_dec=1):
    """Compute (or load cached) flow data and plot fundamental diagram."""
    compute_flow(np.arange(1, 500, 10), kappa=kappa, lambda_=lambda_,
                 L=500, T=25000, dt=0.1, n_dec=n_dec, save_file=save_file)
    vlines = compute_critical_densities(kappa, lambda_) if n_dec == 1 else None
    load_and_plot_fundamental_diagram(save_file, output_png,
                                      f'Fundamental Diagram (κ={kappa}, λ={lambda_})',
                                      vlines=vlines)


# ── Small perturbations (n_dec=1) ────────────────────────────────────────────

def plot_figures_3_4_5():
    """κ=1.0, λ=0.2 — clustering (Fig3), fundamental diagram (Fig4), kink-antikink (Fig5)"""
    kappa, lambda_ = 1.0, 0.2
    _space_time(Config(T=15000, kappa=kappa, lambda_=lambda_, output_file='figure3_positions.npy'))
    _fundamental('flow_data4.npz', 'figure4_replicated.png', kappa, lambda_)
    _space_time(Config(T=15000, N=250, kappa=kappa, lambda_=lambda_, output_file='figure5_positions.npy'))


def plot_figures_6_7_8_9():
    """κ=1.0, λ=0.1 — fundamental (Fig6), stable clusters (Fig7), unstable clusters (Fig8), stochastic (Fig9)"""
    kappa, lambda_ = 1.0, 0.1
    _fundamental('flow_data6.npz', 'figure6_replicated.png', kappa, lambda_)
    _space_time(Config(T=15000, N=230, kappa=kappa, lambda_=lambda_, output_file='figure7_positions.npy'))
    # Fig 8a: FVD model
    _space_time(Config(T=15000, N=300, kappa=kappa, lambda_=lambda_, output_file='figure8a_positions.npy'))
    # Fig 8b: OV model (λ=0, κ=1.2)
    _space_time(Config(T=15000, N=300, kappa=1.2, lambda_=0, output_file='figure8b_positions.npy'))
    # Fig 9: stochastic effect at three noise magnitudes
    for A, suffix in [(0.01, 'a'), (0.05, 'b'), (0.1, 'c')]:
        _space_time(Config(T=15000, N=300, kappa=kappa, lambda_=lambda_,
                           stochastic=True, A=A, output_file=f'figure9{suffix}_positions.npy'))


# ── Large perturbations (n_dec=80) ───────────────────────────────────────────

def plot_figures_10_11():
    """κ=1.0, λ=0.5, large pert — fundamental (Fig10), one cluster no density wave (Fig11)"""
    kappa, lambda_, n_dec = 1.0, 0.5, 80
    _fundamental('flow_data10.npz', 'figure10_replicated.png', kappa, lambda_, n_dec=n_dec)
    _space_time(Config(T=14000, N=220, kappa=kappa, lambda_=lambda_, n_dec=n_dec,
                       output_file='figure11_positions.npy'))


def plot_figures_12_13():
    """κ=1.0, λ=0.2, large pert — fundamental (Fig12), one cluster + density wave (Fig13)"""
    kappa, lambda_, n_dec = 1.0, 0.2, 80
    _fundamental('flow_data12.npz', 'figure12_replicated.png', kappa, lambda_, n_dec=n_dec)
    _space_time(Config(T=14000, N=220, kappa=kappa, lambda_=lambda_, n_dec=n_dec,
                       output_file='figure13_positions.npy'))


def plot_figure14():
    """κ=1.0, λ=0.1, large pert — fundamental diagram only"""
    _fundamental('flow_data14.npz', 'figure14_replicated.png', kappa=1.0, lambda_=0.1, n_dec=80)


def main():
    plot_figure1()
    plot_figure2()
    plot_figures_3_4_5()
    plot_figures_6_7_8_9()
    plot_figures_10_11()
    plot_figures_12_13()
    plot_figure14()


if __name__ == "__main__":
    main()
