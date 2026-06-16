import numpy as np
import tqdm
from config import Config
import matplotlib.pyplot as plt
from fvd_simulation import FVDSimulation
from utils import plot_traffic_pattern, compute_flow, load_and_plot_fundamental_diagram

if __name__ == "__main__":
    load_and_plot_fundamental_diagram('flow_data4.npz', 'figure4_replicated.png', 'Fundamental Diagram (κ=1.0, λ=0.2)')
    # load_and_plot_fundamental_diagram('flow_data10.npz', 'figure10_replicated.png', 'Fundamental Diagram (κ=1.0, λ=0.5)')

