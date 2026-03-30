import numpy as np
from pathlib import Path

"""
Configuration file for 2D State-Dependent Hawkes process simulation. All experiment parameters 
are defined here and imported by SDHawkes_2d_sim.py, SDHawkes_2d_main.py, and other scripts.
"""

### Meta
PROJECT_ROOT = Path(__file__).resolve().parents[1]
output_dir = str(PROJECT_ROOT / "Simulation_Outputs")
USE_DISK = True # whether or not the simulations should be saved to disk or kept in memory. Only set to True if performing many simulations, simulating for long time, or simulating a nearly critical process.
NUM_SIMS = 120 # Use multiple of 12 to take full advantage of parallelization
max_arrivals = 500000
n_system = 10000 # number of grid points to use when plotting ODE solution
NUM_WORKERS = 4 # number of parallel threads to use for parallel simulations; set equal to 1 if wishing to avoid parallel simulations. NOTE: make sure to input a number compatible with your hardware.


### Simulation parameters
dim = 2
mu = np.array([200,120])  # 2D background intensity vector
MAXIMUM_STATE_CUTOFF = 200
# alpha = np.array([[0.8, 0.18], [0.18, 0.8]]) # Very unstable: NOTE: Use MAXIMUM_STATE_CUTOFF = 20 for this choice of alpha. WARNING: if FLLN_scaling = 1000, results in very large simulation files (~2GB).
# alpha = np.array([[0.6, 0.26], [0.26, 0.6]]) # Medium unstable: unstable at extreme states. NOTE: Use MAXIMUM_STATE_CUTOFF = 200.
alpha = np.array([[0.3, 0.2], [0.2, 0.3]]) # Very stable: stable for all states. NOTE: Use MAXIMUM_STATE_CUTOFF = 200.
# alpha = np.array([[0.95, 0.05], [0.05, 0.3]]) # Non-symmetric case 1: asymmetric diagonal dominance. NOTE: Use MAXIMUM_STATE_CUTOFF = 20 for this choice of alpha.
# alpha = np.array([[0.4, 0.7], [0.1, 0.4]]) # Non-symmetric case 2: Strong Asymmetric Cross-Excitation. NOTE: Use MAXIMUM_STATE_CUTOFF = 20 for this choice of alpha.
# alpha = np.array([[0.98, 0.01], [0.01, 0.2]]) # Non-symmetric case 3. NOTE: Use MAXIMUM_STATE_CUTOFF = 20 for this choice of alpha.
# alpha = np.array([[0.3, 0.85], [0.05, 0.3]]) # Non-symmetric case 4: Off-Diagonal dominance. NOTE: Use MAXIMUM_STATE_CUTOFF = 20 for this choice of alpha.
# alpha = np.array([[0.7, 1e-10], [0.7, 1e-10]]) # NOTE: Use MAXIMUM_STATE_CUTOFF = 20 for this choice of alpha.
# alpha = np.array([[0.5, 0.4], [0.4, 0.5]]) # NOTE: Use MAXIMUM_STATE_CUTOFF = 20 for this choice of alpha.
# alpha = np.array([[.96,.06],[.06,.25]]) # NOTE: Use MAXIMUM_STATE_CUTOFF = 20 for this choice of alpha.
beta = np.ones_like(alpha)  
delta = .001

### ODE parameters
Lambda_0 = np.zeros(2)  # ODE initial condition is zero
t_span = (0, 1)  # Simulate for 1 time unit

### FLLN parameters
FLLN_scaling = 100
num_FLLN_paths = 100
YLIM = None # (0,30)  # Y-axis limits for FLLN plot as (ymin, ymax), or None for auto
