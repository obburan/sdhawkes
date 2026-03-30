import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import sys
from functools import partial   
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from SDHawkes_2d_config import (
    alpha, beta, mu, delta,
    Lambda_0, t_span,
    output_dir, NUM_SIMS, NUM_WORKERS, n_system,
    FLLN_scaling, num_FLLN_paths,
    USE_DISK,
    YLIM,
)

from SDHawkes_2d_sim import (
    create_H,
    background_intensity,
    FLLN_sim,
    FLLN_plot,
    compute_FLLN_ODE_difference,
    solve_ode
)

def main():
    """
    Main script to run Hawkes process simulations and plot results alongside ODE solutions.

    Global variables used:
    - t_span: tuple of (t_start, t_end) for ODE solver
    - FLLN_scaling: scaling factor for FLLN simulations
    - num_FLLN_paths: number of FLLN paths to simulate
    """

    global t_span, alpha, beta, delta, mu
    T = t_span[1]

    print("\n" + "="*80)
    print("STEP 1: Solving and Plotting ODE")
    print("="*80)
    
    H_func = create_H(alpha, beta)
    
    params_dict = {
        r'\alpha': alpha,
        r'\beta': beta,
        r'\delta': delta,
        r'\mu': mu
    }

    # Need to solve the ODE system if not doing plotting (which solves ODE within)
    # Create time points for evaluation
    t_eval = np.linspace(t_span[0], t_span[1], n_system)
    sol = solve_ode(t_span, Lambda_0, H_func, background_intensity_func=background_intensity, t_eval=t_eval, method='DOP853', rtol=1e-8, atol=1e-8, max_step=0.01)
    if not sol['success']:
        print(f"  Message: {sol['message']}")

    print("\n" + "="*80)
    print("STEP 2: FLLN Simulations and Comparison with ODE")
    print("="*80)

    # FLLN parameters
    global FLLN_scaling, num_FLLN_paths, USE_DISK, YLIM
    
    # Run FLLN simulations
    print(f"\nRunning FLLN simulations with:")
    print(f"  FLLN scaling (n): {FLLN_scaling}")
    print(f"  Number of paths: {num_FLLN_paths}")
    print(f"  Time interval: [0, {T}]")
    print(f"  Disk mode: {USE_DISK}")
    
    flln_result = FLLN_sim(num_FLLN_paths, T, FLLN_scaling, use_disk=USE_DISK)
    
    # Handle different return types based on use_disk
    if USE_DISK:
        flln_paths_list, subfolder_path = flln_result
        print(f"\nCompleted {len(flln_paths_list)} FLLN simulations successfully!")
    else:
        flln_paths_list = flln_result
        subfolder_path = output_dir
        print(f"\nCompleted {len(flln_paths_list)} FLLN simulations successfully!")

    # Determine stability label for plotting
    alpha_very_unstable = np.array([[0.8, 0.18], [0.18, 0.8]])
    alpha_medium_unstable = np.array([[0.6, 0.26], [0.26, 0.6]])
    alpha_very_stable = np.array([[0.3, 0.2], [0.2, 0.3]])
    
    if np.allclose(alpha, alpha_very_unstable):
        stability_label = "_very_unstable"
    elif np.allclose(alpha, alpha_medium_unstable):
        stability_label = "_medium_unstable"
    elif np.allclose(alpha, alpha_very_stable):
        stability_label = "_very_stable"
    else:
        stability_label = "_CUSTOM"

    # Compute differences between FLLN paths and ODE solution
    print("\nComputing differences between FLLN paths and ODE solution...")
    T = t_span[1]
    diff_results = compute_FLLN_ODE_difference(flln_paths_list, sol, T, num_points=1000)
    print(f"Overall max difference: {diff_results['overall_max_difference']:.6f}")
    print(f"Max difference per path: {diff_results['max_difference_per_path']}")
    print(f"Mean of max differences: {np.mean(diff_results['max_difference_per_path']):.6f}")
    
    # Create plot comparing FLLN paths with ODE solution
    # Save to subfolder if using disk, otherwise to output_dir
    print("\nPlotting FLLN paths with ODE solution...")
    flln_filename = FLLN_plot(
        paths_list=flln_paths_list,
        ode_solution=sol,
        diff_results=diff_results,
        output_dir=subfolder_path,
        FLLN_scaling=FLLN_scaling,
        T_final=T,
        time_grid_size=1000,
        ylim=YLIM,
        stability_label=stability_label
    )
    print(f"FLLN comparison plot saved to {flln_filename}")
    
    print("\n" + "="*80)
    print("All steps completed successfully!")
    print("="*80)



if __name__ == "__main__":
    main()


