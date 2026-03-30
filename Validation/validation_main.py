import numpy as np
from tqdm import trange
from datetime import datetime
import os
import sys
from typing import Tuple, List
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from validation import (
    simulate_SAHawkes,
    simulate_Exp_SDHawkes,
    simulate_SDHawkes,
    simulate_2D_SDHawkes,
    compare_paths,
    compare_implementations_multi_path,
    analytical_exp_Hawkes_count_mean,
    analytical_exp_Hawkes_intensity_mean,
    compute_count_process,
    reconstruct_intensity_from_path,
    plot_count_comparison,
    plot_intensity_comparison
)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Global Parameters and Configuration

# Simulation time horizon
T = 1.0

# Random seed for reproducibility in single-path comparison
SEED = 2026

# Output directory configuration
BASE_OUTPUT_DIR = str(Path(__file__).resolve().parents[2] / "Simulation_Outputs")
VALIDATION_DIR = os.path.join(BASE_OUTPUT_DIR, "Validation")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Single-Path Comparison Parameters

# 2D Hawkes process parameters for equivalence testing
MU_2D = np.array([200.0, 120.0])  # 2D background intensity vector
ALPHA_2D = np.array([[0.8, 0.18], [0.18, 0.8]])  # Very unstable excitation matrix
# ALPHA_2D = np.array([[0.6, 0.26], [0.26, 0.6]])  # Medium unstable: unstable at extreme states
# ALPHA_2D = np.array([[0.3, 0.2], [0.2, 0.3]])  # Very stable: stable for all states
BETA_2D = np.ones_like(ALPHA_2D)  # Decay matrix (all ones)

# Single path simulation settings
NUM_PATHS_SINGLE = 1  # Number of paths for *exact (path-by-path)* equivalence testing
MAX_ARRIVALS_SINGLE = 500000  # Maximum arrivals per path

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Monte Carlo Analytical Comparison Parameters: compare MC means to analytical means for count process and intensity (must be 1-dimensional process)

# 1D Hawkes process parameters for exact analytical comparison
MU_1D = np.array([100.0])  # 1D background intensity
ALPHA_1D = np.array([[0.6]])  # 1D excitation parameter (stable if alpha < beta)
BETA_1D = np.array([[1.0]])  # 1D decay parameter

# Monte Carlo simulation settings
NUM_MC_PATHS = 100  # Number of paths per Monte Carlo experiment
TIME_GRID_POINTS = 1000  # Number of time points for evaluation

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Statistical Analysis Parameters

# Number of times to re-run entire Monte Carlo for statistical analysis
NUM_MC_RUNS = 50  # Each run uses NUM_MC_PATHS paths

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Auxiliary Functions

def background_intensity_func_2d(t: float, state: float) -> np.ndarray:
    """Constant background intensity for 2D process (state-agnostic)."""
    return MU_2D

def background_intensity_func_1d(t: float, state: float) -> np.ndarray:
    """Constant background intensity for 1D process (state-agnostic)."""
    return MU_1D

def r_2d(i: int, j: int, y: float) -> float:
    """No state dependence in excitation kernel for 2D process."""
    return 1.0

def r_1d(i: int, j: int, y: float) -> float:
    """No state dependence in excitation kernel for 1D process."""
    return 1.0

def process_simulations(sims: List[Tuple], mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray, time_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process simulation results to compute mean count and intensity processes.
    
    Parameters:
    -----------
    sims : list
        List of simulation results (paths, info) tuples
    mu : np.ndarray
        Background intensity
    alpha : np.ndarray
        Excitation matrix
    beta : np.ndarray
        Decay matrix
    time_grid : np.ndarray
        Time points for evaluation
        
    Returns:
    --------
    tuple of (mean_count, mean_intensity)
        Mean count and intensity processes (1D arrays)
    """
    count_processes_list = []
    intensity_processes_list = []
    for i in range(len(sims)):
        paths_i, info_i = sims[i]
        count_processes_list.append(compute_count_process(paths_i, time_grid))
        intensity_processes_list.append(reconstruct_intensity_from_path(info_i, mu, alpha, beta, time_grid))
    mean_count = np.mean(count_processes_list, axis=0)[:, 0]  # Extract 1D
    mean_intensity = np.mean(intensity_processes_list, axis=0)[:, 0]  # Extract 1D
    return mean_count, mean_intensity

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Main Script

if __name__ == "__main__":
    
    print("="*80)
    print("HAWKES PROCESS VALIDATION SUITE")
    print("="*80)
    print("\nThis script validates multiple Hawkes process implementations by:")
    print("1. Single-path comparison: Checking if different implementations produce identical results")
    print("2. Monte Carlo comparison: Comparing simulations against analytical solutions")
    print("3. Statistical analysis: Computing statistics on maximum differences across multiple MC runs")
    print("="*80 + "\n")
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ## PART 1: Path Simulation Exactness Comparison
    """
    The three simulation scripts: SAHawkes, SDHawkes (general), and Exp_SDHawkes, should not only produce paths distributed the same, but should be exactly the same in the special case we have constructed. This takes on two forms relevant to us:
    - Single paths being exactly identical: there is no reason the first paths produced from the same random seed should not be exactly the same.
    - Potential seeding issues across multiple paths: when using parallel simulation (for the state-dependent simulations, as parallel simulation is not implemented for state-agnostic Hawkes), the resultant paths should be exactly the same as those produced by the state-agnostic Hawkes. In particular, we should not have that the seeds update in any different way between the different simulations (note that we use num_workers=4 in the definition of the state-dependent simulation functions within validation.py)
    """
    
    print("="*80)
    print("PART 1: SINGLE-PATH SIMULATION COMPARISON")
    print("="*80)
    print("\nComparing 4 implementations with identical random seed:")
    print("  - SAHawkes (state-agnostic)")
    print("  - Exp_SDHawkes (exponential kernel state-dependent)")
    print("  - SDHawkes (general state-dependent)")
    print("  - 2D_SDHawkes (2D-specific state-dependent)")
    print(f"\nParameters: μ={MU_2D}, α={ALPHA_2D[0]}, β={BETA_2D[0]}")
    print(f"Time horizon: T={T}, Seed: {SEED}\n")
    
    # Spawn reproducible child seeds from SeedSequence for cross-implementation comparison
    ss_single = np.random.SeedSequence(SEED)
    single_seeds = ss_single.spawn(NUM_PATHS_SINGLE)
    
    print("Running SAHawkes simulation...")
    SA_sims = simulate_SAHawkes(
        T=T,
        mu=MU_2D,
        alpha=ALPHA_2D,
        beta=BETA_2D,
        num_paths=NUM_PATHS_SINGLE,
        seeds_list=single_seeds
    )
    print("Number of SAHawkes arrivals:", sum(len(SA_sims[0][0][d]) for d in range(len(SA_sims[0][0]))))
    
    print("Running Exp_SDHawkes simulation...")
    Exp_SD_sims = simulate_Exp_SDHawkes(
        T=T,
        background_intensity=background_intensity_func_2d,
        alpha=ALPHA_2D,
        beta=BETA_2D,
        r=r_2d,
        num_paths=NUM_PATHS_SINGLE,
        seeds_list=single_seeds
    )
    print("Number of Exp_SDHawkes arrivals:", sum(len(Exp_SD_sims[0][0][d]) for d in range(len(Exp_SD_sims[0][0]))))
    
    print("Running SDHawkes simulation...")
    SD_sims = simulate_SDHawkes(
        T=T,
        mu=MU_2D,
        alpha=ALPHA_2D,
        beta=BETA_2D,
        num_paths=NUM_PATHS_SINGLE,
        seeds_list=single_seeds
    )
    print("Number of SDHawkes arrivals:", sum(len(SD_sims[0][0][d]) for d in range(len(SD_sims[0][0]))))
    
    print("Running 2D_SDHawkes simulation...")
    SD2d_sims = simulate_2D_SDHawkes(
        T=T,
        background_intensity=background_intensity_func_2d,
        alpha=ALPHA_2D,
        beta=BETA_2D,
        r=r_2d,
        FLLN_scaling=1.0,
        num_paths=NUM_PATHS_SINGLE,
        max_arrivals=MAX_ARRIVALS_SINGLE,
        seeds_list=single_seeds
    )
    print("Number of 2D_SDHawkes arrivals:", sum(len(SD2d_sims[0][0][d]) for d in range(len(SD2d_sims[0][0]))))
    
    print("\nAll simulations complete!")
    
    # Compare simulation results
    print("\n" + "="*80)
    print("COMPARISON: Checking if simulations produce identical results...")
    print("="*80 + "\n")
    
    # Extract first path from each simulation for comparison
    sa_path = SA_sims[0]
    exp_sd_path = Exp_SD_sims[0]
    sd_path = SD_sims[0]
    sd2d_path = SD2d_sims[0]
    
    print("1. Comparing SAHawkes vs Exp_SDHawkes:")
    match_1 = compare_paths(sa_path, exp_sd_path)
    print(f"   Result: {'✓ IDENTICAL' if match_1 else '✗ DIFFERENT'}\n")
    
    print("2. Comparing SAHawkes vs SDHawkes:")
    match_2 = compare_paths(sa_path, sd_path)
    print(f"   Result: {'✓ IDENTICAL' if match_2 else '✗ DIFFERENT'}\n")
    
    print("3. Comparing SAHawkes vs 2D_SDHawkes:")
    match_3 = compare_paths(sa_path, sd2d_path)
    print(f"   Result: {'✓ IDENTICAL' if match_3 else '✗ DIFFERENT'}\n")
    
    print("4. Comparing Exp_SDHawkes vs SDHawkes:")
    match_4 = compare_paths(exp_sd_path, sd_path)
    print(f"   Result: {'✓ IDENTICAL' if match_4 else '✗ DIFFERENT'}\n")
    
    print("="*80)
    if all([match_1, match_2, match_3, match_4]):
        print("SUCCESS: All implementations produce IDENTICAL results! ✓")
    else:
        print("WARNING: Implementations produce DIFFERENT results. Debug needed.")
    print("="*80 + "\n")
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ## PART 1b: Multi-Path Cross-Implementation Overlap Check
    
    print("="*80)
    print("PART 1b: MULTI-PATH CROSS-IMPLEMENTATION OVERLAP CHECK")
    print("="*80)
    NUM_PATHS_MULTI = 5
    print(f"\nGenerating {NUM_PATHS_MULTI} paths from each implementation with SeedSequence")
    print(f"Parameters: μ={MU_2D}, α={ALPHA_2D[0]}, β={BETA_2D[0]}")
    print(f"Time horizon: T={T}, Seed: {SEED}\n")
    
    # Spawn reproducible child seeds for multi-path comparison
    ss_multi = np.random.SeedSequence(SEED)
    multi_seeds = ss_multi.spawn(NUM_PATHS_MULTI)
    
    print(f"Running {NUM_PATHS_MULTI} SAHawkes simulations...")
    SA_multi = simulate_SAHawkes(T=T, mu=MU_2D, alpha=ALPHA_2D, beta=BETA_2D, num_paths=NUM_PATHS_MULTI, seeds_list=multi_seeds)
    
    print(f"Running {NUM_PATHS_MULTI} Exp_SDHawkes simulations...")
    Exp_SD_multi = simulate_Exp_SDHawkes(T=T, background_intensity=background_intensity_func_2d, alpha=ALPHA_2D, beta=BETA_2D, r=r_2d, num_paths=NUM_PATHS_MULTI, seeds_list=multi_seeds)
    
    print(f"Running {NUM_PATHS_MULTI} 2D_SDHawkes simulations...")
    SD2d_multi = simulate_2D_SDHawkes(T=T, background_intensity=background_intensity_func_2d, alpha=ALPHA_2D, beta=BETA_2D, r=r_2d, FLLN_scaling=1.0, num_paths=NUM_PATHS_MULTI, max_arrivals=MAX_ARRIVALS_SINGLE, seeds_list=multi_seeds)
    
    print(f"\nAll multi-path simulations complete!\n")
    
    print("Comparing SAHawkes vs Exp_SDHawkes (path-by-path):")
    multi_match_1 = compare_implementations_multi_path(SA_multi, Exp_SD_multi, "SAHawkes", "Exp_SDHawkes")
    
    print("\nComparing SAHawkes vs 2D_SDHawkes (path-by-path):")
    multi_match_2 = compare_implementations_multi_path(SA_multi, SD2d_multi, "SAHawkes", "2D_SDHawkes")
    
    print("\nComparing Exp_SDHawkes vs 2D_SDHawkes (path-by-path):")
    multi_match_3 = compare_implementations_multi_path(Exp_SD_multi, SD2d_multi, "Exp_SDHawkes", "2D_SDHawkes")
    
    print("\n" + "="*80)
    multi_all_match = all([multi_match_1, multi_match_2, multi_match_3])
    if multi_all_match:
        print(f"SUCCESS: All {NUM_PATHS_MULTI} paths match across all implementations! ✓")
    else:
        print(f"WARNING: Some paths differ between implementations. Debug needed.")
    print("="*80 + "\n")
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ## PART 2: Monte Carlo Analytical Comparison
    
    print("="*80)
    print("PART 2: MONTE CARLO ANALYTICAL COMPARISON")
    print("="*80)
    print("\nComparing Monte Carlo simulations against analytical solutions for 1D Hawkes process")
    print(f"Parameters: μ={MU_1D[0]}, α={ALPHA_1D[0,0]}, β={BETA_1D[0,0]}")
    print(f"Number of MC paths: {NUM_MC_PATHS}")
    print(f"Time grid points: {TIME_GRID_POINTS}\n")
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_mc = os.path.join(VALIDATION_DIR, f"validation_run_{timestamp}")
    os.makedirs(output_dir_mc, exist_ok=True)
    print(f"Output directory: {output_dir_mc}\n")
    
    # Create time grid
    time_grid = np.linspace(0, T, TIME_GRID_POINTS)
    
    # Compute analytical values for 1D process (same for all methods)
    analytical_count = analytical_exp_Hawkes_count_mean(time_grid, MU_1D[0], ALPHA_1D[0, 0], BETA_1D[0, 0])
    analytical_intensity = analytical_exp_Hawkes_intensity_mean(time_grid, MU_1D[0], ALPHA_1D[0, 0], BETA_1D[0, 0])
    
    plot_paths = []
    
    # 1. SAHawkes
    print(f"Running {NUM_MC_PATHS} SAHawkes simulations (1D)...")
    sa_sims = simulate_SAHawkes(T=T, mu=MU_1D, alpha=ALPHA_1D, beta=BETA_1D, num_paths=NUM_MC_PATHS)
    sa_mean_count, sa_mean_intensity = process_simulations(sa_sims, MU_1D, ALPHA_1D, BETA_1D, time_grid)
    plot_paths.append(plot_count_comparison(time_grid, sa_mean_count, analytical_count, "SAHawkes", output_dir_mc, NUM_MC_PATHS))
    plot_paths.append(plot_intensity_comparison(time_grid, sa_mean_intensity, analytical_intensity, "SAHawkes", output_dir_mc, NUM_MC_PATHS))
    
    # 2. Exp_SDHawkes
    print(f"Running {NUM_MC_PATHS} Exp_SDHawkes simulations (1D)...")
    exp_sd_sims = simulate_Exp_SDHawkes(T=T, background_intensity=background_intensity_func_1d, alpha=ALPHA_1D, beta=BETA_1D, r=r_1d, num_paths=NUM_MC_PATHS)
    exp_sd_mean_count, exp_sd_mean_intensity = process_simulations(exp_sd_sims, MU_1D, ALPHA_1D, BETA_1D, time_grid)
    plot_paths.append(plot_count_comparison(time_grid, exp_sd_mean_count, analytical_count, "Exp_SDHawkes", output_dir_mc, NUM_MC_PATHS))
    plot_paths.append(plot_intensity_comparison(time_grid, exp_sd_mean_intensity, analytical_intensity, "Exp_SDHawkes", output_dir_mc, NUM_MC_PATHS))
    
    # 3. SDHawkes
    print(f"Running {NUM_MC_PATHS} SDHawkes simulations (1D)...")
    sd_sims = simulate_SDHawkes(T=T, mu=MU_1D, alpha=ALPHA_1D, beta=BETA_1D, num_paths=NUM_MC_PATHS)
    sd_mean_count, sd_mean_intensity = process_simulations(sd_sims, MU_1D, ALPHA_1D, BETA_1D, time_grid)
    plot_paths.append(plot_count_comparison(time_grid, sd_mean_count, analytical_count, "SDHawkes", output_dir_mc, NUM_MC_PATHS))
    plot_paths.append(plot_intensity_comparison(time_grid, sd_mean_intensity, analytical_intensity, "SDHawkes", output_dir_mc, NUM_MC_PATHS))
    
    # Note: 2D_SDHawkes is skipped for 1D validation as it's hardcoded for 2D processes
    
    print(f"\nMonte-Carlo Analytical Comparison complete!")
    print(f"Number of methods validated: 3 (SAHawkes, Exp_SDHawkes, SDHawkes)")
    print(f"Plots saved to: {output_dir_mc}")
    for plot_path in plot_paths:
        print(f"  - {os.path.basename(plot_path)}")
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ## PART 3: Statistical Analysis of Maximum Differences
    
    print("\n" + "="*80)
    print("PART 3: STATISTICAL ANALYSIS - Maximum Difference Statistics")
    print("="*80)
    print(f"\nRunning {NUM_MC_RUNS} independent MC experiments (each with {NUM_MC_PATHS} paths)...")
    print("This may take a while...\n")
    
    # Storage for max differences across runs
    sa_max_diffs_count = []
    sa_max_diffs_intensity = []
    exp_sd_max_diffs_count = []
    exp_sd_max_diffs_intensity = []
    
    for run_idx in trange(NUM_MC_RUNS, desc="Running MC simulations", unit="runs", ncols=80):
        
        # SAHawkes
        sa_sims_run = simulate_SAHawkes(T=T, mu=MU_1D, alpha=ALPHA_1D, beta=BETA_1D, num_paths=NUM_MC_PATHS)
        sa_mean_count_run, sa_mean_intensity_run = process_simulations(sa_sims_run, MU_1D, ALPHA_1D, BETA_1D, time_grid)
        sa_max_diffs_count.append(np.max(np.abs(sa_mean_count_run - analytical_count)))
        sa_max_diffs_intensity.append(np.max(np.abs(sa_mean_intensity_run - analytical_intensity)))
        
        # Exp_SDHawkes
        exp_sd_sims_run = simulate_Exp_SDHawkes(T=T, background_intensity=background_intensity_func_1d, 
                                                 alpha=ALPHA_1D, beta=BETA_1D, r=r_1d, num_paths=NUM_MC_PATHS)
        exp_sd_mean_count_run, exp_sd_mean_intensity_run = process_simulations(exp_sd_sims_run, MU_1D, ALPHA_1D, BETA_1D, time_grid)
        exp_sd_max_diffs_count.append(np.max(np.abs(exp_sd_mean_count_run - analytical_count)))
        exp_sd_max_diffs_intensity.append(np.max(np.abs(exp_sd_mean_intensity_run - analytical_intensity)))
    
    # Convert to arrays for statistics
    sa_max_diffs_count = np.array(sa_max_diffs_count)
    sa_max_diffs_intensity = np.array(sa_max_diffs_intensity)
    exp_sd_max_diffs_count = np.array(exp_sd_max_diffs_count)
    exp_sd_max_diffs_intensity = np.array(exp_sd_max_diffs_intensity)
    
    # Compute and display statistics
    print(f"\nCompleted {NUM_MC_RUNS} runs!\n")
    print("="*80)
    print("RESULTS: SAHawkes")
    print("="*80)
    print(f"Count - Mean Max Diff: {np.mean(sa_max_diffs_count):.6f}")
    print(f"Count - Variance Max Diff: {np.var(sa_max_diffs_count):.6e}")
    print(f"Count - Std Dev Max Diff: {np.std(sa_max_diffs_count):.6e}")
    print(f"\nIntensity - Mean Max Diff: {np.mean(sa_max_diffs_intensity):.6f}")
    print(f"Intensity - Variance Max Diff: {np.var(sa_max_diffs_intensity):.6e}")
    print(f"Intensity - Std Dev Max Diff: {np.std(sa_max_diffs_intensity):.6e}")
    
    print("\n" + "="*80)
    print("RESULTS: Exp_SDHawkes")
    print("="*80)
    print(f"Count - Mean Max Diff: {np.mean(exp_sd_max_diffs_count):.6f}")
    print(f"Count - Variance Max Diff: {np.var(exp_sd_max_diffs_count):.6e}")
    print(f"Count - Std Dev Max Diff: {np.std(exp_sd_max_diffs_count):.6e}")
    print(f"\nIntensity - Mean Max Diff: {np.mean(exp_sd_max_diffs_intensity):.6f}")
    print(f"Intensity - Variance Max Diff: {np.var(exp_sd_max_diffs_intensity):.6e}")
    print(f"Intensity - Std Dev Max Diff: {np.std(exp_sd_max_diffs_intensity):.6e}")
    
    # Save statistics to file
    stats_file = os.path.join(output_dir_mc, "max_difference_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MAXIMUM DIFFERENCE STATISTICS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Number of MC runs: {NUM_MC_RUNS}\n")
        f.write(f"Paths per MC run: {NUM_MC_PATHS}\n")
        f.write(f"Total time: T = {T}\n")
        f.write(f"Time grid points: {len(time_grid)}\n\n")
        f.write(f"Parameters: μ={MU_1D[0]}, α={ALPHA_1D[0,0]}, β={BETA_1D[0,0]}\n\n")
        
        f.write("="*70 + "\n")
        f.write("SAHawkes\n")
        f.write("="*70 + "\n")
        f.write(f"Count - Mean Max Diff: {np.mean(sa_max_diffs_count):.6f}\n")
        f.write(f"Count - Variance Max Diff: {np.var(sa_max_diffs_count):.6e}\n")
        f.write(f"Count - Std Dev Max Diff: {np.std(sa_max_diffs_count):.6e}\n")
        f.write(f"Count - Min Max Diff: {np.min(sa_max_diffs_count):.6f}\n")
        f.write(f"Count - Max Max Diff: {np.max(sa_max_diffs_count):.6f}\n\n")
        
        f.write(f"Intensity - Mean Max Diff: {np.mean(sa_max_diffs_intensity):.6f}\n")
        f.write(f"Intensity - Variance Max Diff: {np.var(sa_max_diffs_intensity):.6e}\n")
        f.write(f"Intensity - Std Dev Max Diff: {np.std(sa_max_diffs_intensity):.6e}\n")
        f.write(f"Intensity - Min Max Diff: {np.min(sa_max_diffs_intensity):.6f}\n")
        f.write(f"Intensity - Max Max Diff: {np.max(sa_max_diffs_intensity):.6f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("Exp_SDHawkes\n")
        f.write("="*70 + "\n")
        f.write(f"Count - Mean Max Diff: {np.mean(exp_sd_max_diffs_count):.6f}\n")
        f.write(f"Count - Variance Max Diff: {np.var(exp_sd_max_diffs_count):.6e}\n")
        f.write(f"Count - Std Dev Max Diff: {np.std(exp_sd_max_diffs_count):.6e}\n")
        f.write(f"Count - Min Max Diff: {np.min(exp_sd_max_diffs_count):.6f}\n")
        f.write(f"Count - Max Max Diff: {np.max(exp_sd_max_diffs_count):.6f}\n\n")
        
        f.write(f"Intensity - Mean Max Diff: {np.mean(exp_sd_max_diffs_intensity):.6f}\n")
        f.write(f"Intensity - Variance Max Diff: {np.var(exp_sd_max_diffs_intensity):.6e}\n")
        f.write(f"Intensity - Std Dev Max Diff: {np.std(exp_sd_max_diffs_intensity):.6e}\n")
        f.write(f"Intensity - Min Max Diff: {np.min(exp_sd_max_diffs_intensity):.6f}\n")
        f.write(f"Intensity - Max Max Diff: {np.max(exp_sd_max_diffs_intensity):.6f}\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"\nStatistics saved to: {stats_file}")
    print("="*80)
    
    print("\n" + "="*80)
    print("VALIDATION SUITE COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir_mc}")
    print("\nSummary:")
    print(f"  - Single-path comparison: {'✓ PASSED' if all([match_1, match_2, match_3, match_4]) else '✗ FAILED'}")
    print(f"  - Multi-path overlap ({NUM_PATHS_MULTI} paths): {'✓ PASSED' if multi_all_match else '✗ FAILED'}")
    print(f"  - Monte Carlo plots: {len(plot_paths)} plots generated")
    print(f"  - Statistical analysis: {NUM_MC_RUNS} runs completed")
    print("="*80)
