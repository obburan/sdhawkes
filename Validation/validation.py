import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from StateAgnostic_Hawkes_class import SAHawkes
from sdhawkes import SDHawkes, Exp_SDHawkes
import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import Optional, Tuple, List, Callable, Union

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Monte Carlo vs. Analytical quantities tools

def analytical_exp_Hawkes_count_mean(t: float, mu: float, alpha: float, beta: float) -> float:

    """
    The analytical mean of the count process of an exponential Hawkes process is given by
    E[N_t] = μ / (α - β)^2 * (α * (e^(-(β - α) * t) - 1) - β * t * (α - β))
    """
    
    return mu / (alpha - beta)**2 * (alpha * (np.exp(-(beta - alpha) * t) - 1) - beta * t * (alpha - beta))


def analytical_exp_Hawkes_intensity_mean(t: float, mu: float, alpha: float, beta: float) -> float:

    """
    The analytical mean of the mean intensity of an exponential Hawkes process is given by
    E[λ(t)] = μ/(α - β) * (α e^{(α - β)t} - β)
    """
    
    return mu / (alpha - beta) * (alpha * np.exp((alpha - beta) * t) - beta)

def average_paths(paths_list: List[np.ndarray]) -> np.ndarray:
    """
    Average a list of paths (e.g., intensity or count trajectories).
    """
    num_paths = len(paths_list)
    if num_paths == 0:
        raise ValueError("The list of paths is empty.")
    
    # Ensure all paths have the same length
    path_length = len(paths_list[0])
    for path in paths_list:
        if len(path) != path_length:
            raise ValueError("All paths must have the same length.")
    
    # Stack paths and compute mean
    stacked_paths = np.array(paths_list)
    return np.mean(stacked_paths, axis=0)


def compare_paths(sim_A: Tuple, sim_B: Tuple, tol: float = 1e-10) -> bool:
    """
    Compare two path lists for exact equivalence (or near-equivalence within tolerance).
    
    Parameters:
    -----------
    sim_A, sim_B : tuple or list
        Either (paths, full_information) tuples or just paths lists
    tol : float, optional
        Tolerance for floating point comparison (default: 1e-10)
    
    Returns:
    --------
    bool : True if paths are identical (within tolerance), False otherwise
    """
    # Handle different return formats
    if isinstance(sim_A, tuple) and len(sim_A) == 2:
        paths_A, info_A = sim_A
    if isinstance(sim_B, tuple) and len(sim_B) == 2:
        paths_B, info_B = sim_B
    
    # Check if same number of dimensions
    if len(paths_A) != len(paths_B):
        print(f"Different number of dimensions: {len(paths_A)} vs {len(paths_B)}")
        return False
    
    # Check each dimension
    for d in range(len(paths_A)):
        if len(paths_A[d]) != len(paths_B[d]):
            print(f"Dimension {d}: different number of arrivals: {len(paths_A[d])} vs {len(paths_B[d])}")
            return False
        
        # Check arrival times
        for k, (t1, t2) in enumerate(zip(paths_A[d], paths_B[d])):
            if abs(t1 - t2) > tol:
                print(f"Dimension {d}, arrival {k}: times differ: {t1} vs {t2} (diff = {abs(t1-t2)})")
                return False
    
    return True

def reconstruct_intensity_from_path(full_information: List[Tuple], mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the intensity process from a single simulated path using the closed-form exponential excitation kernel: φ_{ij}(t) = α_{ij} * exp(-β_{ij} * t).
    
    Parameters:
    -----------
    full_information : list of tuples
        List of (t, d) or (t, d, state) tuples for each arrival.
        Only the arrival time t and dimension d are used.
    mu : np.ndarray, shape (dim,)
        Background intensity vector
    alpha : np.ndarray, shape (dim, dim)
        Excitation matrix
    beta : np.ndarray, shape (dim, dim)
        Decay rate matrix
    time_grid : np.ndarray, shape (n_grid,)
        Time points at which to evaluate intensity
    
    Returns:
    --------
    intensity_array : np.ndarray, shape (n_grid, dim)
        Reconstructed intensity at each grid point for each dimension.
        intensity_array[k, d] = intensity of dimension d at time time_grid[k]
    """
    dim = len(mu)
    n_grid = len(time_grid)
    intensity_array = np.tile(mu, (n_grid, 1))  # Initialize with background intensity
    
    for arrival in full_information:
        t_arr = arrival[0]
        d_arr = int(arrival[1])
        
        # Add excitation contribution to all grid points after this arrival
        mask = time_grid > t_arr
        if not np.any(mask):
            continue
        dt = time_grid[mask] - t_arr
        
        # Excitation from arrival in dimension d_arr to all dimensions i:
        # φ_{i, d_arr}(t) = α[i, d_arr] * exp(-β[i, d_arr] * t)
        for i in range(dim):
            intensity_array[mask, i] += alpha[i, d_arr] * np.exp(-beta[i, d_arr] * dt)
    
    return intensity_array


def compute_count_process(paths: List[List[float]], time_grid: np.ndarray) -> np.ndarray:
    """
    Compute the count process N(t) on a time grid from simulated paths.
    
    Parameters:
    -----------
    paths : list of lists
        paths[d] contains sorted arrival times for dimension d
    time_grid : np.ndarray, shape (n_grid,)
        Time points at which to evaluate counts
    
    Returns:
    --------
    count_array : np.ndarray, shape (n_grid, dim)
        Count process at each grid point for each dimension.
        count_array[k, d] = number of arrivals in dimension d up to time time_grid[k]
    """
    dim = len(paths)
    n_grid = len(time_grid)
    count_array = np.zeros((n_grid, dim))
    
    for d in range(dim):
        if len(paths[d]) > 0:
            arrival_times_array = np.array(sorted(paths[d]))
            count_array[:, d] = np.searchsorted(arrival_times_array, time_grid, side='right')
    
    return count_array


def plot_count_comparison(time_grid: np.ndarray, mc_mean_count: np.ndarray, analytical_count: np.ndarray, method_name: str, output_dir: str, num_mc_paths: int) -> str:
    """
    Plot Monte Carlo mean count vs analytical expected count.
    
    Parameters:
    -----------
    time_grid : np.ndarray
        Time points for evaluation
    mc_mean_count : np.ndarray, shape (n_grid,)
        Monte Carlo mean count (1D)
    analytical_count : np.ndarray, shape (n_grid,)
        Analytical expected count
    method_name : str
        Name of simulation method (e.g., 'SAHawkes', 'Exp_SDHawkes')
    output_dir : str
        Directory to save plot
    num_mc_paths : int
        Number of Monte Carlo paths used
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(time_grid, mc_mean_count, label=f"MC Mean Count (N={num_mc_paths})", color='blue', linewidth=2)
    ax.plot(time_grid, analytical_count, label="Analytical E[N(t)]", color='red', linestyle='--', linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    ax.set_title(f"1D Hawkes Process: Count Comparison ({method_name})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Calculate and display maximum difference
    max_diff = np.max(np.abs(mc_mean_count - analytical_count))
    textstr = f'Max Diff: {max_diff:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"count_comparison_{method_name}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def plot_intensity_comparison(time_grid: np.ndarray, mc_mean_intensity: np.ndarray, analytical_intensity: np.ndarray, method_name: str, output_dir: str, num_mc_paths: int) -> str:
    """
    Plot Monte Carlo mean intensity vs analytical expected intensity.
    
    Parameters:
    -----------
    time_grid : np.ndarray
        Time points for evaluation
    mc_mean_intensity : np.ndarray, shape (n_grid,)
        Monte Carlo mean intensity (1D)
    analytical_intensity : np.ndarray, shape (n_grid,)
        Analytical expected intensity
    method_name : str
        Name of simulation method (e.g., 'SAHawkes', 'Exp_SDHawkes')
    output_dir : str
        Directory to save plot
    num_mc_paths : int
        Number of Monte Carlo paths used
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(time_grid, mc_mean_intensity, label=f"MC Mean Intensity (N={num_mc_paths})", color='blue', linewidth=2)
    ax.plot(time_grid, analytical_intensity, label="Analytical E[λ(t)]", color='red', linestyle='--', linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Intensity")
    ax.set_title(f"1D Hawkes Process: Intensity Comparison ({method_name})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Calculate and display maximum difference
    max_diff = np.max(np.abs(mc_mean_intensity - analytical_intensity))
    textstr = f'Max Diff: {max_diff:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"intensity_comparison_{method_name}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Simulation Functions

def simulate_SAHawkes(T: float, mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray, num_paths: int, output_file: Optional[str] = None, seeds_list: Optional[List] = None) -> List:
    """
    Simulate multiple paths of a state-agnostic Hawkes process.
    
    Wrapper around SAHawkes class for validation testing. Runs multiple independent
    simulations with optional seeding for reproducibility.
    
    Parameters
    ----------
    T : float
        Simulation time horizon [0, T].
    mu : np.ndarray
        Background intensity vector of shape (dim,).
    alpha : np.ndarray
        Excitation matrix of shape (dim, dim).
    beta : np.ndarray
        Decay rate matrix of shape (dim, dim).
    num_paths : int
        Number of independent simulation paths to generate.
    output_file : str, optional
        Base path for output files. If provided, saves to disk with suffixes _0, _1, etc.
        If None, returns results in memory. Default is None.
    seeds_list : list of SeedSequence, optional
        List of random seeds, one per path. If None, uses OS entropy. Default is None.
    
    Returns
    -------
    list
        List of simulation results, one per path.
        Each element is either a file path (str) if output_file provided, or a tuple (paths, full_information) if in-memory.
    """
    use_disk = output_file is not None
    obj = SAHawkes(mu, alpha, beta, T, max_arrivals=1000000, use_disk=use_disk)
    sims = [[] for _ in range(num_paths)]
    for i in range(num_paths):
        seed_i = seeds_list[i] if seeds_list is not None else None
        if use_disk:
            sims[i] = obj.simulate_path(output_file=f"{output_file}_{i}", seed=seed_i)
        else:
            sims[i] = obj.simulate_path(seed=seed_i)
    
    return sims

def simulate_Exp_SDHawkes(T: float, background_intensity: Callable, alpha: np.ndarray, beta: np.ndarray, r: Callable, num_paths: int, output_file: Optional[str] = None, base_seed = None) -> List:
    """
    Simulate multiple paths using Exp_SDHawkes (efficient exponential kernel implementation).
    
    For validation testing, configured as state-agnostic: r(i,j,y) = 1 and state_matrix = 0. This allows direct comparison with SAHawkes and general SDHawkes implementations.
    
    Parameters
    ----------
    T : float
        Simulation time horizon [0, T].
    background_intensity : Callable
        Function with signature (t, state) -> np.ndarray returning intensity vector.
    alpha : np.ndarray
        Excitation matrix of shape (dim, dim).
    beta : np.ndarray
        Decay rate matrix of shape (dim, dim).
    r : Callable
        State-dependence function (i, j, state) -> float. For validation, returns 1.
    num_paths : int
        Number of independent simulation paths to generate.
    output_file : str, optional
        Base path for output files. If None, returns in memory. Default is None.
    seeds_list : list of SeedSequence, optional
        List of random seeds, one per path. Default is None.
    
    Returns
    -------
    list
        List of simulation results (file paths or (paths, full_information) tuples).
    """
    dim = len(background_intensity(0,0))
    
    # No state changes in state-agnostic case
    state_matrix = np.zeros((1, dim))
    
    # Since mu is constant, background_intensity_max = 0
    background_intensity_max = 0.0
    
    use_disk = output_file is not None
    obj = Exp_SDHawkes(
        background_intensity_func=background_intensity,
        background_intensity_max=background_intensity_max,
        state_matrix=state_matrix,
        alpha=alpha,
        beta=beta,
        r=r,
        max_arrivals=500000,
        num_workers=4,
        use_disk=use_disk
    )
    
    raw_results = obj.sim(
        T=T,
        FLLN_scaling=1.0,
        num_paths=num_paths,
        output_dir=output_file if use_disk else None,
        external_info={"alpha": alpha.tolist(), "beta": beta.tolist(), "mu": background_intensity(0, 0).tolist()},
        base_seed=base_seed
    )
    
    if use_disk:
        return raw_results  # (results_list, subfolder_path)
    return [obj.build_paths_list(*triple) for triple in raw_results]

def simulate_SDHawkes(T: float, mu: np.ndarray, alpha: np.ndarray, beta: np.ndarray, num_paths: int, output_file: Optional[str] = None, base_seed: Optional[int] = None) -> List:
    """
    Simulate multiple paths using general SDHawkes class with manually-coded exponential kernel.
    
    For validation testing, configured as state-agnostic with manually-implemented exponential excitation kernel. This tests the general simulation framework.
    
    Parameters
    ----------
    T : float
        Simulation time horizon [0, T].
    mu : np.ndarray
        Background intensity vector of shape (dim,).
    alpha : np.ndarray
        Excitation matrix of shape (dim, dim).
    beta : np.ndarray
        Decay rate matrix of shape (dim, dim).
    num_paths : int
        Number of independent simulation paths to generate.
    output_file : str, optional
        Base path for output files. Default is None.
    seeds_list : list of SeedSequence, optional
        List of random seeds, one per path. Default is None.
    
    Returns
    -------
    list
        List of simulation results (file paths or (paths, full_information) tuples).
    
    Notes
    -----
    Uses manually-coded exponential kernel to test general simulation framework.
    """
    dim = len(mu)
    
    # Constant background intensity (state-agnostic)
    def background_intensity_func(t, state):
        return mu
    
    # Manually implement exponential excitation kernel
    def excitation_kernel_func(time_diffs, past_dims, past_states):
        """
        Compute excitation matrix from past arrivals.
        For exponential kernel: φ_ij(t) = α_ij * exp(-β_ij * t)
        
        Returns:
        --------
        excitation_matrix : np.ndarray, shape (dim, dim)
            Entry (i,j) is total excitation to dimension i from all past arrivals in dimension j
        """
        excitation_matrix = np.zeros((dim, dim))
        
        for k, (dt, j) in enumerate(zip(time_diffs, past_dims)):
            # Contribution from arrival k (dimension j) to all dimensions i
            for i in range(dim):
                excitation_matrix[i, j] += alpha[i, j] * np.exp(-beta[i, j] * dt)
        
        return excitation_matrix
    
    # No state changes in state-agnostic case
    state_matrix = np.zeros((1, dim))
    
    # Since mu is constant, background_intensity_max may be set to 0 while still resulting in the correct calculation of Max_intensity in sim.py circa line 300
    background_intensity_max = 0.0
    
    use_disk = output_file is not None
    obj = SDHawkes(
        background_intensity_func=background_intensity_func,
        excitation_kernel_func=excitation_kernel_func,
        background_intensity_max=background_intensity_max,
        state_matrix=state_matrix,
        max_arrivals=500000,
        num_workers=4,
        use_disk=use_disk
    )
    
    raw_results = obj.sim(
        T=T,
        FLLN_scaling=1.0,
        num_paths=num_paths,
        output_dir=output_file if use_disk else None,
        external_info={"alpha": alpha.tolist(), "beta": beta.tolist(), "mu": mu.tolist()},
        base_seed=base_seed
    )
    
    if use_disk:
        return raw_results  # (results_list, subfolder_path)
    return [obj.build_paths_list(*triple) for triple in raw_results]

# We copy over from SDHawkes_2d_sim.py the simulation function due to the use of global dependencies within the 2d simulation script
def Hawkes_2d_sim(T: float, background_intensity: Callable, alpha: np.ndarray, beta: np.ndarray, r: Callable, max_arrivals: int, FLLN_scaling: float, output_file: Optional[str] = None, seed: Optional[Union[int, np.random.SeedSequence]] = None) -> Union[str, Tuple[List, List]]:
    """
    Simulate a single path of a state-dependent Hawkes process up to time FLLN_scaling * T using Ogata's modified thinning algorithm.
    
    Parameters:
    -----------
    T : float
        Simulate on interval [0,T]
    FLLN_scaling : float
        Scaling parameter for FLLN. Affects state-dependency via r(y/n, i)
    output_file : str, optional
        Path to save simulation output. We use an output file to reduce memory usage while larger programs calling this function run.
        It is recommended to use an output file for large simulations or nearly unstable simulations, since these cause a lot of arrivals. Otherwise,
        it is always easier to use the returned list objects downstream, as otherwise one needs to re-build them form the output file.
        
    Returns:
    --------
    if output_file = None:
        paths : list
            List of 2 lists, where paths[0] contains arrival times for dimension 1 and paths[1] contains arrival times for dimension 2.
            Format: [[t1, t2, ...], [t1, t2, ...]]
        full_information : list
            List of tuples (t, d, state) for each arrival where:
            - t (float): arrival time
            - d (int): dimension index (0 or 1)
            - state (int): state value N_1 - N_2 at time t
            Format: [(t, d, state), (t, d, state), ...]
    else if output_file is not None:
        Returns output_file (the path to the output file), but saves simulation output to output_file.
    
    Global Dependencies:
    --------------------
    alpha : np.ndarray
        2x2 excitation parameter matrix
    beta : np.ndarray
        2x2 decay parameter matrix
    mu : np.ndarray
        2D background intensity vector
    max_arrivals : int
        Maximum number of arrivals before terminating simulation
    """

    dim = 2
    ones_vec = np.ones(dim)
    rng = np.random.default_rng(seed)

    ## Initialization
    paths = [[] for i in range(dim)] # paths[i] will be a list of arrival times for Hawkes process N_i
    full_information = [] # will be full information, with entries (t,d,state) for each arrival t, corresponding label d, and corresponding state value at time t
    state_dict = {str(0): 0}  # Dictionary to track states at each arrival time, initialized with state 0 at time 0
    # Iteratively updted variables
    t = 0
    current_state = 0
    num_arrivals_so_far = 0 # We cut the simulation off after enough arrivals
    # Self-excitation terms
    excitation_matrix = np.zeros((2, 2))
    excitation_vec = excitation_matrix @ ones_vec
    # Cumulative intensity
    current_intensity_vec = background_intensity(t, current_state/FLLN_scaling) # intensity has no excitation before the first arrival  # NOTE: this has been changed from the 2-d simulation just to make it use the same format as sim.py where the FLLN scaling is passed with the state

    while (t < T) and (num_arrivals_so_far < max_arrivals*FLLN_scaling): #FLLN scaling because we simulate on an extended horizon when FLLN > 1.

        # Update what the previous intensity vector was        
        previous_intensity_vec = current_intensity_vec

        # Upper bound on intensity: current intensity (because we assume a constant background intensity and decaying excitation terms)
        Max_intensity = np.sum(previous_intensity_vec)
        
        t_old = t
        t += rng.exponential(1/Max_intensity)
        U = rng.uniform(0,Max_intensity)

        # Calculate new intensities as time progresses
        # Update intensities: decay old excitation terms
        time_diff = t - t_old
        excitation_matrix *= np.exp(-beta * time_diff) # decay excitation terms
        excitation_vec = excitation_matrix @ ones_vec # sum rows to get total intensity for each dimension
        # Update background intensity due to time progression
        background = background_intensity(t,current_state/FLLN_scaling) # NOTE: this has been changed from the 2-d simulation just to make it use the same format as sim.py where the FLLN scaling is passed with the state
        current_intensity_vec = excitation_vec + background # total intensity for each dimension

        if (t < T) and (U <= np.sum(current_intensity_vec)):
            TF_array = [U <= sum(current_intensity_vec[:i+1]) for i in range(dim)] # divide current intensity value into bins to figure out which dimension the new arrival belongs to
            d = indexOfFirstOne(TF_array)
            # double check d makes sense
            if d < 0:
                raise ValueError("Error: d should not be negative")
            elif d >= 2:
                raise ValueError("Error: d should not be >= 2")
            
            # update state while storing all previous states in array
            arrival_type = [1,-1][d] # if d==0, yields 1, if d==1, yields -1
            previous_state = current_state
            current_state = previous_state + arrival_type

            paths[d].append(t)
            state_dict[str(t)] = previous_state  # Store the state at this arrival time
            full_information.append((t, d, previous_state))
            
            num_arrivals_so_far += 1

            ## Update current intensity vec to value just after arrival
            # Add new jump to Excitation with state dependence via r function
            r_vec = np.array([r(i, d, previous_state / FLLN_scaling) for i in range(dim)])
            excitation_matrix[:, d] += alpha[:, d] * r_vec # Add new jump in column d
            excitation_vec = excitation_matrix @ ones_vec # sum rows to get total intensity for each dimension
            # Calculate total intensity for each dimension
            current_intensity_vec = excitation_vec + background
        
    if output_file:
        with open(output_file, 'wb') as f:
            pickle.dump((paths, full_information), f)
        return output_file
    else:
        return paths, full_information



def simulate_2D_SDHawkes(T: float, background_intensity: Callable, alpha: np.ndarray, beta: np.ndarray, r: Callable, FLLN_scaling: float, num_paths: int, max_arrivals: int, output_file: Optional[str] = None, seeds_list: Optional[List] = None) -> List:
    """
    Simulate multiple paths of 2D state-dependent Hawkes process using specialized 2D implementation.
    
    Parameters:
    -----------
    T : float
        Final simulation time
    mu : np.ndarray, shape (2,)
        Background intensity vector
    alpha : np.ndarray, shape (2, 2)
        Excitation matrix
    beta : np.ndarray, shape (2, 2)
        Decay rate matrix
    delta : float
        State dependence parameter
    FLLN_scaling : float
        FLLN scaling parameter n
    num_paths : int
        Number of simulation paths
    output_file : str, optional
        Base name for output files
    """
    
    sims = [[] for i in range(num_paths)]

    for i in range(num_paths):
        seed_i = seeds_list[i] if seeds_list is not None else None
        if output_file is not None:
            sims[i] = Hawkes_2d_sim(
                T=T,
                background_intensity=background_intensity,
                alpha=alpha,
                beta=beta,
                r=r,
                max_arrivals=max_arrivals,
                FLLN_scaling=FLLN_scaling,
                output_file=f"{output_file}_{i}",
                seed=seed_i
            )
        else:
            sims[i] = Hawkes_2d_sim(
                T=T,
                background_intensity=background_intensity,
                alpha=alpha,
                beta=beta,
                r=r,
                max_arrivals=max_arrivals,
                FLLN_scaling=FLLN_scaling,
                seed=seed_i
            )
    
    return sims

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Helper functions

def check_paths_all_unique(sims_list: List[Tuple], num_arrivals_to_check: int = 20, tol: float = 1e-10) -> List[Tuple[int, int]]:
    """
    Check if any two simulation paths in sims_list have identical initial arrival times.
    Used to detect seeding issues in parallel simulations.
    
    Parameters:
    -----------
    sims_list : list
        List of simulation results in (paths, full_information) format.
        Each element is a tuple (paths, full_information) where:
            - paths: list of lists of arrival times per dimension
            - full_information: list of (t, d, ...) tuples
    num_arrivals_to_check : int
        Number of initial arrivals to compare per path
    tol : float
        Tolerance for floating point comparison
        
    Returns:
    --------
    list of tuples
        List of (i, j) pairs where paths i and j are duplicates.
        Empty list means all paths are unique.
    """
    num_paths = len(sims_list)
    duplicates_list = []
    
    # Extract flat sorted arrival times from each simulation
    all_times_list = []
    for sim in sims_list:
        paths, full_info = sim
        # Extract times from full_information (already sorted by arrival order)
        times = np.array([entry[0] for entry in full_info[:num_arrivals_to_check]])
        all_times_list.append(times)
    
    for i in range(num_paths):
        for j in range(i + 1, num_paths):
            if len(all_times_list[i]) > 0 and len(all_times_list[j]) > 0:
                n = min(len(all_times_list[i]), len(all_times_list[j]))
                if np.allclose(all_times_list[i][:n], all_times_list[j][:n], atol=tol):
                    duplicates_list.append((i, j))
    
    return duplicates_list


def compare_implementations_multi_path(sims_A: List, sims_B: List, name_A: str, name_B: str, num_paths_to_compare: Optional[int] = None, tol: float = 1e-10) -> bool:
    """
    Compare multiple simulation paths between two implementations to verify they produce identical results when run with the same seed.
    
    Parameters:
    -----------
    sims_A : list
        List of simulation results from implementation A, each in (paths, full_information) format.
    sims_B : list
        List of simulation results from implementation B, each in (paths, full_information) format.
    name_A : str
        Name of implementation A (for display)
    name_B : str
        Name of implementation B (for display)
    num_paths_to_compare : int, optional
        Number of paths to compare. If None, compares all available paths (min of len(sims_A), len(sims_B)).
    tol : float
        Tolerance for floating point comparison
        
    Returns:
    --------
    bool
        True if all compared paths are identical, False otherwise
    """
    if num_paths_to_compare is None:
        num_paths_to_compare = min(len(sims_A), len(sims_B))
    
    all_match = True
    for k in range(num_paths_to_compare):
        match = compare_paths(sims_A[k], sims_B[k], tol=tol)
        if not match:
            print(f"  Path {k}: {name_A} vs {name_B} -> DIFFERENT")
            all_match = False
        else:
            print(f"  Path {k}: {name_A} vs {name_B} -> IDENTICAL")
    
    return all_match


def indexOfFirstOne(arr: np.ndarray) -> int:
    """
    Find the index of the first occurrence of 1 in a boolean array.
    
    Parameters:
    -----------
    arr : array-like
        Boolean or binary array to search
        
    Returns:
    --------
    int
        Index of first 1 in array, or -1 if no 1 is found
    """
    idx = np.argmax(arr)

    return idx if arr[idx] else -1
