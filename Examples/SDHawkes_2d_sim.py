import numpy as np
from typing import Callable, Optional, Union, Tuple, List
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import os
import sys
from functools import partial
import pickle
import glob
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

"""
Simulation code for 2D Hawkes process with state-dependent intensity. Excitation kernels are exponential. State process is given by difference of components of the count process.

NOTE: Current implementation assumes constant background intensity and non-increasing excitation terms (these are STRICT assumptions for the current implementation). 

WARNING: Unstable parameter choices, long time horizons, and/or large FLLN scaling factors can result in very large simulation files (~2GB per path).

"""

from SDHawkes_2d_config import *

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Simulation helper functions

def r(current_state: float, i: int, FLLN_scaling: float) -> float:
    """ 
    State-dependent multiplicative factor for excitation intensity. Can amplify or dampen self- and cross-excitation effects.
    
    Captures how the state (difference N_1 - N_2) amplifies or dampens excitation effects to promote mean-reverting behavior toward state = 0.
    
    Parameters
    ----------
    current_state : float
        Current state value y = N_1 - N_2 (difference of arrival counts).
    i : int
        Intensity component (row) index. Use 1 for dimension 0, 2 for dimension 1. (i.e., is 1-indexed, not 0-indexed).
        NOT 0-indexed to match mathematical notation.
    FLLN_scaling : float
        FLLN scaling parameter n. State is rescaled as y/n: r(y/n, i).
    
    Returns
    -------
    float
        Multiplicative amplification/dampening factor.
        - Positive state (more dim-0 arrivals) -> increases dim-1 intensity, decreases dim-0
        - Negative state (more dim-1 arrivals) -> increases dim-0 intensity, decreases dim-1
        This promotes balancing toward state = 0.
        
    Global Dependencies
    -------------------
    delta : float
        State-dependency parameter controlling amplification strength
    """
    global delta, MAXIMUM_STATE_CUTOFF

    # Add maximum state cutoff to ensure satisfies finiteness assumptions:
    if current_state > FLLN_scaling*MAXIMUM_STATE_CUTOFF:
        current_state = FLLN_scaling*MAXIMUM_STATE_CUTOFF
    elif current_state < -FLLN_scaling*MAXIMUM_STATE_CUTOFF:
        current_state = -FLLN_scaling*MAXIMUM_STATE_CUTOFF

    if isinstance(current_state, np.ndarray):
        if current_state.shape != (1,):
            print("ERROR: current_state is not currently a scalar or 1D array")
        current_state = current_state[0]
    if np.abs(current_state) < 1e-10:
        return 1
    if i == 1:
        return (1+delta)** (-1*current_state / FLLN_scaling)
    elif i == 2:
        return (1+delta)** (current_state / FLLN_scaling)
    else:
        raise ValueError("Invalid state or index")

def background_intensity(t: float, y: float, FLLN_scaling: float) -> np.ndarray:
    """
    Calculate the background intensity vector for a given time and state. 
    Currently written to just return the constant background intensity vector.
    
    Parameters:
    -----------
    t : float
        Current time
    y : float
        Current state
    FLLN_scaling : float
        Scaling factor for FLLN.
 
    Returns:
    --------
    np.ndarray
        2D vector of background intensities [μ_1(t,y), μ_2(t,y)]
        
    Global Dependencies:
    --------------------
    mu : np.ndarray
        Global 2D background intensity vector
    """

    global mu
    # If mu is a function of time and state, implement that here. One should plug in y/FLLN_scaling for the state
    return mu

def intensity(t: float, paths: dict, state_dict: dict, current_state: float, FLLN_scaling: float) -> np.ndarray:
    """
    Calculate the intensity vector for a given time and state:
 
    λ_i(t) = μ_i + Σ_j ∫_0^t r(y(s-), i) * α_ij * exp(-β_ij(t - s)) dN_j(s)
    
    where:
    - μ_i is the background intensity for dimension i
    - r(y(s-), i) is the state-dependent amplification factor at state y(s-) for dimension i (this calls the function 'r', and note that this function takes in the FLLN_scaling parameter)
    - α_ij is the excitation parameter (dimension j exciting dimension i)
    - β_ij is the decay parameter
    - y(s) is the state at time s
 
    Note that under this representation, φ is matrix with such that the excitation contribution along dimension i is the row sum of the i^th row. Similarly, an arrival along dimension i spikes the excitation contribution along the i^th column.
    
    Parameters:
    -----------
    t : float
        Current time
    paths : list
        List of 2 lists, where paths[i] contains arrival times for dimension i
    state_dict : dict
        Dictionary mapping arrival time strings to state values at those times
    current_state : float
        Current state value (y = N_1 - N_2)
    FLLN_scaling : float
        Scaling parameter for FLLN
    
    Returns:
    --------
    np.ndarray
        2D vector of intensities [λ_1(t), λ_2(t)]
        
    Global Dependencies:
    --------------------
    alpha : np.ndarray
        2x2 excitation parameter matrix
    beta : np.ndarray
        2x2 decay parameter matrix
    """
    global alpha, beta
    
    dim = np.shape(alpha)[0]
    background_intensity_vec = background_intensity(t, current_state, FLLN_scaling)
    integrand = np.zeros_like(alpha)
    for i in range(dim):
        for j in range(dim):
            integrand[i,j] = sum([r(state_dict[str(s)],i+1, FLLN_scaling) * alpha[i,j]* np.exp(-beta[i,j]*(t - s)) for s in paths[i]]) 
    out = background_intensity_vec + np.sum(integrand, axis = 1)
    return out


def create_H(alpha: np.ndarray, beta: np.ndarray) -> Callable:
    """
    Create the H function that computes the L¹ norms of the excitation functions φᵢⱼ.
    
    For i,j: φ_ij(t) = α_ij * exp(-β_ij*t)
    
    The L1 norm of φ_ij(·) is computed by:
        H[i,j] = ∫_0^∞ |φ_ij(t)| dt = ∫_0^∞ α_ij * exp(-β_ij*t) dt = α_ij * (1/β_ij)
    
    Parameters:
    -----------
    alpha : np.ndarray
        2x2 array of excitation parameters where α₁₁ = α₂₂ and α₁₂ = α₂₁
    beta : np.ndarray
        2x2 array of decay parameters with same symmetry as alpha
        Must have all elements strictly positive for the L¹ norm to exist
        
    Returns:
    --------
    Callable
        Function that takes state y and returns 2x2 matrix H(y)
    """
    def H_func(y: float) -> np.ndarray:
        # Create the H matrix
        H_mat = np.zeros((2,2))
        for i in range(2):
            H_mat[i,:] = r(y, i+1, 1) * alpha[i,:]/beta[i,:]
        return H_mat
    
    return H_func

# We will need the following function that finds the first occurrence of 1 in an array
def indexOfFirstOne(list_of_bools: List[bool]) -> int:
    """
    Find the index of the first occurrence of 1 in a boolean array.
    
    Parameters:
    -----------
    list_of_bools : List[bool]
        Boolean or binary array to search
        
    Returns:
    --------
    int
        Index of first 1 in array, or -1 if no 1 is found
    """
    idx = np.argmax(list_of_bools)

    return idx if list_of_bools[idx] else -1

def write_simulation_parameters(subfolder_path: str, FLLN_scaling: float, T_final: float, num_paths: int, stability_label: str, timestamp: str) -> str:
    """
    Write simulation parameters to a text file in the specified subfolder.
    
    Parameters:
    -----------
    subfolder_path : str
        Path to the subfolder where the parameter file should be saved
    FLLN_scaling : float
        FLLN scaling parameter n
    T_final : float
        Final simulation time
    num_paths : int
        Number of simulation paths
    stability_label : str
        Stability classification label (e.g., "_very_stable", "_CUSTOM", etc.)
    timestamp : str
        Timestamp string for the simulation run
        
    Returns:
    --------
    str
        Path to the created parameter file
    """
    global alpha, beta, mu, delta, dim, max_arrivals, MAXIMUM_STATE_CUTOFF
    
    param_file = os.path.join(subfolder_path, "simulation_parameters.txt")
    
    with open(param_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("FLLN SIMULATION PARAMETERS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("SIMULATION METADATA\n")
        f.write("-" * 70 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Stability Classification: {stability_label.strip('_') if stability_label else 'N/A'}\n")
        f.write(f"Subfolder: {os.path.basename(subfolder_path)}\n\n")
        
        f.write("FLLN PARAMETERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"FLLN Scaling (n): {FLLN_scaling}\n")
        f.write(f"Final Time (T): {T_final}\n")
        f.write(f"Number of Paths: {num_paths}\n")
        f.write(f"Simulated Time Horizon: [0, {FLLN_scaling * T_final}]\n")
        f.write(f"Rescaled Time Horizon: [0, {T_final}]\n\n")
        
        f.write("HAWKES PROCESS PARAMETERS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Dimension: {dim}\n\n")
        
        f.write(f"Alpha (excitation matrix):\n")
        for row in alpha:
            f.write(f"  {row}\n")
        f.write("\n")
        
        f.write(f"Beta (decay matrix):\n")
        for row in beta:
            f.write(f"  {row}\n")
        f.write("\n")
        
        f.write(f"Mu (background intensity vector):\n")
        f.write(f"  {mu}\n\n")
        
        f.write(f"Delta (state-dependency parameter): {delta}\n\n")
        
        f.write("SIMULATION CONSTRAINTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Maximum Arrivals per Simulation: {max_arrivals}\n")
        f.write(f"Maximum State Cutoff: {MAXIMUM_STATE_CUTOFF}\n\n")
        
        f.write("=" * 70 + "\n")
    
    return param_file


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Simulation Code

def Hawkes_2d_sim(
    T: float,
    FLLN_scaling: float,
    output_file: Optional[str] = None,
    seed: Optional[Union[int, np.random.SeedSequence]] = None
) -> Union[str, Tuple[List[List[float]], List[Tuple[float, int, int]]]]:
    """
    Simulate a 2D state-dependent Hawkes process with exponential kernels.
    
    Implements Ogata's modified thinning algorithm for a 2-dimensional Hawkes process where the state is defined as y = N_1 - N_2 (difference of arrival counts). The state-dependence enters through multiplicative factors r(y/n, i) in the excitation kernels, promoting mean-reverting behavior.
    
    Parameters
    ----------
    T : float
        Final simulation time. Actual simulation runs on [0, FLLN_scaling * T].
    FLLN_scaling : float
        FLLN scaling parameter n. Affects time horizon and state. Process is simulated on [0, FLLN_scaling * T]. State is rescaled by plugging in y/n inside the functions r and background_intensity.
    output_file : str, optional
        Path to save simulation output as pickle file. Recommended for large simulations (large T or FLLN_scaling) or near-unstable parameters to reduce memory usage.
        If None, returns data in memory. Default is None.
    seed : int, np.random.SeedSequence, or None, optional
        Seed for random number generator. If None, uses OS entropy. Default is None.
        
    Returns
    -------
    str or tuple
        If output_file provided:
            Returns output_file path (str). Data saved to disk.
        If output_file is None:
            Returns tuple (paths, full_information) where:
            - paths: list of 2 lists, paths[i] contains arrival times for dimension i
              Format: [[t1, t2, ...], [t1, t2, ...]]
            - full_information: list of tuples (t, d, state) for each arrival
              - t (float): arrival time
              - d (int): dimension index (0 or 1)
              - state (int): state value N_1 - N_2 just before arrival at time t
              Format: [(t, d, state), (t, d, state), ...]
    
    Global Dependencies
    -------------------
    alpha : np.ndarray
        2x2 excitation parameter matrix from SDHawkes_2d_config.
    beta : np.ndarray
        2x2 decay rate matrix from SDHawkes_2d_config.
    mu : np.ndarray
        2D background intensity vector from SDHawkes_2d_config.
    max_arrivals : int
        Maximum number of arrivals before terminating (safety cutoff) from SDHawkes_2d_config.
    
    Notes
    -----
    - State y = N_1 - N_2 evolves as: y → y+1 when dim-0 arrives, y → y-1 when dim-1 arrives.
    - Excitation kernel: phi_ij(t,y) = r(y/n,i) * alpha_ij * exp(-beta_ij * t).
    - Background intensity accessed via background_intensity(t, state, FLLN_scaling) function.
    - State-dependence via r(state/n, i) promotes balancing toward state=0.
    - Uses efficient matrix exponential updates: excitation_matrix *= exp(-beta * dt).
    - Simulation terminates when t >= T or arrivals >= max_arrivals * FLLN_scaling.
    
    See Also
    --------
    r : State-dependent multiplicative factor function.
    background_intensity : Background intensity function.
    FLLN_sim : Run multiple parallel simulations with FLLN scaling.
    """
    global alpha, beta, mu, max_arrivals
    rng = np.random.default_rng(seed)

    dim = 2
    ones_vec = np.ones(dim)

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
    current_intensity_vec = background_intensity(t, current_state, FLLN_scaling) # intensity has no excitation before the first arrival

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
        background = background_intensity(t,current_state, FLLN_scaling)
        current_intensity_vec = excitation_vec + background # total intensity for each dimension

        if (t < T) and (U <= np.sum(current_intensity_vec)):
            TF_array = [U <= sum(current_intensity_vec[:i+1]) for i in range(dim)] # divide current intensity value into bins to figure out which dimension the new arrival belongs to
            d = indexOfFirstOne(TF_array)
            
            # update state while storing all previous states in array
            arrival_type = [1,-1][d] # if d==0, yields 1, if d==1, yields -1
            previous_state = current_state
            current_state = previous_state + arrival_type

            paths[d].append(t)
            state_dict[str(t)] = previous_state  # Store the state at this arrival time
            full_information.append((t, d, previous_state))
            
            num_arrivals_so_far += 1

            ## Update current intensity vec to value just after arrival
            # Calculate new background intensity; uses new state because in the algorithm it plays a role in the next arrival
            background = background_intensity(t, current_state, FLLN_scaling)
            r_vec = np.array([r(previous_state, 1, FLLN_scaling), r(previous_state, 2, FLLN_scaling)]) # need to set r_vec with state just before arrival, to be used in newest arrival's jump contribution
            excitation_matrix[:, d] += alpha[:, d] * r_vec # Add new jump in column d
            excitation_vec = excitation_matrix @ ones_vec # sum rows to get total intensity for each dimensio
            # Calculate total intensity for each dimension
            current_intensity_vec = excitation_vec + background
        
    if output_file:
        with open(output_file, 'wb') as f:
            pickle.dump((paths, full_information), f)
        return output_file
    else:
        return paths, full_information

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Parallel simulation helpers

def data_for_parallel_sims(num_sims, use_disk=False, subfolder_path=None, child_seeds=None):
    """
    Prepare data for parallel simulations.
    
    Parameters:
    -----------
    num_sims : int
        Number of simulations to prepare
    use_disk : bool, optional
        If True, includes (sim_index, subfolder_path) for unique filenames.
        If False, sim_index and subfolder_path are None.
    subfolder_path : str, optional
        Path to subfolder where simulation files should be saved (only used if use_disk=True)
    child_seeds : list of np.random.SeedSequence, optional
        Pre-spawned child seeds, one per simulation. If None, no seeding is applied.

    Returns:
    --------
    list of tuples
        Each element is (sim_index_or_None, subfolder_path_or_None, seed_or_None).
    """
    if child_seeds is None:
        child_seeds = [None] * num_sims
    if use_disk:
        return [(i, subfolder_path, child_seeds[i]) for i in range(num_sims)]
    else:
        return [(None, None, child_seeds[i]) for i in range(num_sims)]

def process_single_sim(func_and_params):
    """
    Helper function to run a single simulation in parallel processing context.
    Since we are using global variables, we ignore the params argument.
    
    Parameters:
    -----------
    func_and_params : tuple
        Tuple of (sim_func, params) where:
        - sim_func is a callable (partially applied sim_wrapper)
        - params is either an int (sim_index for disk mode) or [] (memory mode)
        
    Returns:
    --------
    str or tuple
        If disk mode: returns filename (str)
        If memory mode: returns (paths, full_information) tuple
    """
    sim_func, params = func_and_params
    
    # params is always a 3-tuple: (sim_index_or_None, subfolder_path_or_None, seed_or_None)
    sim_index, subfolder_path, seed = params
    if sim_index is not None:
        return sim_func(sim_index, subfolder_path, seed)
    else:
        return sim_func(seed=seed)

def sim_wrapper(T, FLLN_scaling, sim_index=None, subfolder_path=None, seed=None):
    """
    Wrapper function for Hawkes_2d_sim that can be pickled for multiprocessing.
    
    Parameters:
    -----------
    T : float
        Final simulation time
    FLLN_scaling : float
        Scaling parameter for FLLN    
    sim_index : int, optional
        Index of the simulation. If provided, results are written to disk.
        If None (default), results are stored in memory.
    subfolder_path : str, optional
        Path to subfolder where simulation file should be saved.
        Only used if sim_index is provided.

    Returns:
    --------
    str or tuple
        If sim_index provided: returns output filename (str)
        If sim_index is None: returns (paths, full_information) tuple as returned by Hawkes_2d_sim:
            - paths: List of 2 lists containing arrival times
            - full_information: List of tuples (t, d, state)
    """
    if sim_index is not None:
        if subfolder_path is not None:
            output_file = f"{subfolder_path}/sim_{sim_index:04d}.pkl"
        else:
            output_file = f"{output_dir}/sim_{sim_index:04d}.pkl"
        return Hawkes_2d_sim(T, FLLN_scaling, output_file=output_file, seed=seed)
    else:
        output_file = None
        return Hawkes_2d_sim(T, FLLN_scaling, output_file, seed=seed)

def run_parallel_sims(data, process_sim_fun, num_workers):
    """
    Run multiple Hawkes simulations in parallel using multiprocessing.
    
    Parameters:
    -----------
    data : list
        List of parameter placeholders (typically empty lists) with length 
        equal to number of simulations
    process_sim_fun : callable
        Partially applied simulation function (from functools.partial)
    num_workers : int
        Number of parallel worker processes to use
        
    Returns:
    --------
    list
        List of simulation results with length equal to len(data).
        If using disk mode: list of filenames (str)
        If using memory mode: list of tuples (paths, full_information) with:
            - paths: [[arrival_times_dim1], [arrival_times_dim2]]
            - full_information: [(t, d, state), ...]
            Length equals len(data)
    """
    total_sims = len(data)
    print(f"\nRunning {total_sims} parallel simulations with {num_workers} workers...")
    
    # Prepare data with the simulation function
    func_and_params = [(process_sim_fun, params) for params in data]
    
    with Pool(processes=num_workers) as pool:
        result_list = list(tqdm(
            pool.imap_unordered(process_single_sim, func_and_params),
            total=total_sims,
            desc="Simulations progress",
            dynamic_ncols=True,
            mininterval=0.1
        ))
    return result_list

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## LLN Scaling

def FLLN_sim(num_paths: int, T_final: float, FLLN_scaling: float, use_disk=False, base_seed=None):
    """
    Simulate the FLLN (Functional Law of Large Numbers) scaled Hawkes process.
    
    Runs simulations on [0, FLLN_scaling*T_final] with state-dependency r(y/n, i),
    then rescales to [0, T_final] with counts and states divided by FLLN_scaling.
 
    Parameters:
    -----------
    num_paths : int
        Number of simulation paths to generate
    T_final : float
        Final time for the scaled simulation (output on [0, T_final])
    FLLN_scaling : float
        Scaling parameter n for FLLN: simulates (1/n)*N(n*t) for t in [0, T_final]
    use_disk : bool, optional
        If True, write simulation data to disk to save memory (recommended for 
        large FLLN_scaling or many paths). If False, store in memory (default).
        If True, creates a subfolder within the output directory within which
        the simulation files are saved.

    Returns:
    --------
    list or tuple
        If use_disk=False:
            List of scaled simulation paths with format [[(t,d,y/n), ...], ...]
        If use_disk=True:
            Tuple of (scaled_paths_list, subfolder_path) where subfolder_path is the
            directory containing the simulation files
        
        Scaled paths format:
        - Outer list has length num_paths
        - Each inner list represents one path as a list of tuples:
          * t (float): rescaled arrival time in [0, T_final]
          * d (int): dimension index (0 or 1)
          * y/n (float): rescaled state value (N_1 - N_2)/n at time t
          
    Global Dependencies:
    --------------------
    dim : int
        Dimension of the Hawkes process (should be 2)
    NUM_WORKERS : int
        Number of parallel workers for simulations
    if use_disk=True: 
        output_dir : str
            Directory for output files
    """

    global dim, alpha

    # Create timestamped subfolder if using disk
    subfolder_path = None
    if use_disk:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine stability label based on alpha
        stability_label = ""
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
        
        subfolder_name = f"FLLN_n{FLLN_scaling}_T{T_final}_paths{num_paths}{stability_label}_{timestamp}"
        subfolder_path = os.path.join(output_dir, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        print(f"Created simulation folder: {subfolder_path}")
        
        # Write simulation parameters to text file
        param_file = write_simulation_parameters(subfolder_path, FLLN_scaling, T_final, num_paths, stability_label, timestamp)
        print(f"Saved simulation parameters to: {param_file}")
 
    # Spawn independent child seeds from SeedSequence for each simulation path
    ss = np.random.SeedSequence(base_seed)
    child_seeds = ss.spawn(num_paths)
    
    # implement num_paths many simulations until time n*T_final. Need each path to have excitation \phi(t,y/n).
    data = data_for_parallel_sims(num_paths, use_disk=use_disk, subfolder_path=subfolder_path, child_seeds=child_seeds)
    sim_func = partial(sim_wrapper, FLLN_scaling*T_final, FLLN_scaling)
    result_list = run_parallel_sims(data, sim_func, NUM_WORKERS)

    
    # compute (1/n)*N(nt) for each t \in [0,T_final]. This can probably be done in parallel over the simulations
    scaled_paths_list = []
    # Process results: load from disk if needed, or use directly from memory
    for i, result in enumerate(result_list):
        if use_disk:
            # Result is a filename (string) - load from disk
            with open(result, 'rb') as f:
                _, full_information = pickle.load(f)
        else:
            # Result is a tuple (paths, full_information) - extract full_information
            _, full_information = result
        
        # Scale the data
        scaled_full_info = [(t/FLLN_scaling, d, current_state/FLLN_scaling) 
                            for t, d, current_state in full_information]
        scaled_paths_list.append(scaled_full_info)
    
    # Return the scaled paths with time inputs [0,T_final]
    # If using disk, also return the subfolder path for saving plots
    if use_disk:
        return scaled_paths_list, subfolder_path
    else:
        return scaled_paths_list


def compute_FLLN_ODE_difference(flln_paths_list, ode_solution, T_final, num_points=1000):
    """
    Compute maximum difference between FLLN sample paths and ODE solution. Note that, while FLLN sample paths are graphed with piecewise interpolation, the actual FLLN sample paths are piecewise constant, and the maximum difference is computed as such.
    
    Parameters:
    -----------
    flln_paths_list : list
        List of FLLN paths, each path structured as [(t, d, state), ...], i.e., structured like the output of "full_info" in from Hawkes_2d_sim()
    ode_solution : dict
        ODE solution dict with keys 't' and 'P' (state values)
    T_final : float
        Final time for comparison
    num_points : int
        Number of time points for comparison grid
        
    Returns:
    --------
    dict with:
        - 'max_difference_per_path': np.ndarray of max |path - ODE| for each path
        - 'overall_max_difference': float, max over all paths and times
        - 'time_grid': time points used
        - 'path_values': evaluated path values on grid
        - 'ode_values': evaluated ODE values on grid
    """
    # Create uniform time grid
    time_grid = np.linspace(0, T_final, num_points)
    
    # Interpolate ODE solution onto uniform grid
    ode_values = np.interp(time_grid, ode_solution['t'], ode_solution['P'])
    
    # Evaluate FLLN paths on the grid
    num_paths = len(flln_paths_list)
    path_values = np.zeros((num_paths, num_points))
    
    for path_idx, full_info in enumerate(tqdm(flln_paths_list, desc='Computing FLLN ODE difference', unit='paths')):
        if len(full_info) == 0:
            # No arrivals, state is 0 everywhere
            continue
        
        # Extract arrival times and states
        arrival_times = np.array([t for t, d, state in full_info])
        arrival_states = np.array([state for t, d, state in full_info])
        
        # Use searchsorted to find indices efficiently
        indices = np.searchsorted(arrival_times, time_grid, side='right') - 1
        
        # Handle times before first arrival
        path_values[path_idx, :] = np.where(
            indices >= 0,
            arrival_states[indices],
            0  # State is 0 before first arrival
        )

    # Compute differences
    differences = np.abs(path_values - ode_values[np.newaxis, :])  # Broadcasting
    
    # Compute max difference per path
    max_difference_per_path = np.max(differences, axis=1)
    
    # Compute overall max difference
    overall_max_difference = np.max(differences)
    
    return {
        'max_difference_per_path': max_difference_per_path,
        'overall_max_difference': overall_max_difference,
        'time_grid': time_grid,
        'path_values': path_values,
        'ode_values': ode_values,
        'differences': differences
    }

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### ODE Solving

def system_of_odes(t: float, Lambda: np.ndarray, H_func: Callable, background_intensity_func: Callable) -> np.ndarray:
    """
    Define the system of ODEs: dΛ/dt = (I - H_func(P(t)))^(-1) * μ(t, P(t))
    where P(t) = Λ_1 - Λ_2
    
    Parameters:
    -----------
    t : float
        Time variable
    Lambda : np.ndarray
        Current state vector [Λ_1, Λ_2]
    H_func : Callable
        Function H_func: R → R that takes P(t) as input
    background_intensity_func : Callable
        Function background_intensity: R^2 → R^2 that takes (t, P(t)) as input

    Returns:
    --------
    np.ndarray
        The derivative dΛ/dt
    """
    # Calculate P(t) = Λ₁ - Λ₂
    P = Lambda[0] - Lambda[1]
    
    # Calculate H_func(P(t))
    H_P = H_func(P)
    
    # ## Create identity matrix
    # I = np.eye(2)
    # Calculate (I - H_func(P(t)))^(-1)
    # inv_term = np.linalg.inv(I - H_P)

    ## Calculate (I - H_func(P(t)))^(-1) directly
    # For a 2x2 matrix of form I - H, the inverse is:
    # 1/det * [1-H22  H12]
    #         [H21    1-H11]
    # where det = (1-H11)(1-H22) - H12*H21
    det = (1 - H_P[0,0]) * (1 - H_P[1,1]) - H_P[0,1] * H_P[1,0]
    inv_term = (1/det) * np.array([[1 - H_P[1,1], H_P[0,1]],
                                  [H_P[1,0], 1 - H_P[0,0]]])
    
    # Calculate background intensity
    background = background_intensity_func(t, P, FLLN_scaling=1) # FLLN scaling is set to 1 for the purposes of solving ODEs
    
    # Calculate final result
    dLambda_dt = inv_term @ background
    
    return dLambda_dt
    

def solve_ode(t_span: tuple, Lambda_0: np.ndarray, H_func: Callable, background_intensity_func: Callable,
              t_eval: np.ndarray = None, method: str = 'DOP853', rtol: float = 1e-8, 
              atol: float = 1e-8, max_step: float = 0.01, **kwargs) -> dict:
    """
    Solve the system of ODEs using scipy.integrate.solve_ivp
    
    Parameters:
    -----------
    t_span : tuple
        Interval of integration (t0, tf)
    Lambda_0 : np.ndarray
        Initial conditions [Λ_1(0), Λ_2(0)]
    H_func : Callable
        Function H_func: R → R that takes P(t) as input
    background_intensity_func : Callable
        Function background_intensity: R^2 → R^2 that takes (t, P(t)) as input
    t_eval : np.ndarray, optional
        Times at which to store the computed solution
    method : str, optional
        Integration method to use (default: 'DOP853')
    **kwargs
        Additional arguments passed to solve_ivp
        
    Returns:
    --------
    dict
        Dictionary containing solution information including:
        - t: Time points
        - Lambda: Solution array where Lambda[i] = [Λ_1(t[i]), Λ_2(t[i])]
        - P: Array of P(t) = Λ_1(t) - Λ_2(t) values
        - success: Whether the integration was successful
    """
    # Define the ODE function with fixed H_func and mu
    def ode_fn(t, y):
        return system_of_odes(t, y, H_func, background_intensity_func)
    
    # If t_eval is not provided, create a dense time grid
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Use 1000 points for smooth curves
    
    # Solve the ODE system with tighter tolerances and fixed step size
    sol = solve_ivp(ode_fn, t_span, Lambda_0, t_eval=t_eval, method=method,
                    rtol=rtol, atol=atol, max_step=max_step, **kwargs)
    
    # Extract results
    result = {
        't': sol.t,
        'Lambda': sol.y.T,  # Transpose to get shape (n_times, 2)
        'P': sol.y[0] - sol.y[1],  # Calculate P(t) for all t
        'success': sol.success,
        'message': sol.message
    }
    
    return result

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Plotting

def plot_component_paths(paths_list, output_dir, time_grid_size=1000):
    """
    Plot the paths for each component of the 2D Hawkes process and their averages.
    
    Parameters:
    -----------
    paths_list : list
        List of tuples (paths, full_info) from simulations where:
        - paths: [[arrival_times_dim1], [arrival_times_dim2]]
        - full_info: [(t, d, state), ...]
    output_dir : str
        Directory path to save plots
    time_grid_size : int, optional
        Number of points for time interpolation (default: 1000)
        
    Returns:
    --------
    tuple
        (time_grid, values_dim1, values_dim2, differences) where:
        - time_grid: np.ndarray of shape (time_grid_size,) with time points
        - values_dim1: np.ndarray of shape (num_paths, time_grid_size) with N₁(t) values
        - values_dim2: np.ndarray of shape (num_paths, time_grid_size) with N₂(t) values
        - differences: np.ndarray of shape (num_paths, time_grid_size) with (N₁-N₂)(t) values
        
    Side Effects:
    -------------
    Saves a PNG file to output_dir with filename 'hawkes_2d_paths_YYYYMMDD_HHMMSS.png'
    """

    # Get maximum time across all paths
    max_time = 0
    for paths, _ in paths_list:
        max_time = max(max_time, max(max(paths[0]), max(paths[1])))
    
    # Create time grid
    time_grid = np.linspace(0, max_time, time_grid_size)
    
    # Initialize arrays for values
    values_dim1 = np.zeros((len(paths_list), time_grid_size))
    values_dim2 = np.zeros((len(paths_list), time_grid_size))
    differences = np.zeros((len(paths_list), time_grid_size))
    
    # Calculate values for each path
    for i, (paths, _) in enumerate(paths_list):
        # Calculate values for dimension 1
        for j, t in enumerate(time_grid):
            values_dim1[i,j] = sum(1 for x in paths[0] if x <= t)
            values_dim2[i,j] = sum(1 for x in paths[1] if x <= t)
            differences[i,j] = values_dim1[i,j] - values_dim2[i,j]
    
    # Calculate means
    mean_dim1 = np.mean(values_dim1, axis=0)
    mean_dim2 = np.mean(values_dim2, axis=0)
    mean_diff = np.mean(differences, axis=0)
    
    # Create plots
    plt.figure(figsize=(12, 15))
    
    # Plot dimension 1
    plt.subplot(311)
    plt.title('Component 1 (N₁)')
    for i in range(len(paths_list)):
        plt.plot(time_grid, values_dim1[i], 'b-', alpha=0.1)
    plt.plot(time_grid, mean_dim1, 'r-', linewidth=2, label='Empirical Mean')
    plt.xlabel('Time')
    plt.ylabel('N₁(t)')
    plt.legend()
    plt.grid(True)
    
    # Plot dimension 2
    plt.subplot(312)
    plt.title('Component 2 (N₂)')
    for i in range(len(paths_list)):
        plt.plot(time_grid, values_dim2[i], 'g-', alpha=0.1)
    plt.plot(time_grid, mean_dim2, 'r-', linewidth=2, label='Empirical Mean')
    plt.xlabel('Time')
    plt.ylabel('N₂(t)')
    plt.legend()
    plt.grid(True)
    
    # Plot differences
    plt.subplot(313)
    plt.title('State (N₁ - N₂)')
    for i in range(len(paths_list)):
        plt.plot(time_grid, differences[i], 'purple', alpha=0.1)
    plt.plot(time_grid, mean_diff, 'r-', linewidth=2, label='Empirical Mean')
    plt.xlabel('Time')
    plt.ylabel('N₁(t) - N₂(t)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_dir, f'hawkes_2d_paths_{timestamp}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    
    return time_grid, values_dim1, values_dim2, differences

def FLLN_plot(paths_list, ode_solution, diff_results, output_dir, FLLN_scaling, T_final, time_grid_size=1000, ylim=None, stability_label=""):
    """
    Plot FLLN scaled paths compared with ODE solution and display difference statistics.
    
    Parameters:
    -----------
    paths_list : list
        List of scaled simulation paths as outputted by FLLN_sim.
        Format: [[(t, d, state), ...], ...] where each inner list is a single 
        path with time inputs on [0, T_final] and state = (N_1 - N_2)/n.
    ode_solution : dict
        ODE solution dictionary with keys 't' and 'P' (state P = Lambda_1 - Lambda_2)
    diff_results : dict
        Results from compute_FLLN_ODE_difference containing difference statistics
    output_dir : str
        Directory path where the plot will be saved
    FLLN_scaling : float
        The scaling parameter n used for FLLN
    T_final : float
        Final time of simulation
    time_grid_size : int, optional
        Number of points in time grid for plotting (default: 1000)
    ylim : tuple or None, optional
        Y-axis limits as (ymin, ymax). If None (default), limits are auto-selected.
    stability_label : str, optional
        Stability classification label (e.g., "_very_stable", "_CUSTOM", etc.) to include
        in the filename. Default is empty string.
        
    Returns:
    --------
    str
        Path to the saved plot file
    """
    
    
    num_paths = len(paths_list)
    
    # Create time grid for plotting
    time_grid = np.linspace(0, T_final, time_grid_size)
    
    # Compute state values (N_1 - N_2) for each FLLN path on the time grid
    flln_state_values = np.zeros((num_paths, len(time_grid)))
    
    for path_idx, full_info in enumerate(tqdm(paths_list, desc='Plotting Paths', unit='paths')):
        if len(full_info) == 0:
            # No arrivals, state is 0 everywhere
            continue
        
        # Extract arrival times and states
        arrival_times = np.array([t for t, d, state in full_info])
        arrival_states = np.array([state for t, d, state in full_info])
        
        # Use searchsorted to find indices efficiently
        indices = np.searchsorted(arrival_times, time_grid, side='right') - 1
        
        # Handle times before first arrival
        flln_state_values[path_idx, :] = np.where(
            indices >= 0,
            arrival_states[indices],
            0  # State is 0 before first arrival
        )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot FLLN sample paths
    for i in range(num_paths):
        ax.plot(time_grid, flln_state_values[i], 'b-', alpha=0.3, linewidth=1)
    
    # Plot ODE solution for state P(t) = Lambda_1(t) - Lambda_2(t)
    ax.plot(ode_solution['t'], ode_solution['P'], 'r-', linewidth=2.5, label='ODE Solution P(t)')
    
    # Add one dummy line for FLLN paths legend
    ax.plot([], [], 'b-', alpha=0.3, linewidth=1, label=f'FLLN Paths (n={FLLN_scaling})')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('State (N₁ - N₂)', fontsize=12)
    ax.set_title(f'FLLN Scaled Paths vs ODE Solution (n={FLLN_scaling}, {num_paths} paths)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits if specified
    if ylim is not None:
        ax.set_ylim(ylim)

    # Add difference statistics text box
    textstr = '\n'.join([
        f'Max Difference: {diff_results["overall_max_difference"]:.4f}',
        f'Mean Max Diff: {np.mean(diff_results["max_difference_per_path"]):.4f}'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f'FLLN_vs_ODE_n{FLLN_scaling}{stability_label}_{timestamp}.png')
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    return filename

def plot_ODE_solution(t_span: tuple, Lambda_0: np.ndarray, H_func: Callable, background_intensity_func: Callable,
                     params: dict = None, method: str = 'RK45', n_points: int = n_system, **kwargs) -> dict:
    """
    Solve and plot the ODE system on a fine mesh suitable for plotting.
    
    Parameters:
    -----------
    t_span : tuple
        Interval of integration (t0, tf)
    Lambda_0 : np.ndarray
        Initial conditions [Λ_1(0), Λ_2(0)]
    H_func : Callable
        Function H_func: R → R^2 × R^2 that takes P(t) as input
    background_intensity_func : Callable
        Function that takes t and P(t) as input and returns the background intensity
    params : dict, optional
        Dictionary containing system parameters to display on the plot.
        Keys are parameter names (strings), values are parameter values.
        If None, no parameters will be displayed.
    method : str, optional
        Integration method to use (default: 'RK45')
    n_points : int, optional
        Number of points to use for plotting (default: n_system)
    **kwargs
        Additional arguments passed to solve_ivp
        
    Returns:
    --------
    dict
        Dictionary containing:
        - figure: matplotlib figure object
        - solution: ODE solution dictionary

    Requires (global/imported)
    --------
        - r: function that takes two arguments (x, y) and returns the probability of interaction
        - n_system: number of grid points for the ODE solution
    
    """
    # Create time points for evaluation
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    
    # Solve the ODE system
    sol = solve_ode(t_span, Lambda_0, H_func, background_intensity_func, t_eval=t_eval, method=method, **kwargs)
    
    if not sol['success']:
        print(f"Warning: ODE solution failed with message: {sol['message']}")
    
    # Create figure with GridSpec for flexible subplot layout
    fig = plt.figure(figsize=(12, 12))
    gs = plt.GridSpec(4, 2, height_ratios=[3, 3, 2, 2])
    
    # Create main subplot for Lambda components
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax_ode = fig.add_subplot(gs[2, :])
    ax_params = fig.add_subplot(gs[3, :])
    
    # Plot Lambda components
    ax1.plot(sol['t'], sol['Lambda'][:,0], label='Λ₁(t)')
    ax1.plot(sol['t'], sol['Lambda'][:,1], label='Λ₂(t)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Λ(t)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Components of Λ(t)')
    
    # Plot P(t)
    ax2.plot(sol['t'], sol['P'], label='P(t) = Λ₁(t) - Λ₂(t)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('P(t)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Difference Process P(t)')
    
    # Add system of ODEs
    ax_ode.axis('off')
    ode_text = r"System of ODEs:"
    ax_ode.text(0.5, 0.8, ode_text, fontsize=10, ha='center')
    
    equation = r"$\frac{d}{dt} \bar{\Lambda} = (I - H(\bar{P}(t)))^{-1} \bar{\mu}(t, \bar{P}(t))$"
    ax_ode.text(0.5, 0.6, equation, fontsize=12, ha='center')
    
    condition = r"where $\bar{P}(t) = \bar{\Lambda}_1 - \bar{\Lambda}_2$"
    ax_ode.text(0.5, 0.45, condition, fontsize=10, ha='center')
    
    h_def = r"The $(i,j)$ component of $H(y)$ is given by $r(y,i+1) \alpha_{ij} / \beta_{ij}$ where:"
    ax_ode.text(0.5, 0.3, h_def, fontsize=10, ha='center')
    
    # check if we are using the state-dependent version of r or the constant (always returning 1) version of r, and then format the output strings accordingly. This way, the plots come equipped with the information of which definition of r was used in the ODE solution.
    if r(np.e,2, FLLN_scaling=1) > 1:
        r_def1 = r"$r_i(y) = (1+\delta_1)^y$"
        r_def2 = r"$r_i(y) = (1-\delta_2)^y$"
    else:
        r_def1 = r"$r_i(y) = 1$"
        r_def2 = ""
    ax_ode.text(0.5, 0.15, r_def1, fontsize=10, ha='center')
    if r_def2:  # Only show second line if there's content
        ax_ode.text(0.5, 0.05, r_def2, fontsize=10, ha='center')
    
    # Add parameters and equations
    ax_params.axis('off')
    if params is not None:
        param_text = r"System Parameters:" + "\n"
        for param_name, param_value in params.items():
            # Format the value appropriately
            if isinstance(param_value, np.ndarray):
                value_str = str(param_value.tolist())
            else:
                value_str = str(param_value)
            param_text += r"$" + param_name + r" = " + value_str + r"$" + "\n"
        ax_params.text(0.5, 0.5, param_text, fontsize=10, ha='center', va='center')
    else:
        ax_params.text(0.5, 0.5, "No parameters provided", fontsize=10, ha='center', va='center')
    
    plt.tight_layout()
    
    return {
        'figure': fig,
        'solution': sol
    }



def plot_FLLN_from_disk(folder_path, plot_name, ode_solution=None, FLLN_scaling=None, T_final=1.0, time_grid_size=1000, ylim=None):
    """
    Load simulation data from disk files and create FLLN plot, optionally with ODE comparison.
    
    This function processes pickle files ONE AT A TIME to avoid memory issues with large files.
    It loads each file, processes it, and discards it before loading the next one.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing .pkl files with simulation data.
        Each .pkl file should contain tuples of (t, d, state) for each arrival.
    plot_name : str
        Name/title for the plot (used in plot title and filename)
    ode_solution : dict or None, optional
        ODE solution dictionary with keys 't' and 'P' (state values).
        If None, only simulation paths are plotted without ODE comparison.
        Default: None
    FLLN_scaling : float or None, optional
        FLLN scaling parameter n. If provided, data is rescaled:
        - Times: t → t/n (maps [0, n*T_final] to [0, T_final])
        - States: y → y/n
        If None, data is plotted as-is without rescaling.
        Default: None
    T_final : float, optional
        Final time for the plot x-axis (default: 1.0)
    time_grid_size : int, optional
        Number of points in time grid for plotting (default: 1000)
    ylim : tuple or None, optional
        Y-axis limits as (ymin, ymax). If None, limits are auto-selected.
        Default: None
        
    Returns:
    --------
    str
        Path to the saved plot file
        
    Notes:
    ------
    **Memory Efficiency**: This function processes files ONE AT A TIME to handle large
    simulation files (e.g., 800MB each). Each file is loaded, processed into the plot
    data structure, then discarded before loading the next file.
    
    **Requirements for FLLN plotting from existing disk files:**
    
    1. **File Format**: Each .pkl file must contain a single pickled tuple
       (paths, full_information), where full_information is a list of
       (t, d, state) tuples representing:
       - t (float): arrival time
       - d (int): dimension index (0 or 1)
       - state (float): state value at arrival
    
    2. **Time Horizon**: If FLLN_scaling is provided, the simulation files must contain
       data on the extended time horizon [0, n*T_final] where n = FLLN_scaling.
       The function will rescale to [0, T_final].
    
    3. **ODE Solution**: If comparing with ODE, the ode_solution dict must have:
       - 't': array of time points
       - 'P': array of state values P(t) = Lambda_1(t) - Lambda_2(t)
       The ODE should already be solved on [0, T_final] (not the extended horizon).
       NOTE: the parameters for the ODE solution need to be manually ensured to be the same as those for the .pkl files that are being loaded
    
    4. **File Naming**: Files should have .pkl extension. All .pkl files in folder_path
       will be loaded and plotted.
    
    5. **Difference Computation**: If ode_solution is provided, differences between
       FLLN paths and ODE are computed and displayed on the plot.
    
    Example:
    --------
    >>> # Plot FLLN simulations with rescaling and ODE comparison
    >>> plot_FLLN_from_disk(
    ...     folder_path='/path/to/simulations',
    ...     plot_name='FLLN_n100_comparison',
    ...     ode_solution=sol,
    ...     FLLN_scaling=100,
    ...     T_final=1.0,
    ...     ylim=(-10, 10)
    ... )
    
    >>> # Plot raw simulations without rescaling
    >>> plot_FLLN_from_disk(
    ...     folder_path='/path/to/simulations',
    ...     plot_name='raw_simulations',
    ...     FLLN_scaling=None,
    ...     T_final=1.0
    ... )
    """
    
    # Get all .pkl files in the folder
    pkl_files = sorted(glob.glob(os.path.join(folder_path, '*.pkl')))
    
    if len(pkl_files) == 0:
        raise ValueError(f"No .pkl files found in {folder_path}")
    
    num_paths = len(pkl_files)
    print(f"Processing {num_paths} simulation files from {folder_path}...")
    print("Note: Files are processed ONE AT A TIME to conserve memory.")
    
    # Create time grid
    time_grid = np.linspace(0, T_final, time_grid_size)
    
    # Pre-allocate arrays for path state values and ODE differences
    path_state_values = np.zeros((num_paths, time_grid_size))
    if ode_solution is not None:
        ode_values = np.interp(time_grid, ode_solution['t'], ode_solution['P'])
        max_differences = np.zeros(num_paths)
    
    # Process each file ONE AT A TIME to avoid overloading memory
    for path_idx, pkl_file in enumerate(tqdm(pkl_files, desc='Processing files', unit='files')):
        # Load simulation data from this file
        with open(pkl_file, 'rb') as f:
            _, full_information = pickle.load(f)
        
        if len(full_information) == 0:
            # Empty file, skip
            continue
        
        # Convert to numpy arrays efficiently using structured array unpacking
        full_info_array = np.array(full_information, dtype=[('t', float), ('d', int), ('state', float)])
        arrival_times = full_info_array['t']
        arrival_states = full_info_array['state']
        
        # Apply FLLN rescaling if specified
        if FLLN_scaling is not None:
            arrival_times = arrival_times / FLLN_scaling
            arrival_states = arrival_states / FLLN_scaling
        
        # Use searchsorted to find state values on time grid (vectorized)
        indices = np.searchsorted(arrival_times, time_grid, side='right') - 1
        
        # Compute state values on grid
        path_state_values[path_idx, :] = np.where(
            indices >= 0,
            arrival_states[indices],
            0  # State is 0 before first arrival
        )
        
        # Compute difference from ODE if provided
        if ode_solution is not None:
            differences = np.abs(path_state_values[path_idx, :] - ode_values)
            max_differences[path_idx] = np.max(differences)
        
        # Clear the loaded data to free memory before next iteration
        del full_information, full_info_array, arrival_times, arrival_states, indices
    
    # Compute overall difference statistics if ODE provided
    if ode_solution is not None:
        overall_max_difference = np.max(max_differences)
        mean_max_difference = np.mean(max_differences)
        print(f"\nDifference Statistics:")
        print(f"  Overall max difference: {overall_max_difference:.6f}")
        print(f"  Mean of max differences: {mean_max_difference:.6f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot all paths
    for i in range(num_paths):
        ax.plot(time_grid, path_state_values[i], 'b-', alpha=0.3, linewidth=1)
    
    # Plot ODE solution if provided
    if ode_solution is not None:
        ax.plot(ode_solution['t'], ode_solution['P'], 'r-', linewidth=2.5, label='ODE Solution P(t)')
        legend_label = f'Simulation Paths (n={FLLN_scaling})' if FLLN_scaling else 'Simulation Paths'
        ax.plot([], [], 'b-', alpha=0.3, linewidth=1, label=legend_label)
    else:
        ax.plot([], [], 'b-', alpha=0.3, linewidth=1, label=f'{num_paths} Simulation Paths')
    
    # Labels and formatting
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('State (N₁ - N₂)', fontsize=12)
    
    if FLLN_scaling is not None and ode_solution is not None:
        title = f'{plot_name}: FLLN Paths vs ODE (n={FLLN_scaling}, {num_paths} paths)'
    elif FLLN_scaling is not None:
        title = f'{plot_name}: FLLN Scaled Paths (n={FLLN_scaling}, {num_paths} paths)'
    else:
        title = f'{plot_name}: Simulation Paths ({num_paths} paths)'
    
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits if specified
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add difference statistics text box if ODE provided
    if ode_solution is not None:
        textstr = '\n'.join([
            f'Max Difference: {overall_max_difference:.4f}',
            f'Mean Max Diff: {mean_max_difference:.4f}'
        ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f'{plot_name}_{timestamp}.png')
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Plot saved to {filename}")
    
    return filename



