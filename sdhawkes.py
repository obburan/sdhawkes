import numpy as np
from typing import Callable, Optional, Union, Tuple, List
from multiprocessing import Pool
from tqdm import tqdm
import datetime
import os
from functools import partial
import pickle
from scipy.integrate import solve_ivp


"""
Implementation of general State-Dependent Hawkes process simulation and related functions. Allow for user to pass in their own background intensity function and excitation kernel function. 

NOTE: Current implementation assumes non-temporally-increasing excitation terms (this is a STRICT assumption). 

NOTE: FLLN scaling is handled internally - user-provided background_intensity_func and excitation_kernel_func should NOT account for FLLN scaling. The simulation automatically passes rescaled states (state/n) to user functions.
"""


def sim_SDHawkes_once_general(
    dim: int,
    state_dim: int,
    state_matrix: np.ndarray,
    background_intensity_func: Callable[[float, np.ndarray], np.ndarray],
    background_intensity_max: float,
    excitation_kernel_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    max_arrivals: int,
    use_disk: bool,
    T: float,
    FLLN_scaling: float = 1,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    seed: Optional[Union[int, np.random.SeedSequence]] = None
) -> Union[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Simulate one sample path of a general state-dependent Hawkes process using Ogata's modified thinning algorithm.
    
    This is a module-level function designed to be picklable for multiprocessing. It implements a general state-dependent Hawkes process where both background intensity and excitation kernels can depend on the process state.

    NOTE: it is expected that the excitation kernel function accepts arrays of all past arrival information and returns the current value, as specified in the init documentation.
    NOTE: It is assumed that the excitation kernel is non-increasing in time.
    
    Parameters
    ----------
    dim : int
        Number of dimensions (types) in the Hawkes process.
    state_dim : int
        Dimensionality of the state space. Use 1 for scalar state.
    state_matrix : np.ndarray
        State transition matrix. Shape (state_dim, dim) or (1, dim) for scalar state.
        Entry (i, j) specifies how state component i changes when dimension j has an arrival.
    background_intensity_func : Callable[[float, np.ndarray], np.ndarray]
        Function with signature (t, state) -> intensity_vector.
        Returns background intensity vector of shape (dim,) at time t given state.
        State is automatically rescaled by 1/FLLN_scaling before being passed to this function.
    background_intensity_max : float
        Maximum possible change in background intensity per dimension over any time interval.
        Used as upper bound in thinning algorithm. Set to 0 if background is constant.
    excitation_kernel_func : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        Function with signature (time_diffs, past_dims, past_states) -> excitation_matrix.
        - time_diffs: array of shape (n_past,) with times since each past arrival
        - past_dims: array of shape (n_past,) with dimension indices of past arrivals
        - past_states: array of shape (n_past,) or (n_past, state_dim) with states at past arrivals
        Returns excitation matrix of shape (dim, dim) where entry (i,j) is total excitation to dimension i from all past arrivals in dimension j. States are automatically rescaled by 1/FLLN_scaling before being passed to this function.
    max_arrivals : int
        Maximum number of arrivals before terminating simulation (safety cutoff).
    use_disk : bool
        If True, write arrivals to disk at the end of the simulation. If False, keep all arrivals in memory.
    T : float
        Final simulation time. Actual simulation runs on [0, FLLN_scaling * T].
    FLLN_scaling : float, optional
        FLLN scaling parameter n. Extends time horizon to [0, n*T] and automatically rescales states by 1/n when passing to user functions. Default is 1.
    output_dir : str, optional
        Directory to save simulation output. Required if use_disk=True. Default is None.
    output_name : str, optional
        File name for simulation output (appended to output_dir). Required if use_disk=True. Default is None.
    seed : int, np.random.SeedSequence, or None, optional
        Seed for random number generator. If None, uses OS entropy. Default is None.
    
    Returns
    -------
    str or tuple
        If use_disk=True: returns full output file path (str) to saved data.
        If use_disk=False: returns tuple (arrival_times_array, arrival_dims_array, arrival_states_array) where:
            - arrival_times_array: np.ndarray of shape (n_arrivals,) with arrival times
            - arrival_dims_array: np.ndarray of shape (n_arrivals,) with dimension indices
            - arrival_states_array: np.ndarray of shape (n_arrivals,) or (n_arrivals, state_dim) with states
    
    Notes
    -----
    - IMPORTANT: Excitation kernels must be non-increasing in time.
    - IMPORTANT: For parallel execution, all user-provided callables must be defined at module level (within their script; i.e., not within this module, but within the external module that feeds callables into this function or the SDHawkes class).
    - Memory usage: O(max_arrivals * FLLN_scaling) regardless of use_disk setting during simulation. Disk mode only reduces total concurrent memory usage across multiple simulations when running multiple paths.
    
    Examples
    --------
    >>> # Constant background, exponential kernel
    >>> def background(t, state):
    ...     return np.array([100.0, 80.0])
    >>> def kernel(time_diffs, past_dims, past_states):
    ...     # Exponential decay kernel
    ...     alpha = np.array([[0.5, 0.1], [0.1, 0.5]])
    ...     beta = np.array([[1.0, 1.0], [1.0, 1.0]])
    ...     exc = np.zeros((2, 2))
    ...     for k, (dt, j) in enumerate(zip(time_diffs, past_dims)):
    ...         exc[:, j] += alpha[:, j] * np.exp(-beta[:, j] * dt)
    ...     return exc
    >>> state_matrix = np.zeros((1, 2))  # State-agnostic
    >>> times, dims, states = sim_SDHawkes_once_general(
    ...     dim=2, state_dim=1, state_matrix=state_matrix,
    ...     background_intensity_func=background, background_intensity_max=0.0,
    ...     excitation_kernel_func=kernel, max_arrivals=100000,
    ...     use_disk=False, T=10.0, seed=2026
    ... )
    """
    ## Initialization
    rng = np.random.default_rng(seed)
    max_size = int(max_arrivals * FLLN_scaling)
    arrival_times_array = np.zeros(max_size)
    arrival_dims_array = np.zeros(max_size, dtype=int)
    if state_dim == 1:
        arrival_states_array = np.zeros(max_size)
    else:
        arrival_states_array = np.zeros((max_size, state_dim))
    # Iteratively updated variables
    t = 0
    current_state = np.zeros(state_dim)
    num_arrivals_so_far = 0 # We cut the simulation off after enough arrivals
    # Self-excitation terms
    excitation_matrix = np.zeros((dim, dim))
    excitation_vec = np.zeros(dim)
    # Cumulative intensity
    current_intensity_vec = background_intensity_func(t, current_state / FLLN_scaling) # intensity has no excitation before the first arrival

    # Main simulation loop
    while (t < T * FLLN_scaling) and (num_arrivals_so_far < max_size):

        # Update what the previous intensity vector was        
        previous_intensity_vec = current_intensity_vec

        # Upper bound on intensity: current intensity + dim * maximum that background intensity can achieve per dimension
        Max_intensity = np.sum(previous_intensity_vec) + dim * background_intensity_max
        
        t += rng.exponential(1/Max_intensity)
        U = rng.uniform(0,Max_intensity)

        # Calculate new background intensity (pass rescaled state)
        background_intensity_vec = background_intensity_func(t, current_state / FLLN_scaling)

        # Update intensities: re-calculate excitation matrix
        if num_arrivals_so_far > 0:

            time_diffs = t - arrival_times_array[:num_arrivals_so_far]
            past_dims = arrival_dims_array[:num_arrivals_so_far]
            past_states = arrival_states_array[:num_arrivals_so_far]

            # Pass rescaled states to user's excitation kernel function
            excitation_matrix = excitation_kernel_func(
                time_diffs,                    # Array: (n_past,) - time since each past arrival
                past_dims,                     # Array: (n_past,) - dimension of each past arrival
                past_states / FLLN_scaling     # Array: (n_past,) or (n_past, state_dim) - rescaled states
            )
            
            # Sum each row to get total excitation for each dimension
            excitation_vec = np.sum(excitation_matrix, axis=1)

        # Full intensity is background + excitation
        current_intensity_vec = excitation_vec + background_intensity_vec # total intensity for each dimension
        
        # Accept or reject arrival
        if (t < T * FLLN_scaling) and (U <= np.sum(current_intensity_vec)):

            # divide current intensity value into bins to figure out which dimension the new arrival belongs to
            cumsum_intensity_vec = np.cumsum(current_intensity_vec)
            d = np.searchsorted(cumsum_intensity_vec, U)
            arrival_type = np.zeros(dim)
            arrival_type[d] = 1
            previous_state = current_state
            if state_dim == 1:
                current_state = previous_state + state_matrix[0, d]
            else:
                current_state = previous_state + state_matrix @ arrival_type

            # If we are writing to disk, we will still retain the arrays in memory during the simulation, as reading and writing to disk for all past arrivals is too great a cost. So the minimum memory requirements for a single  simulation are the same as if we were not writing to disk, but for multiple  simulations writing to disk prevents the need to store all the arrival data  across all simulations in memory simultaneously.
            arrival_times_array[num_arrivals_so_far] = t
            arrival_dims_array[num_arrivals_so_far] = d
            if state_dim == 1:
                arrival_states_array[num_arrivals_so_far] = previous_state[0]
            else:
                arrival_states_array[num_arrivals_so_far] = previous_state
            
            num_arrivals_so_far += 1 #IMPORTANT: increment must be after the above storing since num_arrivals_so_far is used for indexing the above arrays, which are zero-indexed.

            # Add jump to intensity and update state of background intensity for next arrival
            background_intensity_vec = background_intensity_func(t, current_state / FLLN_scaling)
            
            # Pass rescaled states to user's excitation kernel function
            excitation_matrix += excitation_kernel_func(
                np.array([0]),                    # Array: (n_past,) - time since each past arrival
                np.array([d]),                     # Array: (n_past,) - dimension of each past arrival
                np.array([previous_state / FLLN_scaling])     # Array: (n_past,) or (n_past, state_dim) - rescaled states
            )
            
            # Sum each row to get total excitation for each dimension
            excitation_vec = np.sum(excitation_matrix, axis=1)

            # Full intensity is background + excitation
            current_intensity_vec = excitation_vec + background_intensity_vec # total intensity for each dimension
    
    # Truncate arrays to actual number of arrivals
    total_num_arrivals = num_arrivals_so_far
    arrival_times_array = arrival_times_array[:total_num_arrivals]
    arrival_dims_array = arrival_dims_array[:total_num_arrivals]
    arrival_states_array = arrival_states_array[:total_num_arrivals]

    if use_disk:
        output_file = os.path.join(output_dir, output_name)
        with open(output_file, 'wb') as f:
            pickle.dump((arrival_times_array, arrival_dims_array, arrival_states_array), f)
        return output_file
    else:
        return arrival_times_array, arrival_dims_array, arrival_states_array


def sim_ExpSDHawkes_once(
    dim: int,
    state_dim: int,
    state_matrix: np.ndarray,
    background_intensity_func: Callable[[float, np.ndarray], np.ndarray],
    background_intensity_max: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    r: Callable[[int, int, np.ndarray], float],
    max_arrivals: int,
    use_disk: bool,
    T: float,
    FLLN_scaling: float = 1,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    seed: Optional[Union[int, np.random.SeedSequence]] = None
) -> Union[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Simulate one sample path of a state-dependent Hawkes process with exponential kernels using efficient update formulas.
    
    This specialized implementation exploits the exponential kernel structure to avoid recomputing excitation from all past arrivals at each time step of Ogata's modified thinning algorithm. Instead, it uses a simple time-decay update (which is only possible due to the exponential structure), dramatically improving computational efficiency compared to the general algorithm.
    
    The excitation kernel has the form: 
        φ_ij(t, y) = r_ij(y) * α_ij * exp(-β_ij * t),
    where r_ij(y) provides state-dependent multiplicative scaling.
    
    Parameters
    ----------
    dim : int
        Number of dimensions (types) in the Hawkes process.
    state_dim : int
        Dimensionality of the state space. Use 1 for scalar state.
    state_matrix : np.ndarray
        State transition matrix. Shape (state_dim, dim) or (1, dim) for scalar state.
        Entry (i, j) specifies how state component i changes when dimension j has an arrival.
    background_intensity_func : Callable[[float, np.ndarray], np.ndarray]
        Function with signature (t, state) -> intensity_vector.
        Returns background intensity vector of shape (dim,) at time t given state.
        State is automatically rescaled by 1/FLLN_scaling before being passed to this function.
    background_intensity_max : float
        Maximum possible change in background intensity per dimension over any time interval.
        Used as upper bound in thinning algorithm. Set to 0 if background is constant.
    alpha : np.ndarray
        Excitation parameter matrix of shape (dim, dim).
        Entry (i, j) is the jump magnitude in intensity of dimension i when dimension j has an arrival.
    beta : np.ndarray
        Decay rate matrix of shape (dim, dim).
        Entry (i, j) is the exponential decay rate for excitation from dimension j to dimension i.
    r : Callable[[int, int, np.ndarray], float]
        State-dependence function with signature (i, j, state) -> scalar.
        Returns multiplicative scaling factor for excitation from dimension j to dimension i given state.
        State is automatically rescaled by 1/FLLN_scaling before being passed to this function.
    max_arrivals : int
        Maximum number of arrivals before terminating simulation (safety cutoff).
    use_disk : bool
        If True, write arrivals to disk at the end of the simulation. If False, keep all arrivals in memory.
    T : float
        Final simulation time. Actual simulation runs on [0, FLLN_scaling * T].
    FLLN_scaling : float, optional
        FLLN scaling parameter n. Extends time horizon to [0, n*T] and automatically rescales states by 1/n when passing to user functions. Default is 1.
    output_dir : str, optional
        Directory to save simulation output. Required if use_disk=True. Default is None.
    output_name : str, optional
        File name for simulation output (appended to output_dir). Required if use_disk=True. Default is None.
    seed : int, np.random.SeedSequence, or None, optional
        Seed for random number generator. If None, uses OS entropy. Default is None.
    
    Returns
    -------
    str or tuple
        If use_disk=True: returns full output file path (str) to saved data.
        If use_disk=False: returns tuple (arrival_times_array, arrival_dims_array, arrival_states_array) where:
            - arrival_times_array: np.ndarray of shape (n_arrivals,) with arrival times
            - arrival_dims_array: np.ndarray of shape (n_arrivals,) with dimension indices
            - arrival_states_array: np.ndarray of shape (n_arrivals,) or (n_arrivals, state_dim) with states
    
    Notes
    -----
    - Uses efficient O(dim^2) updates per arrival instead of O(n_past * dim^2) for general kernels.
    - Excitation matrix is updated via: excitation_matrix *= exp(-beta * dt) at each time step.
    - For parallel execution, r function must be defined at module level.
    
    See Also
    --------
    sim_SDHawkes_once_general : General kernel implementation (for non-exponential kernels or for exponential kernels with non-multiplicative state dependency).
    """
    ## Initialization
    rng = np.random.default_rng(seed)
    ones_vec = np.ones(dim)
    max_size = int(max_arrivals * FLLN_scaling)
    arrival_times_array = np.zeros(max_size)
    arrival_dims_array = np.zeros(max_size, dtype=int)
    if state_dim == 1:
        arrival_states_array = np.zeros(max_size)
    else:
        arrival_states_array = np.zeros((max_size, state_dim))
    # Iteratively updted variables
    t = 0
    current_state = np.zeros(state_dim)
    num_arrivals_so_far = 0 # We cut the simulation off after enough arrivals
    # Self-excitation terms
    excitation_matrix = np.zeros((dim, dim))
    excitation_vec = np.zeros(dim)
    # Cumulative intensity
    current_intensity_vec = background_intensity_func(t, current_state / FLLN_scaling) # intensity has no excitation before the first arrival
    
    while (t < T * FLLN_scaling) and (num_arrivals_so_far < max_arrivals*FLLN_scaling): #FLLN scaling because we simulate on an extended horizon when FLLN > 1.

        # Update what the previous intensity vector was        
        previous_intensity_vec = current_intensity_vec

        # Upper bound on intensity: current intensity (because we assume a constant background intensity and decaying excitation terms)
        Max_intensity = np.sum(previous_intensity_vec) + dim * background_intensity_max
        
        t_old = t
        t += rng.exponential(1/Max_intensity)
        U = rng.uniform(0,Max_intensity)

        # Calculate new intensities as time progresses
        # Update intensities: decay old excitation terms
        time_diff = t - t_old
        excitation_matrix *= np.exp(-beta * time_diff) # decay excitation terms
        excitation_vec = excitation_matrix @ ones_vec # sum rows to get total intensity for each dimension
        # Update background intensity due to time progression (pass rescaled state)
        background_intensity_vec = background_intensity_func(t, current_state / FLLN_scaling)
        current_intensity_vec = background_intensity_vec + excitation_vec # total intensity for each dimension

        # Accept or reject arrival
        if (t < T * FLLN_scaling) and (U <= np.sum(current_intensity_vec)):
            # divide current intensity value into bins to figure out which dimension the new arrival belongs to
            cumsum_intensity_vec = np.cumsum(current_intensity_vec)
            d = np.searchsorted(cumsum_intensity_vec, U)
            previous_state = current_state
            arrival_type = np.zeros(dim)
            arrival_type[d] = 1
            if state_dim == 1:
                current_state = previous_state + state_matrix[0, d]
            else:
                current_state = previous_state + state_matrix @ arrival_type

            arrival_times_array[num_arrivals_so_far] = t
            arrival_dims_array[num_arrivals_so_far] = d
            if state_dim == 1:
                arrival_states_array[num_arrivals_so_far] = previous_state[0]
            else:
                arrival_states_array[num_arrivals_so_far] = previous_state
            
            num_arrivals_so_far += 1 #IMPORTANT: increment must be after the above storing since num_arrivals_so_far is used for indexing the above arrays

            # Pass rescaled state to r function; note that state used should be the one just before the arrival
            r_vec = np.array([r(i, d, previous_state / FLLN_scaling) for i in range(dim)])
            
            # Calculate new background intensity; uses new state because in the algorithm it plays a role in the next arrival
            background_intensity_vec = background_intensity_func(t, current_state / FLLN_scaling)

            # Update intensities: add new jump
            excitation_matrix[:, d] += alpha[:, d] * r_vec # Add new jump in column d; note that we use r_vec which uses the state ust before arrival
            excitation_vec = excitation_matrix @ ones_vec # sum rows to get total intensity for each dimension
            current_intensity_vec = background_intensity_vec + excitation_vec # total intensity for each dimension
    
    # Truncate arrays to actual number of arrivals
    total_num_arrivals = num_arrivals_so_far
    arrival_times_array = arrival_times_array[:total_num_arrivals]
    arrival_dims_array = arrival_dims_array[:total_num_arrivals]
    arrival_states_array = arrival_states_array[:total_num_arrivals]

    if use_disk:
        output_file = os.path.join(output_dir, output_name)
        with open(output_file, 'wb') as f:
            pickle.dump((arrival_times_array, arrival_dims_array, arrival_states_array), f)
        return output_file
    else:
        return arrival_times_array, arrival_dims_array, arrival_states_array

def sim_MultiExpSDHawkes_once(dim: int,
    state_dim: int,
    state_matrix: np.ndarray,
    background_intensity_func: Callable[[float, np.ndarray], np.ndarray],
    background_intensity_max: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    r: Callable[[int, int, np.ndarray], float],
    max_arrivals: int,
    use_disk: bool,
    T: float,
    FLLN_scaling: float = 1,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    seed: Optional[Union[int, np.random.SeedSequence]] = None):
    """
    In this version, we simulate another special Markovian case of SDHawkes. In this setting, the excitation is given by:
        φ_ij(t, y) = r_ijk(y) * sum_{k=1}^L α_ijk * exp(-β_ijk * t)
    where r_ijk(y) provides state-dependent multiplicative scaling. This is a generalization of ExpSDHawkes, and useful because sums of exponentials can, to some extent, approximate more complex functions such as power-law decays (see "Optimal approximations of power-laws with exponentials", 2006, by Bochud and Challet).

    Parameters
    ----------
    dim : int
        Number of dimensions (types) in the Hawkes process.
    state_dim : int
        Dimensionality of the state space. Use 1 for scalar state.
    state_matrix : np.ndarray
        State transition matrix. Shape (state_dim, dim) or (1, dim) for scalar state.
        Entry (i, j) specifies how state component i changes when dimension j has an arrival.
    background_intensity_func : Callable[[float, np.ndarray], np.ndarray]
        Function with signature (t, state) -> intensity_vector.
        Returns background intensity vector of shape (dim,) at time t given state.
        State is automatically rescaled by 1/FLLN_scaling before being passed to this function.
    background_intensity_max : float
        Maximum possible change in background intensity per dimension over any time interval.
        Used as upper bound in thinning algorithm. Set to 0 if background is constant.
    alpha : np.ndarray
        Excitation parameter matrix of shape (dim, dim, third_dim).
        Entry (i, j, k) is the jump magnitude in intensity of dimension i when dimension j.
    beta : np.ndarray
        Decay rate matrix of shape (dim, dim, third_dim).
        Entry (i, j, k) is the exponential decay rate for excitation from dimension j to dimension i.
    r : Callable[[int, int, int, np.ndarray], float]
        State-dependence function with signature (i, j, k, state) -> scalar, where (i,j,k) in [dim] x [dim] x [third_dim]
        Returns multiplicative scaling factor for excitation from dimension j to dimension i, along the kth summand, given state.
        State is automatically rescaled by 1/FLLN_scaling before being passed to this function.
    max_arrivals : int
        Maximum number of arrivals before terminating simulation (safety cutoff).
    use_disk : bool
        If True, write arrivals to disk at the end of the simulation. If False, keep all arrivals in memory.
    T : float
        Final simulation time. Actual simulation runs on [0, FLLN_scaling * T].
    FLLN_scaling : float, optional
        FLLN scaling parameter n. Extends time horizon to [0, n*T] and automatically rescales states by 1/n when passing to user functions. Default is 1.
    output_dir : str, optional
        Directory to save simulation output. Required if use_disk=True. Default is None.
    output_name : str, optional
        File name for simulation output (appended to output_dir). Required if use_disk=True. Default is None.
    seed : int, np.random.SeedSequence, or None, optional
        Seed for random number generator. If None, uses OS entropy. Default is None.
    
    Returns
    -------
    str or tuple
        If use_disk=True: returns full output file path (str) to saved data.
        If use_disk=False: returns tuple (arrival_times_array, arrival_dims_array, arrival_states_array) where:
            - arrival_times_array: np.ndarray of shape (n_arrivals,) with arrival times
            - arrival_dims_array: np.ndarray of shape (n_arrivals,) with dimension indices
            - arrival_states_array: np.ndarray of shape (n_arrivals,) or (n_arrivals, state_dim) with states
    """
    ## Initialization
    rng = np.random.default_rng(seed)
    third_dim = np.shape(alpha)[2]
    ones_vec = np.ones(dim)
    max_size = int(max_arrivals * FLLN_scaling)
    arrival_times_array = np.zeros(max_size)
    arrival_dims_array = np.zeros(max_size, dtype=int)
    if state_dim == 1:
        arrival_states_array = np.zeros(max_size)
    else:
        arrival_states_array = np.zeros((max_size, state_dim))
    # Iteratively updted variables
    t = 0
    current_state = np.zeros(state_dim)
    num_arrivals_so_far = 0 # We cut the simulation off after enough arrivals
    # Self-excitation terms
    excitation_tensor = np.zeros((dim, dim, third_dim))
    excitation_vec = np.zeros(dim)
    # Cumulative intensity
    current_intensity_vec = background_intensity_func(t, current_state / FLLN_scaling) # intensity has no excitation before the first arrival
    
    while (t < T * FLLN_scaling) and (num_arrivals_so_far < max_arrivals*FLLN_scaling): #FLLN scaling because we simulate on an extended horizon when FLLN > 1.

        # Update what the previous intensity vector was        
        previous_intensity_vec = current_intensity_vec

        # Upper bound on intensity: current intensity (because we assume a constant background intensity and decaying excitation terms)
        Max_intensity = np.sum(previous_intensity_vec) + dim * background_intensity_max
        
        t_old = t
        t += rng.exponential(1/Max_intensity)
        U = rng.uniform(0,Max_intensity)

        # Calculate new intensities as time progresses
        # Update intensities: decay old excitation terms
        time_diff = t - t_old

        excitation_tensor *= np.exp(-beta * time_diff) # decay excitation terms (now along 3-dimensions)
        excitation_vec = excitation_tensor.sum(axis=2) @ ones_vec # sum rows to get total intensity for each dimension
        # Update background intensity due to time progression (pass rescaled state)
        background_intensity_vec = background_intensity_func(t, current_state / FLLN_scaling)
        current_intensity_vec = background_intensity_vec + excitation_vec # total intensity for each dimension

        # Accept or reject arrival
        if (t < T * FLLN_scaling) and (U <= np.sum(current_intensity_vec)):
            # divide current intensity value into bins to figure out which dimension the new arrival belongs to
            cumsum_intensity_vec = np.cumsum(current_intensity_vec)
            d = np.searchsorted(cumsum_intensity_vec, U)
            previous_state = current_state
            arrival_type = np.zeros(dim)
            arrival_type[d] = 1
            if state_dim == 1:
                current_state = previous_state + state_matrix[0, d]
            else:
                current_state = previous_state + state_matrix @ arrival_type

            arrival_times_array[num_arrivals_so_far] = t
            arrival_dims_array[num_arrivals_so_far] = d
            if state_dim == 1:
                arrival_states_array[num_arrivals_so_far] = previous_state[0]
            else:
                arrival_states_array[num_arrivals_so_far] = previous_state
            
            num_arrivals_so_far += 1 #IMPORTANT: increment must be after the above storing since num_arrivals_so_far is used for indexing the above arrays

            # Calculate new background intensity; uses new state because in the algorithm it plays a role in the next arrival
            background_intensity_vec = background_intensity_func(t, current_state / FLLN_scaling)

            # Pass rescaled state to r function; note that state used should be the one just before the arrival
            r_mat = np.array([[r(i, d, k, previous_state / FLLN_scaling) for k in range(third_dim)] for i in range(dim)]) # shape (dim, L)
            # Update intensities: add new jump
            excitation_tensor[:, d, :] += alpha[:, d, :] * r_mat # Add new jump in column d; note that we use r_mat which uses the state ust before arrival
            excitation_vec = excitation_tensor.sum(axis=2) @ ones_vec # sum rows to get total intensity for each dimension
            
            current_intensity_vec = background_intensity_vec + excitation_vec # total intensity for each dimension
    
    # Truncate arrays to actual number of arrivals
    total_num_arrivals = num_arrivals_so_far
    arrival_times_array = arrival_times_array[:total_num_arrivals]
    arrival_dims_array = arrival_dims_array[:total_num_arrivals]
    arrival_states_array = arrival_states_array[:total_num_arrivals]

    if use_disk:
        output_file = os.path.join(output_dir, output_name)
        with open(output_file, 'wb') as f:
            pickle.dump((arrival_times_array, arrival_dims_array, arrival_states_array), f)
        return output_file
    else:
        return arrival_times_array, arrival_dims_array, arrival_states_array

    pass

def sim_ExpSAHawkes_once(
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    dim: int,
    max_arrivals: int,
    use_disk: bool,
    T: float,
    FLLN_scaling: float = 1,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    seed: Optional[Union[int, np.random.SeedSequence]] = None
) -> Union[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Simulate one sample path of a state-agnostic Hawkes process with exponential kernels using Ogata's thinning algorithm.
    
    This is a module-level function designed to be picklable for multiprocessing. It implements a standard (non-state-dependent) multivariate Hawkes process where intensity depends only on past arrivals, not on any evolving state. Uses exponential kernels for efficient O(dim^2) updates per arrival. This is equivalent to sim_ExpSDHawkes_once() when r_ij(y) = 1 for all i,j.
    
    The intensity for dimension i at time t is:
        λ_i(t) = μ_i + Σ_j ∫_0^t α_{ij} * exp(-β_{ij} * (t - s)) dN_j(s)
    
    Parameters
    ----------
    mu : np.ndarray
        Background intensity vector of shape (dim,).
    alpha : np.ndarray
        Excitation matrix of shape (dim, dim).
        Entry (i,j) is the jump in λ_i when dimension j has an arrival.
    beta : np.ndarray
        Decay rate matrix of shape (dim, dim).
        Entry (i,j) is the exponential decay rate for excitation from j to i.
    dim : int
        Number of dimensions (types) in the Hawkes process.
    max_arrivals : int
        Maximum number of arrivals before terminating simulation (safety cutoff).
    use_disk : bool
        If True, write arrivals to disk at the end of simulation. If False, all arrival information returned as arrays (and thus remain in memory).
    T : float
        Final simulation time. Actual simulation runs on [0, FLLN_scaling * T].
    FLLN_scaling : float, optional
        Scaling parameter. Extends time horizon to [0, FLLN_scaling * T].
        For state-agnostic processes this only affects the time horizon (no state rescaling).
        Default is 1.
    output_dir : str, optional
        Directory path where output file will be saved. Required if use_disk=True. Default is None.
    output_name : str, optional
        Name of output file (e.g., "sim_0000.pkl"). Required if use_disk=True. Default is None.
    seed : int, np.random.SeedSequence, or None, optional
        Seed for random number generator. If None, uses OS entropy. Default is None.
    
    Returns
    -------
    str or tuple
        If use_disk=True: returns full output file path (str) where data was saved.
            Disk format: pickle file containing tuple (arrival_times_array, arrival_dims_array).
        If use_disk=False: returns tuple (arrival_times_array, arrival_dims_array) where:
            - arrival_times_array: np.ndarray of shape (n_arrivals,) with arrival times
            - arrival_dims_array: np.ndarray of shape (n_arrivals,) with dimension indices (0 to dim-1)
    
    Notes
    -----
    - Uses efficient O(dim^2) matrix exponential decay updates per arrival.
    - Excitation matrix is updated via: excitation_matrix *= exp(-beta * dt) at each time step.
    - For parallel execution via multiprocessing, this function must remain at module level.
    
    See Also
    --------
    sim_ExpSDHawkes_once : State-dependent exponential kernel implementation.
    sim_SDHawkes_once_general : General kernel implementation.
    """
    ## Initialization
    rng = np.random.default_rng(seed)
    ones_vec = np.ones(dim)
    T_eff = T * FLLN_scaling
    max_size = int(max_arrivals * FLLN_scaling)
    arrival_times_array = np.zeros(max_size)
    arrival_dims_array = np.zeros(max_size, dtype=int)

    # Iteratively updated variables
    num_arrivals_so_far = 0
    current_time = 0.0
    excitation_matrix = np.zeros((dim, dim))
    current_intensities_vec = mu.copy()

    while current_time < T_eff and num_arrivals_so_far < max_size:

        previous_intensities_vec = current_intensities_vec

        # Upper bound on intensity: sum of current intensities (valid because background is constant and excitation decays)
        Max_intensity = np.sum(previous_intensities_vec)

        # Generate next candidate arrival time
        next_time = current_time + rng.exponential(scale=1.0 / Max_intensity)

        # Decay excitation terms
        time_diff = next_time - current_time
        excitation_matrix *= np.exp(-beta * time_diff)
        current_intensities_vec = mu + excitation_matrix @ ones_vec

        # Accept or reject the arrival
        U = rng.uniform(0, Max_intensity)
        if U <= np.sum(current_intensities_vec) and next_time <= T_eff:

            # Determine which dimension the arrival belongs to
            cumsum_intensity_vec = np.cumsum(current_intensities_vec)
            d = np.searchsorted(cumsum_intensity_vec, U)

            arrival_times_array[num_arrivals_so_far] = next_time
            arrival_dims_array[num_arrivals_so_far] = d
            num_arrivals_so_far += 1

            # Update excitation: add new jump in column d
            excitation_matrix[:, d] += alpha[:, d]
            excitation_vec = excitation_matrix @ ones_vec
            current_intensities_vec = mu + excitation_vec

        # Update current time (whether arrival was accepted or rejected)
        current_time = next_time

    # Write any remaining arrivals in buffer
    if use_disk:
        output_file = os.path.join(output_dir, output_name)
        with open(output_file, 'wb') as f:
            pickle.dump((arrival_times_array, arrival_dims_array), f)
        return output_file
    else:
        # Truncate arrays to actual number of arrivals
        total_num_arrivals = num_arrivals_so_far
        return arrival_times_array[:total_num_arrivals], arrival_dims_array[:total_num_arrivals]


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## MODULE-LEVEL FUNCTIONS: Parallel simulation infrastructure

def _data_for_parallel_sims(
    num_sims: int,
    use_disk: bool = True,
    subfolder_path: Optional[str] = None,
    child_seeds: Optional[List[np.random.SeedSequence]] = None
) -> List[Tuple[Optional[int], Optional[str], Optional[np.random.SeedSequence]]]:
    """
    Generate per-simulation data tuples for the parallel simulation framework.
    
    Creates a list of data tuples, one per simulation, containing the simulation index, output directory path, and random seed. This data is passed to worker processes via the _sim_wrapper function.
    
    Parameters
    ----------
    num_sims : int
        Number of simulations to prepare data for.
    use_disk : bool, optional
        If True, includes simulation indices for unique filenames in disk mode.
        If False (memory mode), indices and paths are None. Default is True.
    subfolder_path : str, optional
        Path to subfolder where simulation files should be saved.
        Only used if use_disk=True. Default is None.
    child_seeds : list of np.random.SeedSequence, optional
        Pre-spawned child seeds from SeedSequence.spawn(), one per simulation.
        If None, seeds will be None (OS entropy used). Default is None.

    Returns
    -------
    list of tuple
        List of length num_sims where each element is a 3-tuple:
        (sim_index_or_None, subfolder_path_or_None, seed_or_None).
        In disk mode: (int, str, SeedSequence).
        In memory mode: (None, None, SeedSequence).
    
    Notes
    -----
    This function is designed to work with _sim_wrapper which unpacks these tuples.
    """
    if child_seeds is None:
        child_seeds = [None] * num_sims
    if use_disk:
        return [(i, subfolder_path, child_seeds[i]) for i in range(num_sims)]
    else:
        # Always include index for sorting after parallel execution
        return [(i, None, child_seeds[i]) for i in range(num_sims)]

def _process_single_sim(
    func_and_params: Tuple[Callable, Tuple[Optional[int], Optional[str], Optional[np.random.SeedSequence]]]
) -> Union[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Top-level helper that multiprocessing.Pool maps over. Unpacks the (sim_func, per_sim_data) tuple and dispatches to sim_func.
    
    Unpacks the (sim_func, per_sim_data) tuple and calls sim_func with per_sim_data.
    Must be a module-level function (not a lambda or nested function) to be picklable
    by multiprocessing.
    
    Parameters
    ----------
    func_and_params : tuple
        2-tuple of (sim_func, per_sim_data) where:
        - sim_func: Callable, partially applied _sim_wrapper with T and FLLN_scaling bound
        - per_sim_data: 3-tuple (sim_index, subfolder_path, seed)
        
    Returns
    --------
    tuple
        (sim_index, result) where result is:
        - If disk mode: output filename (str)
        - If memory mode: (arrival_times_array, arrival_dims_array, arrival_states_array) tuple
    """
    sim_func, per_sim_data = func_and_params
    sim_index = per_sim_data[0]  # Extract index from per_sim_data
    result = sim_func(per_sim_data)
    return (sim_index, result)

def _sim_wrapper(
    sim_partial: Callable,
    T: float,
    FLLN_scaling: float,
    per_sim_data: Tuple[Optional[int], Optional[str], Optional[np.random.SeedSequence]]
) -> Union[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Wrapper that constructs output file paths and calls the simulation function.

    This function is designed to be used with functools.partial to bind sim_partial, T, and FLLN_scaling, leaving only per_sim_data as the free parameter that varies across simulations.
    
    Parameters
    ----------
    sim_partial : functools.partial or Callable
        Partially applied module-level simulation function (sim_SDHawkes_once_general or sim_ExpSDHawkes_once) with all instance-level parameters already bound.
        Expected signature: sim_partial(T, FLLN_scaling, output_dir, output_name, seed) -> result
    T : float
        Final simulation time (before FLLN scaling).
    FLLN_scaling : float
        FLLN scaling parameter n. Actual simulation runs on [0, n*T].
    per_sim_data : tuple
        3-tuple of (sim_index_or_None, subfolder_path_or_None, seed_or_None).
        If sim_index is not None: disk mode, constructs output_name = "sim_{sim_index:04d}.pkl".
        If sim_index is None: memory mode, output_dir and output_name are None.

    Returns
    -------
    str or tuple
        Disk mode: returns full output file path (str).
        Memory mode: returns tuple (arrival_times_array, arrival_dims_array, arrival_states_array).
    
    Notes
    -----
    This wrapper enables pickling for multiprocessing by using functools.partial
    """
    sim_index, subfolder_path, seed = per_sim_data
    if sim_index is not None:
        output_dir = subfolder_path
        output_name = f"sim_{sim_index:04d}.pkl"
    else:
        output_dir = None
        output_name = None
    return sim_partial(T, FLLN_scaling, output_dir, output_name, seed)

def _run_parallel_sims(
    data: List[Tuple[Optional[int], Optional[str], Optional[np.random.SeedSequence]]],
    sim_func: Callable,
    num_workers: int
) -> List[Union[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Run multiple Hawkes simulations in parallel using multiprocessing.Pool.
    
    Distributes simulations across worker processes with a progress bar.
    Each worker receives independent random seeds to ensure reproducible, non-overlapping random streams.
    
    Parameters
    ----------
    data : list of tuple
        List of per-simulation data from _data_for_parallel_sims.
        Each element is a 3-tuple (sim_index, subfolder_path, seed).
    sim_func : callable
        Partially applied _sim_wrapper with sim_partial, T, and FLLN_scaling bound.
        Remaining signature: sim_func(per_sim_data) -> result
    num_workers : int
        Number of parallel worker processes to use
        
    Returns:
    --------
    list
        List of simulation results with length equal to len(data).
        If using disk mode: list of output filenames (str)
        If using memory mode: list of (arrival_times_array, arrival_dims_array, arrival_states_array) tuples
    """
    total_sims = len(data)
    print(f"\nRunning {total_sims} parallel simulations with {num_workers} workers...")
    
    # Pair each per-sim data item with the simulation function
    func_and_params = [(sim_func, per_sim_data) for per_sim_data in data]
    
    # Use imap_unordered for better performance, then sort by index to preserve order
    with Pool(processes=num_workers) as pool:
        indexed_results = list(tqdm(
            pool.imap_unordered(_process_single_sim, func_and_params),
            total=total_sims,
            desc="Simulations progress",
            dynamic_ncols=True,
            mininterval=0.1
        ))
    
    # Sort results by simulation index to ensure consistent ordering
    indexed_results.sort(key=lambda x: x[0])
    
    # Extract just the results (drop indices)
    result_list = [result for _, result in indexed_results]
    
    return result_list

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class SDHawkes:

    def __init__(self, background_intensity_func: Callable, 
                 excitation_kernel_func: Callable, 
                 background_intensity_max: float, 
                 state_matrix: np.ndarray,
                 max_arrivals: int = 500000,
                 num_workers: int = 1,
                 use_disk: bool = True
                 ) -> None:
        """
        Initialize a State-Dependent Hawkes Process simulator.
        
        This class implements a general framework for simulating multivariate state-dependent Hawkes processes using the Poisson thinning algorithm. The state-dependency allows the intensity to depend on the current state of the process (e.g., difference between component counts), enabling more realistic modeling of self-exciting phenomena.
        
        Parameters:
        -----------
        background_intensity_func : Callable
            Function that computes the background intensity vector for each dimension.
            Signature: background_intensity_func(t, state) -> np.ndarray
            
            Arguments:
                - t (float): Current time
                - state (float or np.ndarray): Current state value (e.g., N_1 - N_2 for 2D SDHawkes with 1D state)
            
            Returns:
                - np.ndarray of shape (dim,): Background intensity for each dimension
            
            Example: For constant background mu = [200, 120], this could be:
                lambda t, state: mu * np.array([r(state, 1), r(state, 2)])
        
        excitation_kernel_func : Callable
            Function that computes the full excitation matrix from the history of arrivals.
            Signature: excitation_kernel_func(time_diffs, past_dims, past_states) -> np.ndarray
            
            IMPORTANT: This function must be vectorized to process all past arrivals at once. This function is called once per intensity update to compute the full (dim x dim) excitation matrix, and so should be optimized for efficiency.
            
            Arguments:
                - time_diffs (np.ndarray): Time elapsed since each past arrival, shape (n_past,)
                  Index k gives time since the k-th arrival (k=0 is first arrival)
                - past_dims (np.ndarray): Dimension of each past arrival, shape (n_past,)
                  Index k gives dimension (0 to dim-1) of the k-th arrival
                - past_states (np.ndarray): State just before each past arrival
                  If state_dim=1: shape (n_past,), index k gives scalar state before k-th arrival
                  If state_dim>1: shape (n_past, state_dim), index k gives state vector before k-th arrival
            
            Returns:
                - np.ndarray: Excitation matrix, shape (dim, dim)
                  Entry [i,j] is the total excitation to dimension i from all past arrivals in dimension j
            
            
            Example: For exponential kernel with multiplicative state dependence via a function r (in practice the user should use the optimized Exponential state-dependence Hawkes subclass instead):
                def kernel(time_diffs, past_dims, past_states):
                    # Compute φ_{i,j} = sum over past events of α_{i,j} * r(state) * exp(-β_{i,j} * time_diff)
                    # Return (dim x dim) matrix
        
        background_intensity_max : float
            Maximum value of the background intensity over all time and states, for all components.
            This is used as an upper bound in the Poisson thinning algorithm.
            
            CRITICAL: This must be a valid upper bound for the background intensity.
            If background_intensity_func(t, state, n) can exceed this value for any (t, state, n), the simulation will be incorrect.
            
            For constant background: background_intensity_max = sum(mu)
            For time-varying background: background_intensity_max = max_t sum(mu(t))
        
        state_matrix : np.ndarray
            Matrix defining how the count vector maps to the state process.
            
            Shape options:
            - 1D array of shape (dim,): Treated as a row vector, resulting in scalar state
            - 2D array of shape (state_dim, dim): General transformation matrix
            
            The state is computed as: state = state_matrix @ count_vector
            
            Examples:
            - For 2D process with scalar state = N_1 - N_2:
                state_matrix = np.array([1, -1])  # 1D, gives scalar state
            - For 2D process with vector state = [N_1, N_2]:
                state_matrix = np.eye(2)  # 2D identity matrix
            
            When an arrival occurs in dimension d, the state is updated by adding 
            state_matrix[:, d] to the current state.
            
        max_arrivals: int = 500000,
            Maximum number of arrivals for each simulation; defaults to 500000
            
        num_workers: int = 1,
            Number of worker processes for parallel simulation; defaults to 1 (no parallelization)
            
        use_disk: bool = True,
            Whether to use disk-based storage for large datasets; defaults to True. NOTE: if simulating large datasets and this is set to False, then your hardware needs to have enough memory to store the entire dataset (all paths) in RAM or your computer may crash. Some simulations can take up 2GB **per path**, depending on how unstable the process is and how large the background intensity is, and if simulating on a long time horizon (or using large FLLN scaling).
            
        Attributes:
        -----------
        background_intensity_func : Callable
            Stored background intensity function
        excitation_kernel_func : Callable
            Stored excitation kernel function
        background_intensity_max : float
            Stored maximum background intensity
        state_matrix : np.ndarray
            Stored state update matrix
        dim : int
            Number of dimensions in the Hawkes process, inferred from the output shape of background_intensity_func(0)
        max_arrivals: int = 500000,
            Maximum number of arrivals for each simulation
        num_workers: int = 1,
            Number of worker processes for parallel simulation
        use_disk: bool = False,
            Whether to use disk-based storage for large datasets
        
        Notes:
        ------
        - The Ogata modified thinning algorithm requires that the intensity is bounded above by a constant Max_background_intensity on each inter-arrival interval. This implementation assumes:
          1. Background intensity is bounded by background_intensity_max
          2. Excitation terms are **non-increasing** between arrivals
        
        - For FLLN (Functional Law of Large Numbers) analysis, the FLLN_scaling parameter n is used to scale the simulation horizon and state. The simulation automatically passes rescaled states (state/n) to user-provided functions, demonstrating ucp convergence to a deterministic limit as n → ∞.
        
        Examples:
        ---------
        >>> # 2D Hawkes with exponential kernels and state-dependent background
        >>> mu = np.array([200, 120])
        >>> alpha = np.array([[0.5, 0.3], [0.3, 0.5]])
        >>> beta = np.ones_like(alpha)
        >>> delta = 0.001
        >>> 
        >>> def r(state, i, n):
        >>>     if abs(state) < 1e-10:
        >>>         return 1
        >>>     if i == 1:
        >>>         return (1+delta)**(-state/n)
        >>>     else:
        >>>         return (1+delta)**(state/n)
        >>> 
        >>> def background(t, state, n):
        >>>     return mu * np.array([r(state, 1, n), r(state, 2, n)])
        >>> 
        >>> def kernel(dt, i, j, state):
        >>>     return alpha[i,j] * beta[i,j] * np.exp(-beta[i,j] * dt)
        >>> 
        >>> sim = SDHawkes(
        >>>     background_intensity_func=background,
        >>>     excitation_kernel_func=kernel,
        >>>     background_intensity_max=np.sum(mu) * (1+delta),
        >>>     state_matrix=np.array([1, -1])
        >>> )
        """
        
        self.background_intensity_func = background_intensity_func
        self.excitation_kernel_func = excitation_kernel_func
        self.background_intensity_max = background_intensity_max
        self.max_arrivals = max_arrivals
        self.num_workers = num_workers
        self.use_disk = use_disk

        # Infer dimension of point process from state_matrix (which must multiply with the counting process vector, and hence have the same number of columns as the dimension of the counting process)
        if state_matrix.ndim == 1:
            self.dim = np.shape(state_matrix)[0]
        elif state_matrix.ndim == 2:
            self.dim = np.shape(state_matrix)[1]
        else:
            raise ValueError("state_matrix must be 1D or 2D array")

        # Normalize state_matrix to 2D array
        if state_matrix.ndim == 1:
            # Convert 1D vector to row vector (1, dim)
            self.state_matrix = state_matrix.reshape(1, -1)
            self.state_dim = 1
        elif state_matrix.ndim == 2:
            self.state_matrix = state_matrix
            self.state_dim = state_matrix.shape[0]
        else:
            raise ValueError("state_matrix must be 1D or 2D array")
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ## Simulation Functions

    def _make_sim_partial(self):
        """ 
        Create a picklable partial of the module-level simulation function with all fixed (instance-level) parameters bound.
        
        The returned callable has signature:
            sim_partial(T, FLLN_scaling, output_dir, output_name, seed) -> result
        
        This is used by sim() to create a function that can be safely sent to worker processes without pickling `self` or any bound methods.
        
        IMPORTANT: For this to be picklable by multiprocessing, all user-provided callables (background_intensity_func, excitation_kernel_func) must be defined at module level in the user's script (not lambdas or nested functions).
        
        Returns
        -------
        functools.partial
            Partial of sim_SDHawkes_once_general with fixed instance parameters bound.
            Remaining free parameters: T (float), FLLN_scaling (float), output_dir (str or None), output_name (str or None), seed (SeedSequence or None)
        """
        return partial(sim_SDHawkes_once_general,
            self.dim, self.state_dim, self.state_matrix,
            self.background_intensity_func, self.background_intensity_max,
            self.excitation_kernel_func, self.max_arrivals, self.use_disk)


    def sim(self, T:float, num_paths:int, FLLN_scaling:float=1, output_dir:str=None, external_info:dict=None, base_seed=None):
        """
        Run (possibly FLLN-scaled) parallel simulations of the state-dependent Hawkes process.
        
        Simulates num_paths independent paths on [0, FLLN_scaling * T], with states rescaled by 1/FLLN_scaling inside user functions. Uses multiprocessing for parallelism when num_workers > 1. Results are returned in a deterministic order regardless of completion order.
        
        Parameters
        ----------
        T : float
            Final time for the FLLN-scaled simulation (before scaling).
        num_paths : int
            Number of independent simulation paths to generate.
        FLLN_scaling : float, optional
            Scaling parameter n for FLLN. Actual simulation runs on [0, n*T]. Default is 1.
        output_dir : str, optional
            Directory for output files. Required if use_disk=True. A timestamped subfolder will be created.
        external_info : dict, optional
            User-provided parameters to write to the simulation parameter file (e.g., alpha, beta, mu). 
            Only used if use_disk=True. Default is None.
        base_seed : int, np.random.SeedSequence, or None, optional
            Base seed for reproducible random number generation. If provided, spawns independent child seeds for each simulation path using np.random.SeedSequence. If None, uses OS entropy. Default is None.
            
        Returns
        -------
        tuple or list
            If use_disk=True:
                tuple of (results_list, subfolder_path) where:
                - results_list: list of str, output file paths for each simulation
                - subfolder_path: str, path to the timestamped subfolder containing all simulation files
            If use_disk=False:
                list of tuples, one per simulation path, where each tuple is:
                (arrival_times_array, arrival_dims_array, arrival_states_array)
        
        Notes
        -----
        - Uses imap_unordered for parallel execution efficiency, then sorts results by simulation index.
        - Independent random seeds ensure reproducible, non-overlapping random streams across workers.
        - Background intensity and excitation functions must be defined at module level for multiprocessing.
        """
        # Validate output_dir for disk mode
        if self.use_disk and output_dir is None:
            raise ValueError("output_dir must be provided when use_disk=True")

        # Create timestamped subfolder if using disk
        subfolder_path = None
        if self.use_disk:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            subfolder_name = f"FLLN_n{FLLN_scaling}_T{T}_paths{num_paths}_{timestamp}"
            subfolder_path = os.path.join(output_dir, subfolder_name)
            os.makedirs(subfolder_path, exist_ok=True)
            print(f"Created simulation folder: {subfolder_path}")
            
            # Write simulation parameters to text file
            param_file = self.write_simulation_parameters(subfolder_path, output_dir, FLLN_scaling, T, num_paths, timestamp, external_info)
            print(f"Saved simulation parameters to: {param_file}")
        
        # Build a picklable partial of the module-level simulation function with all instance parameters to remove references to self.
        sim_partial = self._make_sim_partial()
        
        # Pass sim_partial into a partially evaluated _sim_wrapper along with other function-call-specific parameters.
        sim_func = partial(_sim_wrapper, sim_partial, T, FLLN_scaling)
        
        # Spawn independent child seeds from SeedSequence for each simulation path
        ss = np.random.SeedSequence(base_seed)
        child_seeds = ss.spawn(num_paths)
        
        # Generate per-simulation data (sim indices + seeds for disk mode, seeds only for memory mode)
        data = _data_for_parallel_sims(num_paths, use_disk=self.use_disk, subfolder_path=subfolder_path, child_seeds=child_seeds)
        
        # Run parallel simulations
        results_list = _run_parallel_sims(data, sim_func, self.num_workers)

        if self.use_disk:
            return results_list, subfolder_path
        else:
            return results_list

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ## Plotting and other output functions

    def write_simulation_parameters(self, subfolder_path, output_dir, FLLN_scaling, T, num_paths, timestamp, external_info=None):
        """
        Write simulation parameters to a text file in the specified subfolder.
        
        Parameters:
        -----------
        subfolder_path : str
            Path to the subfolder where the parameter file should be saved
        output_dir : str
            Root output directory (passed from sim(), not stored on self)
        FLLN_scaling : float
            FLLN scaling parameter n
        T : float
            Final simulation time
        num_paths : int
            Number of simulation paths
        timestamp : str
            Timestamp string for the simulation run
        external_info : dict, optional
            Dictionary containing additional external information to be included in the parameter file. This should contain user-specific parameters related to background intensity and excitation functions (e.g., alpha, beta, mu, delta, or any other custom parameters). If None, only class attributes are written.
            
        Returns:
        --------
        str
            Path to the created parameter file
        """

        # Extract class attributes
        dim = self.dim
        state_dim = self.state_dim
        background_intensity_max = self.background_intensity_max
        state_matrix = self.state_matrix
        max_arrivals = self.max_arrivals
        num_workers = self.num_workers
        use_disk = self.use_disk
        
        param_file = os.path.join(subfolder_path, "simulation_parameters.txt")
        
        with open(param_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("FLLN SIMULATION PARAMETERS\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("SIMULATION METADATA\n")
            f.write("-" * 70 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Subfolder: {os.path.basename(subfolder_path)}\n\n")
            
            f.write("FLLN PARAMETERS\n")
            f.write("-" * 70 + "\n")
            f.write(f"FLLN Scaling (n): {FLLN_scaling}\n")
            f.write(f"Final Time (T): {T}\n")
            f.write(f"Number of Paths: {num_paths}\n")
            f.write(f"Simulated Time Horizon: [0, {FLLN_scaling * T}]\n")
            f.write(f"Rescaled Time Horizon: [0, {T}]\n\n")
            
            f.write("HAWKES PROCESS CLASS PARAMETERS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Dimension (dim): {dim}\n")
            f.write(f"State Dimension (state_dim): {state_dim}\n")
            f.write(f"Background Intensity Max: {background_intensity_max}\n\n")
            
            f.write(f"State Transformation Matrix:\n")
            for row in state_matrix:
                f.write(f"  {row}\n")
            f.write("\n")
            
            f.write("SIMULATION INFRASTRUCTURE PARAMETERS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Output Directory: {output_dir}\n")
            f.write(f"Maximum Arrivals per Simulation: {max_arrivals}\n")
            f.write(f"Number of Workers (parallel threads): {num_workers}\n")
            f.write(f"Use Disk Storage: {use_disk}\n\n")
            
            # Write external user-provided information
            if external_info is not None and len(external_info) > 0:
                f.write("USER-PROVIDED PARAMETERS\n")
                f.write("-" * 70 + "\n")
                f.write("(Parameters specific to background intensity and excitation functions)\n\n")
                
                for key, value in external_info.items():
                    # Handle different types of values
                    if isinstance(value, np.ndarray):
                        f.write(f"{key}:\n")
                        if value.ndim == 1:
                            f.write(f"  {value}\n")
                        else:
                            for row in value:
                                f.write(f"  {row}\n")
                        f.write("\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
            
            f.write("=" * 70 + "\n")
        
        return param_file

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    ### ODE Solving

    def system_of_odes(self, t:float, Lambda:np.ndarray, H_func:Callable, background_intensity_func:Callable) -> np.ndarray:
        """
        Define the system of ODEs: dΛ/dt = (I - H_func(P(t)))^(-1) * μ(t, P(t))
        where P(t) = state_matrix @ Lambda(t)
        
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
        # Calculate P(t) = state_matrix @ Lambda(t)
        P = self.state_matrix @ Lambda
        
        # Calculate H_func(P(t))
        H_P = H_func(P)
        
        ## Create identity matrix
        I = np.eye(self.dim)
        # Calculate (I - H_func(P(t)))^(-1)
        inv_term = np.linalg.inv(I - H_P)
        
        # Calculate background intensity
        background = background_intensity_func(t, P, FLLN_scaling=1) # FLLN scaling is set to 1 for the purposes of solving ODEs
        
        # Calculate final result
        dLambda_dt = inv_term @ background
        
        return dLambda_dt

    def solve_ode(self, t_span: tuple, Lambda_0: np.ndarray, H_func: Callable, background_intensity_func: Callable,
              t_eval: np.ndarray = None, method: str = 'DOP853', rtol: float = 1e-8, 
              atol: float = 1e-8, max_step: float = 0.01, **kwargs) -> dict:
        """
        Solve the system of ODEs using scipy.integrate.solve_ivp
        
        Parameters:
        -----------
        t_span : tuple
            Interval of integration (t0, tf)
        Lambda_0 : np.ndarray
            Initial condition Λ(0)
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
            - Lambda: Solution array of dimension self.dim
            - P: Array of P(t) = MΛ(t) values where M is state_matrix
            - success: Whether the integration was successful
        """
        # Define the ODE function with fixed H_func and mu
        def ode_fn(t, y):
            return self.system_of_odes(t, y, H_func, background_intensity_func)
        
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
    ## External Helper Functions

    def build_paths_list(self, arrival_times_array, arrival_dims_array, arrival_states_array):
        """
        Build a list of file paths for storing simulation results.
        
        Parameters:
        -----------
        arrival_times_array : numpy.ndarray
            Array of arrival times
        arrival_dims_array : numpy.ndarray
            Array of arrival dimensions
        arrival_states_array : numpy.ndarray
            Array of arrival states
            
        Returns:
        --------
        paths : list
            List of dim lists, where paths[i] contains arrival times for dimension i 
            Format: [[t1, t2, ...], [t1, t2, ...]]
        full_information : list
            List of tuples (t, i, state) for each arrival where:
            - t (float): arrival time
            - i (int): dimension index 
            - state (int): state value at time t
            Format: [(t, i, state), (t, i, state), ...]
        """

        num_arrivals = len(arrival_times_array)
        if len(arrival_dims_array) != num_arrivals or len(arrival_states_array) != num_arrivals:
            raise ValueError("All input arrays must have the same length")

        paths = [[] for _ in range(self.dim)]
        full_information = []
        for k in range(num_arrivals):
            paths[arrival_dims_array[k]].append(arrival_times_array[k])
            full_information.append((arrival_times_array[k], arrival_dims_array[k], arrival_states_array[k]))

        return paths, full_information

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Subclass: Exponential Excitation Kernel with multiplicative state dependence


class ExpSDHawkes(SDHawkes):
    """
    Subclass of SDHawkes specialized for exponential excitation kernels with 
    multiplicative state dependence. Contains an optimized simulation algorithm 
    utilizing the special excitation structure.
    
    This class uses the time decay of exponentials to efficiently compute excitation terms
    for exponential kernels of the form:
        phi(t, i, j, y) = r(i,j,y) * alpha[i,j] * exp(-beta[i,j] * t)
    
    Key assumptions:
    - Excitation kernels are exponential with decay rates beta[i,j]
    - State dependence enters multiplicatively through function r(i,j,y)
    - Excitation terms decay monotonically between arrivals
    
    NOTE: For this subclass, you do NOT need to specify an excitation_kernel_func.
    The excitation structure is fully determined by the alpha, beta, and r parameters.

    """
    def __init__(self, background_intensity_func, background_intensity_max, state_matrix, alpha, beta, r,
                 max_arrivals: int = 500000, num_workers: int = 1, use_disk: bool = True):
        """
        Initialize the exponential kernel Hawkes process simulator.
        
        Parameters:
        -----------
        background_intensity_func : Callable
            See parent class SDHawkes for details
        
        background_intensity_max : float
            See parent class SDHawkes for details
        
        state_matrix : np.ndarray
            See parent class SDHawkes for details
        
        alpha : np.ndarray
            Excitation parameter matrix, shape (dim, dim).
            alpha[i,j] represents the magnitude of excitation from dimension j to dimension i
        
        beta : np.ndarray
            Decay parameter matrix, shape (dim, dim).
            beta[i,j] is the exponential decay rate for excitation from j to i
        
        r : Callable
            State-dependent amplification function.
            Signature: r(i, j, y) -> float
            Arguments:
                - i (int): Dimension index (from 0 to dim-1)
                - j (int): Dimension index (from 0 to dim-1)
                - y (float or np.ndarray): Current state value
            Returns:
                - float: Amplification factor for excitation kernel element at (i,j) for given state y
            Example: For state-dependent damping with scalar parameter delta and scalar-valued states:
                def r(i, j, state):
                    return (1+delta)**(-state)
        
        max_arrivals : int, optional
            See parent class SDHawkes for details (default: 500000)
        
        num_workers : int, optional
            See parent class SDHawkes for details (default: 1)
        
        use_disk : bool, optional
            See parent class SDHawkes for details (default: True)
        
        """
        super().__init__(background_intensity_func, None, background_intensity_max, state_matrix,
                         max_arrivals, num_workers, use_disk)
        self.alpha = alpha
        self.beta = beta
        self.r = r
        self.excitation_kernel_func = self._build_exponential_excitation_kernel_func(alpha,beta,r,self.dim)

    @staticmethod
    def _build_exponential_excitation_kernel_func(alpha:np.ndarray, beta:np.ndarray, r:Callable, dim:int) -> Callable:
        """
        Build an exponential excitation kernel function for Hawkes processes with exponential kernels and multiplicatively-factored state dependence, i.e., 
            φ_{ij}(t,y) = r_{ij}(y) α_{ij} e^{-β_{ij} t}
        NOTE: The constructed function is not used within the simulation code, but is constructed to retain compatibility with the general class structure.

        Parameters:
        -----------
        alpha : np.ndarray
            Excitation parameter matrix, shape (dim, dim).
            alpha[i,j] represents the magnitude of excitation from dimension j to dimension i
        
        beta : np.ndarray
            Decay parameter matrix, shape (dim, dim).
            beta[i,j] is the exponential decay rate for excitation from j to i
        
        r : Callable
            State-dependent amplification function.
            Signature: r(i, j, y) -> float
            Arguments:
                - i (int): Dimension index (from 0 to dim-1)
                - j (int): Dimension index (from 0 to dim-1)
                - y (float or np.ndarray): Current state value
            Returns:
                - float: Amplification factor for excitation kernel element at (i,j) for given state y
            Example: For state-dependent damping with scalar parameter delta and scalar-valued states:
                def r(i, j, state):
                    return (1+delta)**(-state) 
        """
        def excitation_kernel_func(time_diffs,past_dims,past_states) -> np.ndarray:
            """
            NOTE: This function is not used within the simulation code, but is constructed to retain compatibility with the existing code structure.
            As such, we implement a very sub-optimal version of this function.

            excitation_kernel_func : Callable
            Function that computes the full excitation matrix from the history of arrivals.
            Signature: excitation_kernel_func(time_diffs, past_dims, past_states) -> np.ndarray
            
            Arguments:
                - time_diffs (np.ndarray): Time elapsed since each past arrival, shape (n_past,)
                  Index k gives time since the k-th arrival (k=0 is first arrival)
                - past_dims (np.ndarray): Dimension of each past arrival, shape (n_past,)
                  Index k gives dimension (0 to dim-1) of the k-th arrival
                - past_states (np.ndarray): State just before each past arrival
                  If state_dim=1: shape (n_past,), index k gives scalar state before k-th arrival
                  If state_dim>1: shape (n_past, state_dim), index k gives state vector before k-th arrival
            
            Returns:
                - np.ndarray: Excitation matrix, shape (dim, dim)
                  Entry [i,j] is the total excitation to dimension i from all past arrivals in dimension j
            """
            out = np.zeros(dim)
            for j in range(dim):
                mask = (past_dims == j)
                if not np.any(mask):
                    continue

                td_j = time_diffs[mask]         # shape (n_j,)
                Y_j = past_states[mask]      # shape (n_j, m)

                # decay for this source-dimension j against all target i
                decay = np.exp(-beta[:, j, None] * td_j[None, :]) # shape: (d, n_j)

                # evaluate r_{ij}(y_k) for fixed j, all i, all k
                R = np.empty((dim, len(td_j)), dtype=float) # shape: (d, n_j)
                for i in range(dim):
                    R[i] = np.array([r(i,j,y) for y in Y_j])

                # contribution from this j to each out_i
                # alpha[:, j] has shape (d,)
                # result shape: (d,)
                out += alpha[:, j] * np.sum(R * decay, axis=1)

            return out

        return excitation_kernel_func

    # wrapper for module-level Exponential SDHawkes simulation function
    def _make_sim_partial(self):
        """
        Create a picklable partial of the exponential simulation function with all fixed (instance-level) parameters bound.
        
        The returned callable has signature:
            sim_partial(T, FLLN_scaling, output_dir, output_name, seed) -> result
        
        Overrides SDHawkes._make_sim_partial to use sim_ExpSDHawkes_once with exponential-specific parameters (alpha, beta, r) instead of the general excitation_kernel_func.
        
        IMPORTANT: For this to be picklable by multiprocessing, all user-provided callables (background_intensity_func, r) must be defined at module level in the user's script (not lambdas or nested functions).
        
        Returns
        -------
        functools.partial
            Partial of sim_ExpSDHawkes_once with fixed instance parameters bound.
            Remaining free parameters: T (float), FLLN_scaling (float), output_dir (str or None), output_name (str or None), seed (SeedSequence or None)
        """
        return partial(sim_ExpSDHawkes_once,
            self.dim, self.state_dim, self.state_matrix,
            self.background_intensity_func, self.background_intensity_max,
            self.alpha, self.beta, self.r, self.max_arrivals, self.use_disk)


class MultiExpSDHawkes(SDHawkes):
    """
    dim: int,
    state_dim: int,
    third_dim: int,
    state_matrix: np.ndarray,
    background_intensity_func: Callable[[float, np.ndarray], np.ndarray],
    background_intensity_max: float,
    alpha: np.ndarray,
    beta: np.ndarray,
    r: Callable[[int, int, np.ndarray], float],
    max_arrivals: int,
    use_disk: bool,
    T: float,
    FLLN_scaling: float = 1,
    output_dir: Optional[str] = None,
    output_name: Optional[str] = None,
    seed: Optional[Union[int, np.random.SeedSequence]] = None

    """

    def __init__(self, background_intensity_func, background_intensity_max, state_matrix, alpha, beta, r,
                 max_arrivals: int = 500000, num_workers: int = 1, use_disk: bool = True):
        """
        Initialize the multi-exponential kernel Hawkes process simulator.
        
        Parameters:
        -----------
        background_intensity_func : Callable
            See parent class SDHawkes for details
        
        background_intensity_max : float
            See parent class SDHawkes for details
        
        state_matrix : np.ndarray
            See parent class SDHawkes for details
        
        alpha : np.ndarray
            Excitation parameter matrix, shape (dim, dim, third_dim).
            alpha[i,j,k] represents the magnitude of excitation from dimension j to dimension i at sum index k. The parameter 'third_dim' is implicitly passed by the shape of alpha (and beta).
            NOTE: alpha and beta must have the same shape. 
        
        beta : np.ndarray
            Decay parameter matrix, shape (dim, dim, third_dim).
            beta[i,j,k] is the exponential decay rate for excitation from j to i at sum index k.
            NOTE: alpha and beta must have the same shape.
        
        r : Callable
            State-dependent amplification function.
            Signature: r(i, j, k, y) -> float
            Arguments:
                - i (int): Dimension index (from 0 to dim-1)
                - j (int): Dimension index (from 0 to dim-1)
                - k (int): Sum index (from 0 to third_dim-1), where third_dim = np.shape(alpha)[2] = np.shape(beta)[2]
                - y (float or np.ndarray): Current state value
            Returns:
                - float: Amplification factor for excitation kernel element at (i,j,k) for given state y
            Example: For state-dependent damping with scalar parameter delta and scalar-valued states:
                def r(i, j, k, state):
                    return (1+delta)**(-state)
        
        max_arrivals : int, optional
            See parent class SDHawkes for details (default: 500000)
        
        num_workers : int, optional
            See parent class SDHawkes for details (default: 1)
        
        use_disk : bool, optional
            See parent class SDHawkes for details (default: True)
        
        """
        super().__init__(background_intensity_func, None, background_intensity_max, state_matrix,
                         max_arrivals, num_workers, use_disk)
                
        if np.shape(alpha) != np.shape(beta):
            raise ValueError("alpha and beta must have the same shape for the MultiExponential Hawkes model. Note that you can always set some alphas to 0 and set some betas to 0 if the sum lengths are inhomogeneous.")
        
        self.third_dim = np.shape(alpha)[2]
        
        self.alpha = alpha
        self.beta = beta
        self.r = r
        # TODO: build backup excitation kernel function for compatibility 
        #self.excitation_kernel_func = self._build_exponential_excitation_kernel_func(alpha,beta,r,self.dim)


    def _make_sim_partial(self):
        """
        Create a picklable partial of the exponential simulation function with all fixed (instance-level) parameters bound.
        
        The returned callable has signature:
            sim_partial(T, FLLN_scaling, output_dir, output_name, seed) -> result
        
        Overrides SDHawkes._make_sim_partial to use sim_ExpSDHawkes_once with exponential-specific parameters (alpha, beta, r) instead of the general excitation_kernel_func.
        
        IMPORTANT: For this to be picklable by multiprocessing, all user-provided callables (background_intensity_func, r) must be defined at module level in the user's script (not lambdas or nested functions).
        
        Returns
        -------
        functools.partial
            Partial of sim_MultiExpSDHawkes_once with fixed instance parameters bound.
            Remaining free parameters: T (float), FLLN_scaling (float), output_dir (str or None), output_name (str or None), seed (SeedSequence or None)
        """
        return partial(sim_MultiExpSDHawkes_once,
            self.dim, self.state_dim, self.state_matrix,
            self.background_intensity_func, self.background_intensity_max,
            self.alpha, self.beta, self.r, self.max_arrivals, self.use_disk)




class ExpSAHawkes(SDHawkes):
    """
    State-Agnostic Hawkes process with exponential excitation kernels.
    
    Implements a standard (non-state-dependent) multivariate Hawkes process where intensity only depends on past arrivals, not on any evolving state. Inherits from SDHawkes and uses exponential kernels for efficient O(dim^2) time decay updates.
    
    The intensity for dimension i at time t is:
        λ_i(t) = μ_i + Σ_j ∫_0^t α_{ij} * exp(-β_{ij} * (t - s)) dN_j(s)

    Parameters
    ----------
    mu : np.ndarray
        Background intensity vector of shape (dim,).
        Constant baseline intensity for each dimension.
    alpha : np.ndarray
        Excitation matrix of shape (dim, dim).
        Entry (i,j) is the jump in λ_i when dimension j has an arrival.
    beta : np.ndarray
        Decay rate matrix of shape (dim, dim).
        Entry (i,j) is the exponential decay rate for excitation from j to i.
    max_arrivals : int, optional
        Maximum number of arrivals before terminating (safety cutoff). Default is 1,000,000.
    use_disk : bool, optional
        If True, simulations write output to disk. If False, keep all arrivals in memory. Default is True.
    num_workers : int, optional
        Number of parallel worker processes for multi-path simulations. Default is 1.
    
    Attributes
    ----------
    dim : int
        Number of dimensions in the process.
    mu : np.ndarray
        Background intensity vector.
    alpha : np.ndarray
        Excitation matrix.
    beta : np.ndarray
        Decay rate matrix.
    
    Raises
    ------
    ValueError
        If stability condition ρ(α_{ij}/β_{ij}) < 1 (where ρ is spectral radius) is not satisfied.
    
    Examples
    --------
    >>> mu = np.array([100.0, 80.0])
    >>> alpha = np.array([[0.5, 0.1], [0.1, 0.5]])
    >>> beta = np.ones((2, 2))
    >>> hawkes = ExpSAHawkes(mu, alpha, beta)
    >>> results = hawkes.sim(T=10.0, num_paths=100, base_seed=2026)
    
    Notes
    -----
    - Stability condition ρ(α_{ij}/β_{ij}) < 1 is checked in __init__.
    - Uses Ogata's modified thinning algorithm via sim_ExpSAHawkes_once.
    - Efficient O(dim^2) updates per arrival using matrix exponential decay.
    - Supports parallel multi-path simulation via inherited sim() method.
    - For parallel execution, uses multiprocessing with proper seed spawning.
    """
    def __init__(
        self,
        mu: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        max_arrivals: int = 1000000,
        use_disk: bool = True,
        num_workers: int = 1
    ):

        # Check stability condition
        H = alpha / beta
        rho = np.max(np.linalg.eigvals(H))
        if rho >= 1:
            raise ValueError("Spectral radius of H := (α_ij/β_ij)_ij must be strictly less than 1")
        
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.dim = len(mu)
        self.max_arrivals = max_arrivals
        self.use_disk = use_disk
        self.num_workers = num_workers


    def _make_sim_partial(self):
        """ 
        Create a picklable partial of the module-level simulation function with all fixed (instance-level) parameters bound.
        
        The returned callable has signature:
            sim_partial(T, FLLN_scaling, output_dir, output_name, seed) -> result
        
        This is used by sim() to create a function that can be safely sent to worker processes without pickling `self` or any bound methods.
        
        Returns
        -------
        functools.partial
            Partial of sim_ExpSAHawkes_once with fixed instance parameters bound.
            Remaining free parameters: T (float), FLLN_scaling (float), output_dir (str or None), output_name (str or None), seed (SeedSequence or None)
        """
        return partial(
            sim_ExpSAHawkes_once,
            self.mu, self.alpha, self.beta,
            self.dim, self.max_arrivals, self.use_disk)
    
