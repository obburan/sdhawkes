import numpy as np
import matplotlib.pyplot as plt
import pickle
from typing import Optional, Union, Tuple, List

class SAHawkes:
    """
    State-Agnostic Hawkes process with exponential excitation kernels.
    
    Implements a standard (non-state-dependent) multivariate Hawkes process where intensity only depends on past arrivals, not on any evolving state. Uses exponential kernels to take advantage of efficient time decay updates.
    
    The intensity for dimension i at time t is:
        λ_i(t) = μ_i + Σ_j int_0^t α_{ij} * exp(-β_{ij} * (t - s)) dN_j(s)

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
    T : float
        Simulation time horizon [0, T].
    max_arrivals : int, optional
        Maximum number of arrivals before terminating (safety cutoff). Default is 1,000,000.
    use_disk : bool, optional
        If True, write arrivals to disk at the end of the simulation.
        If False, keep all arrivals in memory. Default is True.
    
    Attributes
    ----------
    dim : int
        Number of dimensions in the process.
    
    Raises
    ------
    ValueError
        If stability condition ρ(α_{ij}/β_{ij})_ij < 1 (where ρ is spectral radius) is not satisfied.
    
    Examples
    --------
    >>> mu = np.array([100.0, 80.0])
    >>> alpha = np.array([[0.5, 0.1], [0.1, 0.5]])
    >>> beta = np.ones((2, 2))
    >>> hawkes = SAHawkes(mu, alpha, beta, T=10.0)
    >>> paths, full_info = hawkes.simulate_path()
    
    Notes
    -----
    - Well-definedness condition: α_{ij}/β_{ij} < 1 for all i,j is checked in __init__.
    - Uses Ogata's modified thinning algorithm for simulation.
    - Efficient O(dim^2) updates per arrival using matrix exponential decay.
    """
    def __init__(
        self,
        mu: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        T: float,
        max_arrivals: int = 1000000,
        use_disk: bool = True
    ):

        # Check stability condition
        H = alpha / beta
        rho = np.max(np.linalg.eigvals(H))
        if rho >= 1:
            raise ValueError("Spectral radius of H := (α_ij/β_ij)_ij must be strictly less than 1")
        
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.dim = len(mu)
        self.max_arrivals = max_arrivals
        self.use_disk = use_disk
    
    def simulate_path(
        self,
        output_file: Optional[str] = None,
        seed: Optional[Union[int, np.random.SeedSequence]] = None
    ) -> Union[str, Tuple[List[List[float]], List[Tuple[float, int]]]]:
        """
        Simulate a complete sample path of the Hawkes process using Ogata's thinning algorithm.
        
        Generates arrivals on [0, T] using exponential inter-arrival times and acceptance/rejection.
        Efficiently updates excitation matrix using exp(-beta * dt) decay between events.
        
        Parameters
        ----------
        output_file : str, optional
            Path to save simulation output. If provided, arrivals are written to disk
            in batches. If None, arrivals are returned in memory. Default is None.
        seed : int, np.random.SeedSequence, or None, optional
            Seed for the random number generator. If None, uses OS entropy. Default is None.
        
        Returns
        -------
        str or tuple
            If output_file is provided (use_disk=True):
                Returns output_file path (str).
            If output_file is None (use_disk=False):
                Returns tuple (paths, full_information) where:
                - paths: list of dim lists, paths[i] contains arrival times for dimension i
                - full_information: list of tuples (arrival_time, dimension_index)
        
        Notes
        -----
        - Simulation terminates when t >= T or num_arrivals >= max_arrivals.
        - Disk mode writes all arrivals to disk at the end of the simulation.
        - Memory mode keeps all arrivals in memory during simulation.
        - Uses efficient matrix updates: excitation_matrix *= exp(-beta * time_diff).
        
        Examples
        --------
        >>> hawkes = SAHawkes(mu=np.array([100.0]), alpha=np.array([[0.5]]),
        ...                   beta=np.array([[1.0]]), T=10.0, use_disk=False)
        >>> paths, full_info = hawkes.simulate_path(seed=42)
        >>> print(f"Total arrivals: {len(full_info)}")
        >>> print(f"First 5 arrivals: {full_info[:5]}")
        """

        mu = self.mu
        alpha = self.alpha
        beta = self.beta
        dim = self.dim
        ones_vec = np.ones(dim)
        use_disk = self.use_disk

        # Initializations
        rng = np.random.default_rng(seed)
        full_information = []
        paths = [[] for _ in range(dim)]
        num_arrivals_so_far = 0
        current_time = 0.0
        current_intensities_vec = self.mu
        previous_intensities_vec = self.mu
        excitation_matrix = np.zeros((dim,dim))
        
        while current_time < self.T and num_arrivals_so_far < self.max_arrivals:

            previous_intensities_vec = current_intensities_vec
            # Generate next arrival time
            Max_intensity = np.sum(previous_intensities_vec)
            next_time = current_time + rng.exponential(scale=1.0/Max_intensity)
            
            # decay excitation terms
            time_diff = next_time - current_time
            excitation_matrix *= np.exp(-beta * time_diff)
            current_intensities_vec = mu + excitation_matrix @ ones_vec

            # Accept or reject the arrival
            U = rng.uniform(0, Max_intensity)
            if U <= np.sum(current_intensities_vec) and next_time <= self.T:
                
                # divide current intensity value into bins to figure out which dimension the new arrival belongs to
                cumsum_intensity_vec = np.cumsum(current_intensities_vec)
                d = np.searchsorted(cumsum_intensity_vec, U)

                num_arrivals_so_far += 1

                # divide current intensity value into bins to figure out which dimension the new arrival belongs to
                cumsum_intensity_vec = np.cumsum(current_intensities_vec)
                d = np.searchsorted(cumsum_intensity_vec, U)

                excitation_matrix[:, d] += alpha[:, d] # Add new jump in column d
                excitation_vec = excitation_matrix @ ones_vec # sum rows to get total intensity for each dimension
                # Calculate total intensity for each dimension
                current_intensities_vec = mu + excitation_vec

                paths[d].append(next_time)
                full_information.append((next_time, d))
            
            # Update current time (whether arrival was accepted or rejected)
            current_time = next_time

        if self.use_disk:
            with open(output_file, 'wb') as f:
                pickle.dump((paths, full_information), f)
            return output_file
        
        return paths, full_information
