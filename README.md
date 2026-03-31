# State-Dependent Hawkes Simulation Class

The main class `sdhawkes.py` of this repository implements exact simulation of state-dependent Hawkes processes via Ogata's modified thinning algorithm. The `Validation/` sub-directory contains scripts that were used to validate the simulation algorithm against analytical moment formulae in the state-agnostic case (no analytical results exist for the state-dependent case), and validating against the Functional Law of Large Numbers for state-dependent Hawkes. The `Examples/` sub-directory contains a manual implementation of a 2-dimensional state-dependent Hawkes process, a numerical example displaying convergence to the FLLN limit, and a script for stability verification.

## Repository Layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── sdhawkes.py
├── SDHawkes_2d_config.py
├── ContractivityCondClass.py
├── Examples/
│   ├── ContractivityCondEx.py
│   ├── SDHawkes_2d_main.py
│   └── SDHawkes_2d_sim.py
└── Validation/
    ├── StateAgnostic_Hawkes_class.py
    ├── validation.py
    └── validation_main.py
```


## Mathematical Model

For a `d`-dimensional counting process $N(t) = (N_1(t), \ldots, N_d(t))$ we define a state process $Y(t) = M N(t)$, where `M` (`state_matrix` in the code) can be a row vector (scalar state) or a full matrix (vector state). The state-dependent Hawkes intensity for component `i` is

$$
\lambda_i(t) = \mu_i(t, Y(t-)) + \sum_{j=1}^d \int_{[0,t)} \phi_{ij}(t-s, Y(s-))\, dN_j(s),
$$

where $\mu$ is encoded via `background_intensity_func` and $\phi$ is encoded by `excitation_kernel_func`, both of which are specified by the user, so that the class can simulate a wide range of state-dependent Hawkes processes.

### Exponential/Markovian special case

The `Examples/SDHawkes_2d_sim.py` and the `Exp_SDHawkes` subclass implement the frequently used exponential kernel, but now adapted to the state-dependent setting with multiplicative state dependence:

$$
\phi_{ij}(t-s, Y(s-)) = r_i(Y(s-)) \alpha_{ij} e^{-\beta_{ij}(t-s)},
$$

where $r_i$ is the state-dependent amplification factor. Because the kernel is Markovian, the implementation maintains an excitation matrix `A` instead of recomputing the integral for the contribution from the excitation term from scratch: between arrivals `A` is decayed via `A *= exp(-β Δt)` and after an arrival in dimension `j` the column `A[:, j]` receives the rank-one update `α[:, j] * r`. This produces per-arrival $O(d^2)$ updates, which is significantly faster than iterating through the complete history (as one is forced to do in the general state-dependent case).

### Deterministic seeding & parallelism

- Batch simulations use `SeedSequence.spawn(num_paths)` to hand each worker a **child seed**. This ensures that (i) no two workers ever reuse the same random stream, (ii) rerunning with the same base seed reproduces every path exactly, and (iii) validation scripts can compare outputs from different simulators path-by-path, even when multiprocessing is enabled.
- The validation suite (`Validation/validation_main.py`) explicitly checks both single-path equality and parallel multi-path equality to guard against duplicate paths or diverging RNG updates.

### Recovering classical models from state-dependent models

| Scenario | How to obtain it |
| --- | --- |
| **General state-agnostic Hawkes** | Set `background_intensity_func(t, state)` to ignore `state`, and supply an `excitation_kernel_func` that depends only on time differences. Under these choices `λ_i(t)` reduces to the usual linear Hawkes intensity with deterministic background and history kernel. Note that this works for any state-agnostic background intensity and state-agnostic, non-temporally-increasing background kernel |
| **Exponential (Markovian) state-agnostic Hawkes** | Use the `Exp_SDHawkes` subclass (or provide an `excitation_kernel_func` that reproduces $\alpha_{ij} e^{-\beta_{ij}(t-s)}$) and let `r` be constant `1` to recover the classical state-agnostic exponential Hawkes. Setting `r` to a nontrivial function yields the state-dependent exponential model. |
| **State-Agnostic (inhomogeneous) Poisson** | Set `α = 0` (or make `excitation_kernel_func` return zeros) so that only `background_intensity_func` contributes (and make `background_intensity_func` not depend on `state`). This reproduces an inhomogeneous Poisson process. |

These reductions make it easy to benchmark the state-dependent simulator against the analytical 1D formulas and classical limits included in `Validation/`.


## Seeding, Parallelization, and Disk Usage Details

- **Disk-backed storage**: When `use_disk=True`, each simulation path is written to disk at the end of the simulation. This prevents the need to hold all paths in memory simultaneously when running many simulations, though each individual simulation still requires its full arrival history in memory for intensity updates.
- **Parallel execution**: `sdhawkes.py` and `Examples/SDHawkes_2d_sim.py` use `multiprocessing.Pool`. Workloads are prepared via helper functions (`data_for_parallel_sims`, `run_parallel_sims`), ensuring each worker receives a unique child seed and optional disk target.
- **Deterministic RNG control**: Every simulator accepts either integers or full `np.random.SeedSequence` objects. Validation scripts routinely spawn child seeds (`SeedSequence.spawn(num_paths)`) to standardize rng behavior across parallel simulations.


## Getting Started with the `SDHawkes` class

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Create your simulation script**:
   ```python
   from sdhawkes import SDHawkes, Exp_SDHawkes
   import numpy as np
   ```

3. **Define the model callbacks** (examples shown for a 2D process):
   IMPORTANT: define background and excitation functions (that are fed into an SDHawkes object) at module level so they remain picklable when the simulator parallelizes work. Not doing so will cause errors to be raised whenever the simulator is allowed to use multiple workers.
   IMPORTANT: It is a strict assumption of the simulation algorithm that the excitation kernel is non-increasing in time.
   ```python
   def background_intensity_func(t, state):
       # state can be scalar or vector depending on state_matrix
       return np.array([200.0, 120.0])

   def excitation_kernel_func(time_diffs, past_dims, past_states):
       # Return the dim x dim excitation matrix for arbitrary kernels
       ...  # use time_diffs/past_dims/past_states to build \phi(t-s, state)
   ```
   For exponential kernels, prefer `Exp_SDHawkes` and define a module-level `r(i, j, state)` callable plus `alpha`, `beta`, and `state_matrix` (see `Examples/SDHawkes_2d_sim.py`).

4. **Instantiate the simulator**:
   ```python
   sim = SDHawkes(
       background_intensity_func=background_intensity_func,
       excitation_kernel_func=excitation_kernel_func,
       background_intensity_max=400.0,
       state_matrix=np.array([1.0, -1.0]),
       max_arrivals=500_000,
       num_workers=4,
       use_disk=True,
   )
   ```
   - `state_matrix` maps counts to your chosen state.
   - `background_intensity_max` must upper-bound the background intensity for thinning correctness.

5. **Run simulations**:
   ```python
   result = sim.SDHawkes_sim(T=1.0, FLLN_scaling=1.0, output_file="/tmp/path.pkl")
   ```
   - If `use_disk=True`, `result` is a filename containing `(arrival_times, arrival_dims, arrival_states)`.
   - If `use_disk=False`, the method returns that tuple directly.
   - Provide either an integer seed (`np.random.seed`) or a `np.random.SeedSequence` to control reproducibility.

6. **Parallel batches & FLLN scaling**:
   The `FLLN_sim()` method for the SDHawkes class can be used with scaling parameter $n=1$ (no scaling) to generate many simulations in parallel. The `FLLN_sim()` method also can be used, as the name suggests, to simulate FLLN-scaled paths. See `Examples/SDHawkes_2d_sim.py` for an example of this.

7. **Post-processing**:
   - Convert disk outputs to arrays via `pickle.load`.
   - Use the utilities in `Examples/SDHawkes_2d_sim.py` (e.g., `compute_FLLN_ODE_difference`, plotting helpers) or craft your own analytics.


### Core Components

| File | Description |
| --- | --- |
| `sdhawkes.py` | Main simulator exposing `SDHawkes` (general kernels) and `Exp_SDHawkes` (exponential kernels) with disk-backed storage, multiprocessing, and FLLN-scaling aware callbacks. |
| `SDHawkes_2d_config.py` | Centralized configuration for the 2D experiments (intensity parameters, scaling, parallel worker counts, disk usage, etc.). |
| `ContractivityCondClass.py` | Reusable class for computing contractivity conditions given user-supplied H matrices and LP solvers. |
| `Examples/SDHawkes_2d_sim.py` | Specialized 2D exponential simulator plus helper utilities (`FLLN_sim`, `compute_FLLN_ODE_difference`, `solve_ode`, etc.) used in limit-theorem experiments. |
| `Examples/SDHawkes_2d_main.py` | Driver script that solves the deterministic limit ODE, runs large batches of FLLN-scaled simulations, compares trajectories, and plots deviations. |
| `Examples/ContractivityCondEx.py` | Computes contractivity conditions by building L¹ norm matrices `H(y)` and solving feasibility LPs (bisection and line-search) to verify stability for the chosen parameters. |
| `Validation/StateAgnostic_Hawkes_class.py` | Lightweight baseline Hawkes simulator (`SAHawkes`) used whenever analytical formulas exist (no state dependence). |
| `Validation/validation.py` | Validation utilities: analytical mean formulas, Monte Carlo aggregators, plotting helpers, multiple simulation wrappers, and exact path-comparison helpers. |
| `Validation/validation_main.py` | Orchestrates the validation suite (single-path equality tests, multi-path overlap checks, Monte Carlo vs. analytical comparisons, and statistical summaries). |


## Thesis Examples

`Examples/` contains numerical scripts from my masters thesis:

- `Examples/SDHawkes_2d_main.py` demonstrates convergence to the Functional Law of Large Numbers by solving the limit ODE, simulating many FLLN-scaled paths, and plotting deviations between simulations and the deterministic limit. 
- `Examples/ContractivityCondEx.py` checks contractivity/stability condition for multivariate state-dependent Hawkes using linear programs constructed from the excitation kernel.
- `Examples/SDHawkes_2d_config.py` allows one to go in and select different parameters for the FLLN and/or stability checking numerical experiments.

These scripts are included so results from the thesis can be reproduced, but day-to-day users only need them if they want to repeat those experiments verbatim, or are curious to see an example simulation/stability check for state-dependent Hawkes.


## Validation of Simulation Algorithms

`Validation/` exists to demonstrate correctness of the simulations in SDHawkes:

1. **Single-path equivalence**: `validation_main.py` drives `SAHawkes`, `SDHawkes`, `Exp_SDHawkes`, and the 2D simulator with the same `SeedSequence` child and verifies path-by-path equality.
2. **Parallel overlap checks**: batches of paths use spawned seeds so that parallel execution never duplicates randomness.
3. **Analytical benchmarking**: 1D exponential Hawkes simulations are compared against closed-form expectations for counts/intensities.
4. **Statistical robustness**: repeated Monte Carlo runs summarize maximum deviations across many experiments. These can be re-run to observe the convergence behavior of mean error and variance.

You do not need to run these scripts to use `SDHawkes`, but they are there to show the implementation has been thoroughly vetted.


## License

This project is licensed under the MIT License (see `LICENSE`).
