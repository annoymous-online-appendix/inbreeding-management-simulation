# Genetic Breeding Strategy Optimization

A simulation framework for analyzing optimal artificial insemination (AI) strategies in livestock breeding under inbreeding depression. The project compares three regulatory perspectives: robust (Monte Carlo) simulation, myopic (single-period) optimization, and forward-looking (dynamic programming) optimization.

## Project Structure

```
├── core.py                # Core library: yield models, simulation engines, policy solver, DP solver
├── plotting.py            # All visualization functions
├── run_robust.py          # Interface 1: Robust (Monte Carlo) simulation — single run
├── run_robust_band.py     # Interface 1b: Replicated robust simulation with confidence bands
├── run_myopic.py          # Interface 2: Myopic regulator simulation
├── run_dp.py              # Interface 3: Forward-looking (DP) regulator simulation
├── compare_policies.py    # Compare myopic vs. forward-looking policy functions
└── requirements.txt       # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Core Concepts

The model tracks a population of diploid organisms with two alleles at a single locus: **A** (high-yield) and **a** (wild-type). A regulator chooses `a_t` — the proportion of allele A used in controlled breeding each generation — to maximize total yield while managing costs.

**Key variables:**

| Symbol | Description |
|--------|-------------|
| `p`    | Frequency of allele A in the maternal pool |
| `a`    | Proportion of allele A in controlled breeding (policy variable) |
| `Δ`    | Yield advantage of allele A |
| `h`    | Dominance coefficient of heterozygotes |

### Two Yield Models

The project uses two distinct yield calculation models, each defined once in `core.py`:

**Robust simulation** (`calculate_yield_robust`) uses a disease-effect model with parameters `π` (disease incidence) and `γ` (disease severity). Homozygotes (AA and aa) are affected by disease while heterozygotes (Aa) are not. This model is evaluated via full Monte Carlo simulation with individual-level genotype tracking.

**Myopic and DP simulations** (`calculate_yield_analytical`) use a parameterized loss function `C1 × AA^n + C2 × aa^m`, where `C1`, `C2`, `n`, `m` are configurable. This model is evaluated analytically without Monte Carlo sampling.

## Usage

### Interface 1: Robust Simulation (`run_robust.py`)

Runs full Monte Carlo simulations with individual-level genotype tracking. Performs comparative statics analysis — sweeping one parameter while holding others fixed.

```bash
# Run a simulation sweeping parameter h
python run_robust.py simulate h

# Run a simulation sweeping parameter Delta
python run_robust.py simulate Delta

# Plot results from a saved file
python run_robust.py plot results_robust/results_h_20260306-120000.npy
```

Available sweep parameters: `h`, `Delta`, `pi`, `gamma`, `num_generations`, `initial_A_proportion`.

Model parameters are configured in the `BASE_PARAMS` dictionary inside the script.

### Interface 1b: Replicated Robust Simulation (`run_robust_band.py`)

Because Monte Carlo noise can make single-replication policy curves appear jittery, this script runs the comparative statics analysis multiple times (default: 100 replications) and presents results as mean curves with 95% confidence bands.

```bash
# Run with default 100 replications
python run_robust_band.py simulate h

# Run with custom number of replications
python run_robust_band.py simulate Delta --reps 50

# Plot from saved results
python run_robust_band.py plot results_robust_band/results_h_20260306-120000.npy
```

Produces three plots per analysis:
1. **Band plot** — Mean optimal policy with 95% confidence band
2. **Benefit band** — Mean potential benefit (%) with 95% confidence band
3. **Spaghetti plot** — All individual replications overlaid with the mean

### Interface 2: Myopic Simulation (`run_myopic.py`)

Analyzes the myopic (single-period) regulator who optimizes yield one generation at a time using analytical expected values. Includes three tasks controlled by boolean flags at the top of the script:

```bash
python run_myopic.py
```

| Flag | Task |
|------|------|
| `RUN_POLICY_FUNCTION_ANALYSIS` | Compute and plot the optimal policy function a*(p) |
| `RUN_DYNAMIC_SIMULATION` | Simulate p and a dynamics from multiple initial conditions |
| `RUN_YIELD_CURVE_EXPLANATION` | Plot yield vs. a curves for selected p values |

Parameters are configured in the `MODEL_PARAMS` dictionary inside the script.

### Interface 3: Forward-Looking Simulation (`run_dp.py`)

Solves the infinite-horizon dynamic programming problem via value function iteration. The forward-looking regulator accounts for how today's breeding decision affects future genetic composition.

```bash
python run_dp.py
```

Outputs include the value function, optimal policy function, and simulated trajectories from multiple initial conditions. Parameters are set when instantiating `GeneticOptimizer` at the top of the script.

### Policy Comparison (`compare_policies.py`)

Computes both policy functions under identical genetic parameters and plots them side by side.

```bash
python compare_policies.py
```

## Configuring Parameters

**Robust simulation** (`run_robust.py`, `run_robust_band.py`) — disease-effect parameters:

```python
BASE_PARAMS = {
    "population_size": 10000,
    "num_generations": 25,
    "initial_A_proportion": 0.5,
    "h": 0.8,
    "Delta": 0.3,
    "pi": 0.3,     # Disease incidence coefficient
    "gamma": 0.8,  # Disease severity
}
```

**Myopic and DP simulations** — loss function parameters:

```python
# In run_myopic.py or compare_policies.py:
"C1": 2.0,   # Multiplier on AA homozygote term
"C2": 2.0,   # Multiplier on aa homozygote term
"n": 2,      # Exponent on AA homozygote term
"m": 2,      # Exponent on aa homozygote term

# In run_dp.py (passed to GeneticOptimizer constructor):
GeneticOptimizer(h=0.6, Delta=0.4, beta=0.95, C1=2.0, C2=2.0, n=2, m=2)
```

## Output

Each simulation script saves results (`.npy` files and `.png` plots) to its own output directory:

| Script | Output Directory |
|--------|-----------------|
| `run_robust.py` | `results_robust/` |
| `run_robust_band.py` | `results_robust_band/` |
| `run_myopic.py` | `results_myopic/` |
| `run_dp.py` | `results_dp/` |
| `compare_policies.py` | `results_comparison/` |
