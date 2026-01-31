"""Example: Test sensitivity of SPARCED model to state and parameter variations.

This script demonstrates how state and parameter variations affect model outputs:
1. Apply Gaussian noise to state values (non-growth factors) and measure outputs
2. Apply Gaussian noise to a subset of parameter values and measure outputs

Note: Different parameters affect different pathways. For example:
- k1_1 (Rb degradation) affects Ribosome levels
- Parameters in EGFR/MAPK pathway affect ppERK
- Parameters in PI3K/AKT pathway affect ppAKT

Outputs:
- Time course plots showing n simulations
- Histograms of final timepoint values for key species
"""

import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.run_model import create_simulator


# Key species to track (with their indices)
KEY_SPECIES = {
    'ppERK': 717,    # free ppERK (index 717 in state_ids)
    'ppAKT': 696,    # free ppAKT (index 696)
    'ERK': 715,      # free ERK
    'AKT': 694,      # free AKT
    'Ribosome': 0,   # Ribosome (affected by k1_1 - Rb degradation)
}


def get_non_ligand_species(simulator):
    """Get list of species that are not growth factors (ligands)."""
    ligand_names = {'E', 'H', 'HGF', 'P', 'F', 'I', 'INS'}
    species_df = pd.read_csv(simulator.species_file, sep="\t", header=0, index_col=0, encoding='latin-1')
    all_species = list(species_df.index)
    return [s for s in all_species if s not in ligand_names]


def get_default_state_values(simulator, species_names):
    """Get default initial values for specified species from Species.txt."""
    species_df = pd.read_csv(simulator.species_file, sep="\t", header=0, index_col=0, encoding='latin-1')
    default_values = {}
    for species in species_names:
        if species in species_df.index:
            default_values[species] = species_df.loc[species, 'IC_Xinitialized']
    return default_values


def get_default_parameter_values(simulator):
    """Get default values for all AMICI fixed parameters."""
    param_names = simulator.model.getFixedParameterNames()
    default_values = {}
    for param in param_names:
        default_values[param] = simulator.model.getFixedParameterByName(param)
    return default_values


def test_ribosome_parameter_override(simulator):
    """Simple test to verify parameter override affects Ribosome levels.

    k1_1 is the Rb degradation rate parameter. Changing it affects Ribosome
    concentration because Rb regulates E2F which controls ribosome biogenesis.

    Expected: Increasing k1_1 (faster Rb degradation) → lower Ribosome levels
    """
    print(f"\n{'='*60}")
    print("RIBOSOME PARAMETER OVERRIDE TEST")
    print(f"{'='*60}")

    # Get original k1_1 value
    test_param = 'k1_1'
    original_value = simulator.model.getFixedParameterByName(test_param)
    print(f"Parameter: {test_param}")
    print(f"Original value: {original_value}")

    # Run baseline simulation
    print("\nRunning baseline simulation...")
    baseline_df = run_single_simulation(simulator, stop=60.0)
    baseline_ribosome = baseline_df['Ribosome'].values
    print(f"  Baseline Ribosome (final): {baseline_ribosome[-1]:.2f} nM")

    # Change k1_1 to a much larger value
    new_value = 0.1  # ~24x increase from 0.0042
    simulator.model.setFixedParameterByName(test_param, new_value)
    print(f"\nSet {test_param} to: {new_value}")

    # Run simulation with modified parameter
    print("Running simulation with modified parameter...")
    modified_df = run_single_simulation(simulator, stop=60.0)
    modified_ribosome = modified_df['Ribosome'].values
    print(f"  Modified Ribosome (final): {modified_ribosome[-1]:.2f} nM")

    # Calculate difference
    diff = baseline_ribosome[-1] - modified_ribosome[-1]
    pct_change = (diff / baseline_ribosome[-1]) * 100
    print(f"\nDifference: {diff:.2f} nM ({pct_change:.1f}%)")

    # Restore original value
    simulator.model.setFixedParameterByName(test_param, original_value)
    print(f"Restored {test_param} to: {original_value}")

    if abs(diff) > 100:  # Expect at least 100 nM difference
        print(f"\n✓ SUCCESS: Parameter override produces visible effect!")
    else:
        print(f"\n⚠ WARNING: Small effect detected")

    return baseline_ribosome, modified_ribosome


def add_gaussian_noise(values, noise_fraction=0.1, rng=None):
    """Add Gaussian noise to values.

    Args:
        values: Dict of name -> value
        noise_fraction: Standard deviation as fraction of mean
        rng: numpy random generator

    Returns:
        Dict with noisy values (clipped to be non-negative)
    """
    if rng is None:
        rng = np.random.default_rng()

    noisy_values = {}
    for name, value in values.items():
        std = abs(value) * noise_fraction
        noise = rng.normal(0, std)
        noisy_value = max(0, value + noise)  # Ensure non-negative
        noisy_values[name] = noisy_value

    return noisy_values


def run_single_simulation(simulator, state_values=None, parameter_values=None, stop=60.0):
    """Run a single simulation with optional overrides."""
    results_df = simulator.simulate(
        start=0.0,
        stop=stop,
        step=0.5,
        state_values=state_values,
        parameter_values=parameter_values
    )
    return results_df


def extract_key_species(results_df, species_names):
    """Extract time courses for key species."""
    time_min = results_df["time"].values

    extracted = {'time': time_min}
    for name, idx in KEY_SPECIES.items():
        species_name = species_names[idx]
        if species_name in results_df.columns:
            extracted[name] = results_df[species_name].values
            extracted[f'{name}_name'] = species_name
        else:
            print(f"  Warning: {species_name} not found in results")
            extracted[name] = np.full_like(time_min, np.nan)

    return extracted


def run_state_sensitivity_analysis(simulator, n_simulations=10, noise_fraction=0.1,
                                   simulation_duration_min=60.0, seed=42):
    """Run sensitivity analysis on state values."""
    print(f"\n{'='*60}")
    print(f"STATE SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Simulations: {n_simulations}")
    print(f"Noise fraction: {noise_fraction}")
    print(f"Simulation duration: {simulation_duration_min} minutes")

    rng = np.random.default_rng(seed)
    non_ligand_species = get_non_ligand_species(simulator)
    default_states = get_default_state_values(simulator, non_ligand_species)
    print(f"Non-ligand species: {len(non_ligand_species)}")

    all_time_courses = []
    final_values = {name: [] for name in KEY_SPECIES.keys()}

    for i in range(n_simulations):
        noisy_states = add_gaussian_noise(default_states, noise_fraction, rng)
        results_df = run_single_simulation(
            simulator,
            state_values=noisy_states,
            stop=simulation_duration_min
        )
        key_data = extract_key_species(results_df, simulator.state_ids)
        all_time_courses.append(key_data)
        for name in KEY_SPECIES.keys():
            final_values[name].append(key_data[name][-1])

        if (i + 1) % max(1, n_simulations // 5) == 0:
            print(f"  Completed {i + 1}/{n_simulations} simulations...")

    return {
        'time_courses': all_time_courses,
        'final_values': final_values,
        'species_names': simulator.state_ids
    }


def run_parameter_sensitivity_analysis(simulator, n_simulations=10, noise_fraction=0.1,
                                        simulation_duration_min=60.0, n_params=50, seed=42):
    """Run sensitivity analysis on a subset of parameter values."""
    print(f"\n{'='*60}")
    print(f"PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Simulations: {n_simulations}")
    print(f"Noise fraction: {noise_fraction}")
    print(f"Parameters tested: {n_params} (subset of 2715 total)")
    print(f"Simulation duration: {simulation_duration_min} minutes")

    rng = np.random.default_rng(seed + 1000)
    all_params = get_default_parameter_values(simulator)

    # Use all parameters if n_params is None, otherwise use subset
    if n_params is None:
        test_params = all_params
    else:
        test_params = dict(list(all_params.items())[:n_params])
    print(f"Testing parameters: {len(test_params)} total")
    print(f"Sample: {list(test_params.keys())[:5]}...")

    all_time_courses = []
    final_values = {name: [] for name in KEY_SPECIES.keys()}

    for i in range(n_simulations):
        noisy_params = add_gaussian_noise(test_params, noise_fraction, rng)
        results_df = run_single_simulation(
            simulator,
            parameter_values=noisy_params,
            stop=simulation_duration_min
        )
        key_data = extract_key_species(results_df, simulator.state_ids)
        all_time_courses.append(key_data)
        for name in KEY_SPECIES.keys():
            final_values[name].append(key_data[name][-1])

        if (i + 1) % max(1, n_simulations // 5) == 0:
            print(f"  Completed {i + 1}/{n_simulations} simulations...")

    return {
        'time_courses': all_time_courses,
        'final_values': final_values,
        'species_names': simulator.state_ids
    }


def plot_time_courses(all_time_courses, title, output_path):
    """Plot time courses for multiple simulations."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=150)
    axes = axes.flat

    time_hr = all_time_courses[0]['time'] / 60.0  # Convert to hours

    for i, (species_name, species_label) in enumerate([('ppERK', 'ppERK (nM)'), ('ppAKT', 'ppAKT (nM)'),
                                                        ('ERK', 'ERK (nM)'), ('AKT', 'AKT (nM)'),
                                                        ('Ribosome', 'Ribosome (nM)'), ('ppERK_zoom', 'ppERK zoomed')]):
        ax = axes[i]

        if species_name == 'ppERK_zoom':
            # Zoomed view of ppERK first 20 minutes
            for tc in all_time_courses:
                mask = time_hr <= 20
                ax.plot(time_hr[mask], tc['ppERK'][mask], 'r-', alpha=0.5, linewidth=1)
            ax.set_xlim(0, 20)
            ax.set_title('ppERK (0-20 hr zoom)')
        else:
            for tc in all_time_courses:
                if species_name in tc:
                    alpha = 0.6 if len(all_time_courses) > 5 else 0.8
                    ax.plot(time_hr, tc[species_name], color=['red', 'blue', 'red', 'blue', 'green', 'red'][i],
                           alpha=alpha, linewidth=1)
            ax.set_title(species_label)

        ax.set_xlabel('Time (hr)')
        ax.set_ylabel('Concentration (nM)')
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close(fig)


def plot_final_value_histograms(final_values, title, output_path):
    """Plot histograms of final timepoint values."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=150)
    axes = axes.flat

    colors = {'ppERK': 'red', 'ppAKT': 'blue', 'ERK': 'darkred', 'AKT': 'darkblue', 'Ribosome': 'green'}

    for i, species in enumerate(['ppERK', 'ppAKT', 'ERK', 'AKT', 'Ribosome']):
        ax = axes[i]
        vals = np.array(final_values[species])

        # Remove NaN values
        vals = vals[~np.isnan(vals)]

        if len(vals) > 0:
            n_bins = min(20, len(vals))
            ax.hist(vals, bins=n_bins, color=colors.get(species, 'gray'), alpha=0.7, edgecolor='black')

            mean_val = np.mean(vals)
            median_val = np.median(vals)
            std_val = np.std(vals)

            ax.axvline(mean_val, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='gray', linestyle=':', linewidth=2, label=f'Median: {median_val:.2f}')

            ax.set_xlabel('Concentration (nM)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{species} (CV: {std_val/mean_val*100:.1f}%)')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Hide last subplot if not used
    axes[-1].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close(fig)


def print_statistics(final_values, analysis_name):
    """Print statistics for final values."""
    print(f"\n{analysis_name} - Final Timepoint Statistics:")
    print(f"  {'Species':<12} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12} {'CV':<10}")
    print(f"  {'-'*70}")

    for species in ['ppERK', 'ppAKT', 'ERK', 'AKT', 'Ribosome']:
        vals = np.array(final_values[species])
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            cv = std_val / mean_val * 100 if mean_val > 0 else 0
            print(f"  {species:<12} {mean_val:<12.4f} {std_val:<12.4f} {np.min(vals):<12.4f} {np.max(vals):<12.4f} {cv:<10.2f}%")


if __name__ == "__main__":
    # Configuration
    N_SIMULATIONS = 20
    NOISE_FRACTION = 0.5     # 50% Gaussian noise (larger to see effects)
    DURATION_MIN = 60.0      # 1 hour simulation
    N_PARAMS_TO_TEST = None  # None = test ALL 2715 parameters
    RANDOM_SEED = 42

    print("="*60)
    print("SPARCED SENSITIVITY ANALYSIS")
    print("="*60)
    print(f"Simulations per analysis: {N_SIMULATIONS}")
    print(f"Noise fraction: {NOISE_FRACTION * 100}% (larger noise for visible effects)")
    print(f"Simulation duration: {DURATION_MIN} minutes")
    print(f"Parameters tested: {'ALL' if N_PARAMS_TO_TEST is None else N_PARAMS_TO_TEST} of 2715 total")
    print()
    print("Note: Small parameters (e.g., k1_1=0.0042) require larger noise fractions")
    print("      to produce visible output variations. Using 50% noise for this test.")

    # Initialize simulator
    print("\nInitializing SPARCED simulator...")
    simulator = create_simulator()
    print(f"Model loaded: {len(simulator.state_ids)} species")

    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    # Run simple Ribosome test first
    test_ribosome_parameter_override(simulator)
    output_dir.mkdir(exist_ok=True)

    # Run state sensitivity analysis
    state_results = run_state_sensitivity_analysis(
        simulator,
        n_simulations=N_SIMULATIONS,
        noise_fraction=NOISE_FRACTION,
        simulation_duration_min=DURATION_MIN,
        seed=RANDOM_SEED
    )

    # Plot state analysis results
    state_tc_path = output_dir / "sensitivity_state_timecourses.png"
    state_hist_path = output_dir / "sensitivity_state_histograms.png"

    plot_time_courses(
        state_results['time_courses'],
        f"State Variability ({N_SIMULATIONS} simulations, {NOISE_FRACTION*100}% noise)",
        state_tc_path
    )
    plot_final_value_histograms(
        state_results['final_values'],
        f"Final Values - State Variability",
        state_hist_path
    )
    print_statistics(state_results['final_values'], "STATE SENSITIVITY")

    # Run parameter sensitivity analysis
    param_results = run_parameter_sensitivity_analysis(
        simulator,
        n_simulations=N_SIMULATIONS,
        noise_fraction=NOISE_FRACTION,
        simulation_duration_min=DURATION_MIN,
        n_params=N_PARAMS_TO_TEST,
        seed=RANDOM_SEED
    )

    # Plot parameter analysis results
    param_tc_path = output_dir / "sensitivity_parameter_timecourses.png"
    param_hist_path = output_dir / "sensitivity_parameter_histograms.png"

    plot_time_courses(
        param_results['time_courses'],
        f"Parameter Variability ({N_SIMULATIONS} simulations, {NOISE_FRACTION*100}% noise on {N_PARAMS_TO_TEST} params)",
        param_tc_path
    )
    plot_final_value_histograms(
        param_results['final_values'],
        f"Final Values - Parameter Variability",
        param_hist_path
    )
    print_statistics(param_results['final_values'], "PARAMETER SENSITIVITY")

    # Compare results
    print("\n" + "="*60)
    print("SENSITIVITY SUMMARY")
    print("="*60)

    for species in ['ppERK', 'ppAKT', 'Ribosome']:
        state_cv = np.std(state_results['final_values'][species]) / np.mean(state_results['final_values'][species]) * 100
        param_cv = np.std(param_results['final_values'][species]) / np.mean(param_results['final_values'][species]) * 100

        print(f"\n{species}:")
        print(f"  State variability CV:     {state_cv:.4f}%")
        print(f"  Parameter variability CV: {param_cv:.4f}%")

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\nOutput files: {output_dir}")
    print(f"  - {state_tc_path.name}")
    print(f"  - {state_hist_path.name}")
    print(f"  - {param_tc_path.name}")
    print(f"  - {param_hist_path.name}")
