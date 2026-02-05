"""Sensitivity analysis for ppAKT using only modifiable states.

This script tests which modifiable states (excluding ppAKT, pAKT, AKT)
actually affect ppAKT output when perturbed. This helps identify which
species are good candidates for perturbation experiments in pan-cancer
treatment response prediction.
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


# Key AKT pathway species to test (all modifiable - excludes ppAKT, pAKT, AKT)
AKT_PATHWAY_SPECIES = {
    # Receptor ligands
    'E': 100.0,          # EGF
    'HGF': 100.0,        # HGF
    'P': 100.0,          # PDGF

    # Upstream regulators
    'PDK1': 11.5,        # PDK1 kinase
    'PTEN': 2.32,        # PTEN phosphatase (negative regulator)
    'mTORC2': None,      # mTORC2 complex
    'RICTOR': 0.167,     # RICTOR (mTORC2 component)
    'mTOR': 1.18,        # mTOR (mTORC2 component)

    # Membrane lipids
    'PIP3': None,        # PIP3 lipid
    'PIP2': None,        # PIP2 lipid

    # AKT substrates
    'GSK3b': None,
    'FOXO3': None,
    'TSC2': None,

    # Complexes (these ARE modifiable)
    'PIP3_ppAKT': 0.00396,  # ppAKT bound to PIP3
    'ppAKT_BAD': 3.35e-5,   # ppAKT bound to BAD
}

# Key parameters affecting ppAKT (all modifiable - excludes k1806)
AKT_PATHWAY_PARAMETERS = {
    'k1754': 10.0,      # mTORC2 phosphorylation of pAKT -> ppAKT
    'k1756': 1.0,       # Release of ppAKT from PIP3 complex
    'k1750': 10.0,      # PDK1 phosphorylation of AKT -> pAKT
    'k1757': 0.0001,    # ppAKT binding to PIP3
    'k1755': 0.001,     # PIP3_ppAKT -> PIP3_pAKT

    # Translation rates
    'k105_1': None,     # AKT1 translation
    'k106_1': None,     # AKT2 translation
    'k107_1': None,     # PDK1 translation
    'k104_1': None,     # PTEN translation (negative regulator)
}


def run_single_simulation(simulator, state_values=None, parameter_values=None,
                          stop=720.0):
    """Run a single simulation with optional overrides."""
    results_df = simulator.simulate(
        start=0.0,
        stop=stop,
        step=0.5,
        state_values=state_values,
        parameter_values=parameter_values
    )
    return results_df


def test_state_perturbation(simulator, species_name, perturbation_factor=2.0):
    """Test effect of perturbing a single species on ppAKT output."""
    # Get default state values
    defaults = simulator.get_state_defaults()

    if species_name not in defaults:
        return None

    original_value = defaults[species_name]
    if original_value == 0:
        return None

    # Run baseline
    baseline_df = run_single_simulation(simulator, stop=720.0)
    baseline_ppAKT = baseline_df['ppAKT'].values

    # Perturb by factor
    perturbed_value = original_value * perturbation_factor
    state_values = {species_name: perturbed_value}
    perturbed_df = run_single_simulation(simulator, state_values=state_values, stop=720.0)
    perturbed_ppAKT = perturbed_df['ppAKT'].values

    # Calculate metrics
    baseline_final = baseline_ppAKT[-1]
    perturbed_final = perturbed_ppAKT[-1]
    baseline_max = np.max(baseline_ppAKT)
    perturbed_max = np.max(perturbed_ppAKT)

    # Calculate effect sizes
    final_diff = perturbed_final - baseline_final
    final_pct = (final_diff / baseline_final * 100) if baseline_final > 0 else 0
    max_diff = perturbed_max - baseline_max
    max_pct = (max_diff / baseline_max * 100) if baseline_max > 0 else 0

    return {
        'species': species_name,
        'original': original_value,
        'perturbed': perturbed_value,
        'baseline_final': baseline_final,
        'perturbed_final': perturbed_final,
        'final_diff': final_diff,
        'final_pct': final_pct,
        'baseline_max': baseline_max,
        'perturbed_max': perturbed_max,
        'max_diff': max_diff,
        'max_pct': max_pct,
        'baseline_trajectory': baseline_ppAKT,
        'perturbed_trajectory': perturbed_ppAKT,
    }


def test_parameter_perturbation(simulator, param_name, perturbation_factor=2.0):
    """Test effect of perturbing a single parameter on ppAKT output."""
    # Get default parameter values
    defaults = simulator.get_parameter_defaults()

    if param_name not in defaults:
        return None

    original_value = defaults[param_name]
    if original_value == 0:
        return None

    # Run baseline
    baseline_df = run_single_simulation(simulator, stop=720.0)
    baseline_ppAKT = baseline_df['ppAKT'].values

    # Perturb by factor
    perturbed_value = original_value * perturbation_factor
    parameter_values = {param_name: perturbed_value}
    perturbed_df = run_single_simulation(simulator, parameter_values=parameter_values, stop=720.0)
    perturbed_ppAKT = perturbed_df['ppAKT'].values

    # Calculate metrics
    baseline_final = baseline_ppAKT[-1]
    perturbed_final = perturbed_ppAKT[-1]
    baseline_max = np.max(baseline_ppAKT)
    perturbed_max = np.max(perturbed_ppAKT)

    # Calculate effect sizes
    final_diff = perturbed_final - baseline_final
    final_pct = (final_diff / baseline_final * 100) if baseline_final > 0 else 0
    max_diff = perturbed_max - baseline_max
    max_pct = (max_diff / baseline_max * 100) if baseline_max > 0 else 0

    return {
        'parameter': param_name,
        'original': original_value,
        'perturbed': perturbed_value,
        'baseline_final': baseline_final,
        'perturbed_final': perturbed_final,
        'final_diff': final_diff,
        'final_pct': final_pct,
        'baseline_max': baseline_max,
        'perturbed_max': perturbed_max,
        'max_diff': max_diff,
        'max_pct': max_pct,
        'baseline_trajectory': baseline_ppAKT,
        'perturbed_trajectory': perturbed_ppAKT,
    }


def plot_sensitivity_results(state_results, param_results, output_dir):
    """Plot sensitivity analysis results."""
    # Filter successful results
    state_results = [r for r in state_results if r is not None]
    param_results = [r for r in param_results if r is not None]

    # Sort by effect size
    state_results_sorted = sorted(state_results, key=lambda x: abs(x['final_pct']), reverse=True)
    param_results_sorted = sorted(param_results, key=lambda x: abs(x['final_pct']), reverse=True)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # Plot 1: State perturbation effects on final ppAKT
    ax = axes[0, 0]
    if state_results_sorted:
        species = [r['species'] for r in state_results_sorted[:15]]
        effects = [r['final_pct'] for r in state_results_sorted[:15]]
        colors = ['green' if e > 0 else 'red' for e in effects]

        bars = ax.barh(species, effects, color=colors, alpha=0.7)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Change in final ppAKT (%)')
        ax.set_title('State Perturbations (2x)')
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, eff in zip(bars, effects):
            width = bar.get_width()
            ax.text(width + (1 if width > 0 else -1), bar.get_y() + bar.get_height()/2,
                    f'{eff:.1f}%', ha='left' if width > 0 else 'right', va='center', fontsize=8)

    # Plot 2: Parameter perturbation effects on final ppAKT
    ax = axes[0, 1]
    if param_results_sorted:
        params = [r['parameter'] for r in param_results_sorted[:15]]
        effects = [r['final_pct'] for r in param_results_sorted[:15]]
        colors = ['green' if e > 0 else 'red' for e in effects]

        bars = ax.barh(params, effects, color=colors, alpha=0.7)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Change in final ppAKT (%)')
        ax.set_title('Parameter Perturbations (2x)')
        ax.grid(True, alpha=0.3)

        for bar, eff in zip(bars, effects):
            width = bar.get_width()
            ax.text(width + (1 if width > 0 else -1), bar.get_y() + bar.get_height()/2,
                    f'{eff:.1f}%', ha='left' if width > 0 else 'right', va='center', fontsize=8)

    # Plot 3: Time course for top state perturbation
    ax = axes[1, 0]
    if state_results_sorted:
        top = state_results_sorted[0]
        time_h = np.linspace(0, 12, len(top['baseline_trajectory']))
        ax.plot(time_h, top['baseline_trajectory'], 'b-', linewidth=2, label='Baseline', alpha=0.7)
        ax.plot(time_h, top['perturbed_trajectory'], 'r-', linewidth=2,
                label=f"{top['species']} ({top['perturbed']/top['original']:.1f}x)", alpha=0.7)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('ppAKT (nM)')
        ax.set_title(f'Top State: {top["species"]} (Δ{top["final_pct"]:.1f}%)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 4: Time course for top parameter perturbation
    ax = axes[1, 1]
    if param_results_sorted:
        top = param_results_sorted[0]
        time_h = np.linspace(0, 12, len(top['baseline_trajectory']))
        ax.plot(time_h, top['baseline_trajectory'], 'b-', linewidth=2, label='Baseline', alpha=0.7)
        ax.plot(time_h, top['perturbed_trajectory'], 'r-', linewidth=2,
                label=f"{top['parameter']} ({top['perturbed']/top['original']:.1f}x)", alpha=0.7)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('ppAKT (nM)')
        ax.set_title(f'Top Parameter: {top["parameter"]} (Δ{top["final_pct"]:.1f}%)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('ppAKT Sensitivity Analysis (Modifiable States & Parameters Only)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'ppakt_sensitivity_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close(fig)


def print_summary_table(state_results, param_results):
    """Print summary table of sensitivity results."""
    state_results = [r for r in state_results if r is not None]
    param_results = [r for r in param_results if r is not None]

    state_results_sorted = sorted(state_results, key=lambda x: abs(x['final_pct']), reverse=True)
    param_results_sorted = sorted(param_results, key=lambda x: abs(x['final_pct']), reverse=True)

    print("\n" + "="*80)
    print("PPAKT SENSITIVITY ANALYSIS SUMMARY")
    print("="*80)

    print("\nTOP MODIFIABLE STATES AFFECTING ppAKT:")
    print(f"  {'Species':<15} {'Original':<12} {'Perturbed':<12} {'ΔFinal':<12} {'ΔMax':<12}")
    print("  " + "-"*65)
    for r in state_results_sorted[:15]:
        print(f"  {r['species']:<15} {r['original']:<12.4f} {r['perturbed']:<12.4f} "
              f"{r['final_pct']:>8.2f}% {r['max_pct']:>8.2f}%")

    print("\nTOP MODIFIABLE PARAMETERS AFFECTING ppAKT:")
    print(f"  {'Parameter':<15} {'Original':<12} {'Perturbed':<12} {'ΔFinal':<12} {'ΔMax':<12}")
    print("  " + "-"*65)
    for r in param_results_sorted[:15]:
        print(f"  {r['parameter']:<15} {r['original']:<12.4f} {r['perturbed']:<12.4f} "
              f"{r['final_pct']:>8.2f}% {r['max_pct']:>8.2f}%")

    # Check that excluded species are not tested
    print("\n" + "="*80)
    print("VALIDATION OF EXCLUSION RULES:")
    print("="*80)
    print("  The following species should NOT be in modifiable states:")
    print("    - ppAKT (outcome variable)")
    print("    - pAKT (direct precursor)")
    print("    - AKT (direct precursor)")
    print()
    modifiable_states = simulator.get_modifiable_states()
    print(f"  Verified: 'ppAKT' in modifiable: {('ppAKT' in modifiable_states)}")
    print(f"  Verified: 'pAKT' in modifiable: {('pAKT' in modifiable_states)}")
    print(f"  Verified: 'AKT' in modifiable: {('AKT' in modifiable_states)}")
    print()
    modifiable_params = simulator.get_modifiable_parameters()
    print(f"  Verified: 'k1806' in modifiable: {('k1806' in modifiable_params)}")

    print("\n" + "="*80)


if __name__ == "__main__":
    print("="*80)
    print("PPAKT SENSITIVITY ANALYSIS - Modifiable States & Parameters Only")
    print("="*80)
    print("\nThis test analyzes ppAKT sensitivity to perturbations in:")
    print("  1. Modifiable states (excludes ppAKT, pAKT, AKT)")
    print("  2. Modifiable parameters (excludes k1806)")
    print("\nPerturbation: 2x baseline value")
    print("Simulation: 12 hours (720 minutes)")

    # Initialize simulator
    print("\nInitializing SPARCED simulator...")
    simulator = create_simulator()
    print(f"Model loaded: {len(simulator.state_ids)} species")
    print(f"Modifiable states: {len(simulator.get_modifiable_states())}")
    print(f"Modifiable parameters: {len(simulator.get_modifiable_parameters())}")

    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    # Test state perturbations
    print("\n" + "="*80)
    print("TESTING STATE PERTURBATIONS")
    print("="*80)

    state_results = []
    for species_name, expected_value in AKT_PATHWAY_SPECIES.items():
        print(f"  Testing {species_name}...", end=" ")
        result = test_state_perturbation(simulator, species_name, perturbation_factor=2.0)
        if result:
            state_results.append(result)
            print(f"ΔFinal: {result['final_pct']:.2f}%, ΔMax: {result['max_pct']:.2f}%")
        else:
            print("SKIPPED (not found or zero value)")

    # Test parameter perturbations
    print("\n" + "="*80)
    print("TESTING PARAMETER PERTURBATIONS")
    print("="*80)

    param_results = []
    for param_name, expected_value in AKT_PATHWAY_PARAMETERS.items():
        print(f"  Testing {param_name}...", end=" ")
        result = test_parameter_perturbation(simulator, param_name, perturbation_factor=2.0)
        if result:
            param_results.append(result)
            print(f"ΔFinal: {result['final_pct']:.2f}%, ΔMax: {result['max_pct']:.2f}%")
        else:
            print("SKIPPED (not found or zero value)")

    # Plot and print results
    plot_sensitivity_results(state_results, param_results, output_dir)
    print_summary_table(state_results, param_results)

    print("\nCOMPLETE")
    print(f"\nOutput file: {output_dir / 'ppakt_sensitivity_analysis.png'}")
