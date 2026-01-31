"""Example: Validate API correctness against notebook simulation.

This script demonstrates that the SPARCEDSimulator produces identical results
to the notebook simulation in Demo/runModel.ipynb.

The notebook plots these species by index:
- xoutS_all[:,717] = free ppERK
- xoutS_all[:,696] = free ppAKT
- xoutS_all[:,715] = free ERK
- xoutS_all[:,694] = free AKT
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


def run_simulation():
    """Run 12-hour simulation matching notebook parameters."""
    print("Initializing SPARCED simulator...")
    simulator = create_simulator()

    print(f"State IDs (sample): {simulator.state_ids[:5]}")
    print(f"Total species: {len(simulator.state_ids)}")

    # Get species names at the indices used in notebook
    # Notebook uses: xoutS_all[:,717], xoutS_all[:,696], xoutS_all[:,715], xoutS_all[:,694]
    species_names = simulator.state_ids
    print(f"\nKey species from notebook indices:")
    print(f"  Index 717: {species_names[717]} (free ppERK)")
    print(f"  Index 696: {species_names[696]} (free ppAKT)")
    print(f"  Index 715: {species_names[715]} (free ERK)")
    print(f"  Index 694: {species_names[694]} (free AKT)")

    print("\nRunning 12-hour simulation (720 minutes)...")
    results_df = simulator.simulate(
        start=0.0,
        stop=720.0,   # 12 hours in minutes
        step=0.5      # 30-second internal timestep
    )

    return results_df, species_names


def plot_results(results_df, species_names):
    """Plot ppERK, ppAKT, ERK, AKT matching notebook."""
    # Convert time to hours for plotting (notebook: tt = tout_all/3600.0)
    time_hr = results_df["time"].values / 60.0

    # Get the key species by name
    ppERK_name = species_names[717]
    ppAKT_name = species_names[696]
    ERK_name = species_names[715]
    AKT_name = species_names[694]

    ppERK = results_df[ppERK_name].values
    ppAKT = results_df[ppAKT_name].values
    ERK = results_df[ERK_name].values
    AKT = results_df[AKT_name].values

    # Create figure matching notebook (figsize=(8, 2), dpi=300)
    fig = plt.figure(figsize=(8, 2), dpi=300, facecolor='w', edgecolor='k')

    ax1 = fig.add_subplot(121)
    plt.plot(time_hr, ppERK, 'r', linewidth=3, markersize=12, label='free ppERK')
    plt.plot(time_hr, ppAKT, 'b', linewidth=3, markersize=12, label='free ppAKT')
    plt.xlim([-1, 13])
    plt.legend()
    ax1.set_ylabel('Concentration (nM)')
    ax1.set_xlabel('Time (hr)')
    plt.xticks(np.arange(0, 13, step=4))

    ax1 = fig.add_subplot(122)
    plt.plot(time_hr, ERK, 'r--', linewidth=3, markersize=12, label='free ERK')
    plt.plot(time_hr, AKT, 'b--', linewidth=3, markersize=12, label='free AKT')
    plt.legend()
    ax1.set_xlabel('Time (hr)')
    plt.xlim([-1, 13])
    plt.xticks(np.arange(0, 13, step=4))

    plt.tight_layout()

    # Save plot
    output_path = Path(__file__).parent / "validation_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {output_path}")

    return output_path


def compare_with_reference(results_df, species_names):
    """Compare results with reference data from Demo/GrowthStim_S_0.txt."""
    ref_path = Path(__file__).parent.parent / "Demo" / "GrowthStim_S_0.txt"

    if not ref_path.exists():
        print(f"\nReference file not found: {ref_path}")
        print("Skipping numerical comparison.")
        return

    print(f"\nComparing with reference data: {ref_path}")
    reference_df = pd.read_csv(ref_path, header=0, index_col=0, sep="\t")

    # Reference time is in seconds (first column), convert to minutes
    ref_time_min = reference_df.index.values / 60.0
    api_time = results_df["time"].values

    print(f"  Reference timepoints: {len(ref_time_min)}")
    print(f"  API timepoints: {len(api_time)}")

    # Compare key species at notebook indices
    for idx, name in [(717, "ppERK"), (696, "ppAKT"), (715, "ERK"), (694, "AKT")]:
        species_name = species_names[idx]
        if species_name in reference_df.columns and species_name in results_df.columns:
            ref_values = reference_df[species_name].values
            api_values = results_df[species_name].values

            # Initial value comparison
            print(f"\n  {name} ({species_name}):")
            print(f"    Initial - Reference: {ref_values[0]:.4f}, API: {api_values[0]:.4f}")

            # Peak value comparison
            print(f"    Peak - Reference: {np.max(ref_values):.4f}, API: {np.max(api_values):.4f}")

            # Max absolute difference
            max_diff = np.max(np.abs(ref_values - api_values))
            print(f"    Max absolute difference: {max_diff:.6f} nM")

            # Check if results are essentially identical
            if max_diff < 1e-6:
                print(f"    Results match!")
            elif max_diff < 1e-3:
                print(f"    Results match within tolerance")
            else:
                print(f"    Warning: Results differ by {max_diff}")


if __name__ == "__main__":
    # Run simulation
    results_df, species_names = run_simulation()

    # Plot results
    plot_results(results_df, species_names)

    # Compare with reference
    compare_with_reference(results_df, species_names)

    print("\nValidation complete!")
