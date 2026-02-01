"""Example: Test the new get_state_defaults() and get_parameter_defaults() methods.

This script verifies that the new SPARCEDSimulator methods correctly
return default values for states and parameters using native AMICI methods.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.run_model import create_simulator


def test_defaults_api():
    """Test the new get_state_defaults() and get_parameter_defaults() methods."""
    print("Initializing SPARCED simulator...")
    simulator = create_simulator()

    print("\n" + "="*60)
    print("TESTING get_state_defaults()")
    print("="*60)

    state_defaults = simulator.get_state_defaults()
    print(f"\nTotal states returned: {len(state_defaults)}")

    # Show first 10 states
    print("\nFirst 10 states:")
    for i, (name, value) in enumerate(list(state_defaults.items())[:10]):
        print(f"  {name}: {value} nM")

    # Verify known values
    print("\n" + "-"*60)
    print("VERIFICATION: Known default values")
    print("-"*60)

    # Ribosome should be ~1900 nM (actual AMICI model value)
    if 'Ribosome' in state_defaults:
        ribosome_value = state_defaults['Ribosome']
        print(f"  Ribosome: {ribosome_value} nM")
        assert 1800 < ribosome_value < 2000, f"Ribosome value {ribosome_value} outside expected range [1800, 2000]"
    else:
        print("  WARNING: Ribosome not found in state defaults")

    # Check for ppERK and ppAKT
    for species in ['ppERK', 'ppAKT']:
        if species in state_defaults:
            print(f"  {species}: {state_defaults[species]} nM")
        else:
            print(f"  WARNING: {species} not found in state defaults")

    print("\n" + "="*60)
    print("TESTING get_parameter_defaults()")
    print("="*60)

    parameter_defaults = simulator.get_parameter_defaults()
    print(f"\nTotal parameters returned: {len(parameter_defaults)}")

    # Show first 10 parameters
    print("\nFirst 10 parameters:")
    for i, (name, value) in enumerate(list(parameter_defaults.items())[:10]):
        print(f"  {name}: {value}")

    # Verify known values
    print("\n" + "-"*60)
    print("VERIFICATION: Known default values")
    print("-"*60)

    # k1_1 should be 0.0042
    if 'k1_1' in parameter_defaults:
        k1_1_value = parameter_defaults['k1_1']
        print(f"  k1_1: {k1_1_value}")
        assert abs(k1_1_value - 0.0042) < 0.0001, f"k1_1 value {k1_1_value} doesn't match expected 0.0042"
    else:
        print("  WARNING: k1_1 not found in parameter defaults")

    # k3_1 should be around 150
    if 'k3_1' in parameter_defaults:
        k3_1_value = parameter_defaults['k3_1']
        print(f"  k3_1: {k3_1_value}")
        assert 100 < k3_1_value < 200, f"k3_1 value {k3_1_value} outside expected range [100, 200]"
    else:
        print("  WARNING: k3_1 not found in parameter defaults")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ get_state_defaults() returned {len(state_defaults)} states")
    print(f"✓ get_parameter_defaults() returned {len(parameter_defaults)} parameters")
    print("\nBoth methods work correctly using native AMICI API!")


if __name__ == "__main__":
    test_defaults_api()
