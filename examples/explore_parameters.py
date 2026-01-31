"""Example: Explore available AMICI model parameters.

This script queries the AMICI model to identify:
- Available fixed parameters that can be overridden
- Default parameter values
- How to use setFixedParameterByName() for runtime overrides

The API now uses AMICI's native parameter names directly (e.g., k1_1, k3_1, etc.)
as found in ParamsAll.txt or via model.getFixedParameterNames().
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.run_model import create_simulator

def explore_parameters():
    """Query AMICI model for available parameters."""
    print("Initializing SPARCED simulator...")
    simulator = create_simulator()

    model = simulator.model

    print("\n" + "="*60)
    print("AMICI MODEL PARAMETERS")
    print("="*60)

    # Fixed parameters - these can be overridden via setFixedParameterByName()
    fixed_params = model.getFixedParameterNames()
    print(f"\nFixed parameters (overridable): {len(fixed_params)}")
    print(f"Sample: {list(fixed_params)[:10]}")

    # Show first 20 fixed parameters with their values
    print("\n" + "-"*60)
    print("FIXED PARAMETERS (first 20 with values)")
    print("-"*60)
    for i, param_name in enumerate(list(fixed_params)[:20]):
        value = model.getFixedParameterByName(param_name)
        print(f"  {param_name}: {value}")

    # Test setFixedParameterByName() to see if parameters can be changed
    print("\n" + "-"*60)
    print("TESTING setFixedParameterByName()")
    print("-"*60)

    # Pick a test parameter (k1_1 is the first ratelaw parameter for vbR reaction)
    test_param = 'k1_1'
    original_value = model.getFixedParameterByName(test_param)
    print(f"\nOriginal value of {test_param}: {original_value}")

    # Try to change it
    new_value = 999.0
    model.setFixedParameterByName(test_param, new_value)
    print(f"Set {test_param} to: {new_value}")

    # Verify the change
    changed_value = model.getFixedParameterByName(test_param)
    print(f"Verified value after setFixedParameterByName(): {changed_value}")

    # Restore original value
    model.setFixedParameterByName(test_param, original_value)
    print(f"Restored original value: {original_value}")

    if changed_value == new_value:
        print(f"\nSUCCESS: Parameter {test_param} can be changed at runtime!")
    else:
        print(f"\nFAILED: Parameter {test_param} value did not change.")

    # Show how parameters relate to reactions (from ParamsAll.txt)
    print("\n" + "-"*60)
    print("PARAMETER TO REACTION MAPPING (from ParamsAll.txt)")
    print("-"*60)

    paramsall_path = Path(__file__).parent.parent / "Demo" / "ParamsAll.txt"
    import pandas as pd
    params_df = pd.read_csv(paramsall_path, sep="\t", header=0, index_col=0)

    # Show first 10 parameters with their reaction names
    print(f"\nTotal parameters in ParamsAll.txt: {len(params_df)}")
    print("\nSample mapping (AMICI param_name -> reaction -> value):")
    for i, (param_name, row) in enumerate(params_df.head(10).iterrows()):
        print(f"  {param_name}: reaction={row['rxn']}, value={row['value']}, idx={row['idx']}")

    return {
        'fixed_params': fixed_params,
        'total_params': len(fixed_params)
    }

if __name__ == "__main__":
    result = explore_parameters()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Fixed parameters available for override: {result['total_params']}")
    print(f"\nTo override parameters, use AMICI parameter names:")
    print(f"  Example: {{'parameter_values': {{'k1_1': 0.005, 'k3_1': 150.0}}}}")
