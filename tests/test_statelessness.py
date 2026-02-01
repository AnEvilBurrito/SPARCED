"""Tests for SPARCED simulator statelessness.

These tests verify that the SPARCEDSimulator.simulate() function is stateless -
each call must not modify internal parameter/state of the model.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.run_model import create_simulator


@pytest.fixture
def simulator():
    """Fixture providing a simulator instance for tests.

    Creates a fresh simulator instance for each test function.
    """
    return create_simulator()


class TestSimulateStatelessness:
    """Tests ensuring simulate() is stateless and idempotent."""

    def test_simulate_idempotency(self, simulator):
        """Test that calling simulate() twice with same inputs produces identical results.

        This verifies that a simulation call doesn't leave any side effects
        that would affect a subsequent identical call.
        """
        # Use short simulation for speed (60 minutes)
        result1 = simulator.simulate(start=0, stop=60, step=0.5)
        result2 = simulator.simulate(start=0, stop=60, step=0.5)

        # DataFrame equality check (values + dtypes + index)
        pd.testing.assert_frame_equal(result1, result2)

    def test_parameter_override_isolation(self, simulator):
        """Test that parameter overrides don't leak between simulate() calls.

        This is a three-sequence test:
        1. Baseline: Run with defaults → capture result A
        2. Modified: Run with parameter override → capture result B (should differ from A)
        3. Revert: Run with defaults again → capture result C (should match A)

        This confirms that parameter overrides in one call don't "leak"
        into subsequent calls - the model resets to its original state.
        """
        # Baseline simulation with defaults
        result_baseline = simulator.simulate(start=0, stop=60, step=0.5)

        # Simulation with parameter override (k1_1 is a rate parameter)
        # Default value is ~0.0042, we set it to 0.01 to see different results
        result_modified = simulator.simulate(
            start=0, stop=60, step=0.5,
            parameter_values={'k1_1': 0.01}
        )

        # Results should differ - the modified parameter should affect output
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(result_baseline, result_modified)

        # Revert to defaults - should match baseline exactly
        result_revert = simulator.simulate(start=0, stop=60, step=0.5)
        pd.testing.assert_frame_equal(result_baseline, result_revert)

    def test_state_override_isolation(self, simulator):
        """Test that state overrides don't leak between simulate() calls.

        Similar to test_parameter_override_isolation but for state (species) overrides.
        """
        # Baseline simulation with defaults
        result_baseline = simulator.simulate(start=0, stop=60, step=0.5)

        # Simulation with state override (modify Ribosome initial concentration)
        # Default Ribosome is ~1900 nM, we set it to a different value
        result_modified = simulator.simulate(
            start=0, stop=60, step=0.5,
            state_values={'Ribosome': 1000.0}
        )

        # Results should differ - the modified state should affect output
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(result_baseline, result_modified)

        # Revert to defaults - should match baseline exactly
        result_revert = simulator.simulate(start=0, stop=60, step=0.5)
        pd.testing.assert_frame_equal(result_baseline, result_revert)

    def test_combined_override_isolation(self, simulator):
        """Test that combined state and parameter overrides don't leak between calls.

        Tests that when both state_values and parameter_values are provided,
        the model still resets properly after the call.
        """
        # Baseline
        result_baseline = simulator.simulate(start=0, stop=60, step=0.5)

        # Modified with both state and parameter overrides
        result_modified = simulator.simulate(
            start=0, stop=60, step=0.5,
            state_values={'Ribosome': 1000.0},
            parameter_values={'k1_1': 0.01}
        )

        # Should differ
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(result_baseline, result_modified)

        # Revert - should match baseline
        result_revert = simulator.simulate(start=0, stop=60, step=0.5)
        pd.testing.assert_frame_equal(result_baseline, result_revert)
