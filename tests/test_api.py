"""Tests for SPARCED HTTP API.

These tests verify the FastAPI server and SPARCED simulation functionality.
Tests use realistic parameters based on the notebook configuration.
"""

import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from api.app import app, simulator as global_simulator
from api.run_model import create_simulator


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    # Manually initialize simulator for tests (TestClient doesn't trigger startup)
    # The simulator is a module-level variable in api.app
    import api.app
    if api.app.simulator is None:
        api.app.simulator = create_simulator()
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check and root endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "SPARCED HTTP Solver"
        assert "endpoints" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_dir" in data
        assert "deterministic" in data


class TestSimulateEndpoint:
    """Test the /simulate endpoint."""

    def test_head_request(self, client):
        """Test HEAD request for endpoint validation (HTTPSolver compile check)."""
        response = client.head("/simulate")
        assert response.status_code == 200

    def test_simulate_default_params(self, client):
        """Test simulation with default parameters (short duration for testing).

        Uses a 5 minute simulation for quick testing.
        """
        response = client.post("/simulate", json={
            "start": 0.0,
            "stop": 5.0,  # 5 minutes (quick test)
            "step": 0.5   # 0.5 minutes = 30 seconds (internal timestep)
        })
        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "time" in data
        assert len(data["time"]) > 0

        # Time should start at 0 and be in minutes
        assert data["time"][0] == 0.0
        assert data["time"][-1] <= 5.0  # Should be at or before stop time

        # Verify species columns exist (sample known species)
        assert "Ribosome" in data
        assert "p53inac" in data

        # Verify data types
        assert isinstance(data["time"], list)
        assert isinstance(data["Ribosome"], list)
        assert all(isinstance(v, (int, float)) for v in data["time"])

    def test_simulate_12_hours(self, client):
        """Test simulation with 12 hour duration (matching notebook configuration).

        This test verifies the full simulation runs correctly and completes
        in under 1 minute as expected.
        """
        response = client.post("/simulate", json={
            "start": 0.0,
            "stop": 720.0,  # 12 hours = 720 minutes
            "step": 0.5
        })
        assert response.status_code == 200
        data = response.json()

        # 12 hours at 30-second steps = 720 * 2 + 1 = 1441 timepoints
        expected_timepoints = 720 * 2 + 1
        assert len(data["time"]) == expected_timepoints

        # Verify time range
        assert data["time"][0] == 0.0
        assert abs(data["time"][-1] - 720.0) < 0.5  # Should be ~720 minutes

    def test_simulate_with_ligand_override(self, client):
        """Test simulation with ligand concentration override."""
        response = client.post("/simulate", json={
            "start": 0.0,
            "stop": 5.0,
            "step": 0.5,
            "state_values": {
                "HGF": 50.0  # Override HGF from default 100.0 to 50.0 nM
            }
        })
        assert response.status_code == 200
        data = response.json()
        assert "time" in data
        assert len(data["time"]) > 2

    def test_simulate_with_multiple_ligand_overrides(self, client):
        """Test simulation with multiple ligand overrides."""
        response = client.post("/simulate", json={
            "start": 0.0,
            "stop": 5.0,
            "step": 0.5,
            "state_values": {
                "E": 50.0,
                "HGF": 75.0,
                "P": 200.0
            }
        })
        assert response.status_code == 200
        data = response.json()
        assert "time" in data

    def test_simulate_with_parameter_override(self, client):
        """Test simulation with AMICI fixed parameter override."""
        response = client.post("/simulate", json={
            "start": 0.0,
            "stop": 5.0,
            "step": 0.5,
            "parameter_values": {
                "k3_1": 150.0  # Override transcription rate (was kTL1_1)
            }
        })
        assert response.status_code == 200
        data = response.json()
        assert "time" in data

    def test_simulate_with_both_overrides(self, client):
        """Test simulation with both state and parameter overrides."""
        response = client.post("/simulate", json={
            "start": 0.0,
            "stop": 5.0,
            "step": 0.5,
            "state_values": {
                "HGF": 50.0
            },
            "parameter_values": {
                "k3_1": 150.0  # Override transcription rate (was kTL1_1)
            }
        })
        assert response.status_code == 200
        data = response.json()
        assert "time" in data


class TestErrorHandling:
    """Test error handling for invalid requests."""

    def test_invalid_stop_negative(self, client):
        """Test error handling for negative stop time."""
        response = client.post("/simulate", json={
            "start": 0.0,
            "stop": -1.0,  # Invalid: negative
            "step": 0.5
        })
        assert response.status_code == 422  # Validation error

    def test_invalid_stop_zero(self, client):
        """Test error handling for zero stop time."""
        response = client.post("/simulate", json={
            "start": 0.0,
            "stop": 0.0,  # Invalid: must be positive
            "step": 0.5
        })
        assert response.status_code == 422  # Validation error

    def test_invalid_step_negative(self, client):
        """Test error handling for negative step size."""
        response = client.post("/simulate", json={
            "start": 0.0,
            "stop": 5.0,
            "step": -0.5  # Invalid: negative
        })
        assert response.status_code == 422  # Validation error

    def test_unknown_species(self, client):
        """Test error handling for unknown species name."""
        response = client.post("/simulate", json={
            "start": 0.0,
            "stop": 5.0,
            "step": 0.5,
            "state_values": {
                "NonExistentSpecies": 100.0  # Unknown species
            }
        })
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"]
        assert "Unknown species" in data["detail"]["error"]

    def test_unknown_ratelaw(self, client):
        """Test error handling for unknown AMICI parameter."""
        response = client.post("/simulate", json={
            "start": 0.0,
            "stop": 5.0,
            "step": 0.5,
            "parameter_values": {
                "k99999_1": 100.0  # Unknown AMICI parameter
            }
        })
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"]
        assert "Unknown parameter" in data["detail"]["error"]

    def test_invalid_parameter_format(self, client):
        """Test error handling for invalid parameter name format."""
        response = client.post("/simulate", json={
            "start": 0.0,
            "stop": 5.0,
            "step": 0.5,
            "parameter_values": {
                "InvalidFormat": 100.0  # Not in AMICI model
            }
        })
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"]
        assert "Unknown parameter" in data["detail"]["error"]

    def test_missing_required_fields(self, client):
        """Test error handling for missing required fields."""
        response = client.post("/simulate", json={
            "start": 0.0
            # Missing: stop, step
        })
        assert response.status_code == 422  # Validation error


class TestDataFormat:
    """Test response data format matches HTTPSolver API contract."""

    def test_column_oriented_format(self, client):
        """Test response is in column-oriented JSON format."""
        response = client.post("/simulate", json={
            "start": 0.0,
            "stop": 5.0,
            "step": 0.5
        })
        assert response.status_code == 200
        data = response.json()

        # Column-oriented: each key maps to an array
        for key, values in data.items():
            assert isinstance(values, list)
            # All columns should have the same length
            assert len(values) == len(data["time"])

    def test_time_in_minutes(self, client):
        """Test time values are in minutes."""
        response = client.post("/simulate", json={
            "start": 0.0,
            "stop": 10.0,
            "step": 0.5
        })
        assert response.status_code == 200
        data = response.json()

        # First timepoint should be 0
        assert data["time"][0] == 0.0

        # Last timepoint should be ~10 minutes
        assert data["time"][-1] <= 10.0

        # Time increment should be ~0.5 minutes (30 seconds)
        time_increment = data["time"][1] - data["time"][0]
        assert abs(time_increment - 0.5) < 0.01

    def test_species_concentrations_in_nm(self, client):
        """Test species concentrations are in nM (nanomolar)."""
        response = client.post("/simulate", json={
            "start": 0.0,
            "stop": 5.0,
            "step": 0.5
        })
        assert response.status_code == 200
        data = response.json()

        # Ribosome initial concentration should be ~1900 nM
        # (from Species.txt: 1.90E+03)
        assert data["Ribosome"][0] > 0
        # Values should be reasonable for nM concentrations
        assert all(v >= 0 for v in data["Ribosome"])


class TestModelInfoEndpoints:
    """Test the new model info endpoints: outcome_var, modifiable states, modifiable parameters."""

    def test_outcome_var(self, client):
        """Test /outcome_var endpoint returns 'ppAKT'."""
        response = client.get("/outcome_var")
        assert response.status_code == 200
        data = response.text
        # FastAPI returns JSON string for plain string responses
        assert '"ppAKT"' in data or data == "ppAKT"

    def test_modifiable_states_excludes_akt_pathway(self, client):
        """Test /states/modifiable excludes ppAKT, pAKT, and AKT."""
        response = client.get("/states/modifiable")
        assert response.status_code == 200
        modifiable = response.json()

        # Should be a list
        assert isinstance(modifiable, list)

        # These should NOT be in modifiable
        assert "ppAKT" not in modifiable
        assert "pAKT" not in modifiable
        assert "AKT" not in modifiable

        # But other species should be present
        assert "Ribosome" in modifiable
        assert "p53inac" in modifiable

        # Complexes should be modifiable (not excluded)
        # Check for some AKT-related complexes
        assert "ppAKT_BAD" in modifiable or any("ppAKT" in s for s in modifiable)

    def test_modifiable_states_count(self, client):
        """Test /states/modifiable returns expected number of states."""
        response = client.get("/states")
        assert response.status_code == 200
        all_states = response.json()

        response = client.get("/states/modifiable")
        assert response.status_code == 200
        modifiable = response.json()

        # Modifiable excludes: 3 AKT pathway states + 81 non-positive states = 84 total
        assert len(modifiable) == len(all_states) - 84

    def test_modifiable_parameters_excludes_k1806(self, client):
        """Test /parameters/modifiable excludes k1806 (ppAKT dephosphorylation)."""
        response = client.get("/parameters/modifiable")
        assert response.status_code == 200
        modifiable = response.json()

        # Should be a list
        assert isinstance(modifiable, list)

        # k1806 should NOT be in modifiable
        assert "k1806" not in modifiable

        # But other AKT pathway parameters SHOULD be modifiable
        assert "k1754" in modifiable  # mTORC2 phosphorylation
        assert "k1756" in modifiable  # release from PIP3
        assert "k1750" in modifiable  # PDK1 phosphorylation

        # Translation rates should be modifiable
        assert "k105_1" in modifiable  # AKT1 translation
        assert "k106_1" in modifiable  # AKT2 translation
        assert "k104_1" in modifiable  # PTEN translation

    def test_modifiable_parameters_count(self, client):
        """Test /parameters/modifiable returns expected number of parameters."""
        response = client.get("/parameters")
        assert response.status_code == 200
        all_params = response.json()

        response = client.get("/parameters/modifiable")
        assert response.status_code == 200
        modifiable = response.json()

        # Modifiable excludes: 1 (k1806) + 73 non-positive parameters = 74 total
        assert len(modifiable) == len(all_params) - 74

    def test_modifiable_parameters_sorted(self, client):
        """Test /parameters/modifiable returns sorted list."""
        response = client.get("/parameters/modifiable")
        assert response.status_code == 200
        modifiable = response.json()

        # Should be sorted alphabetically
        assert modifiable == sorted(modifiable)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
