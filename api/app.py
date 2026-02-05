"""FastAPI HTTP server for SPARCED model simulation.

This module provides a FastAPI server that implements the HTTPSolver API
contract for running SPARCED simulations via HTTP POST requests.
"""

import asyncio
import os
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field, field_validator

from .run_model import create_simulator


# Pydantic models for request/response
class SimulationRequest(BaseModel):
    """Simulation request following HTTPSolver API contract."""
    start: float = Field(..., description="Start time in minutes (SPARCED always starts at 0)")
    stop: float = Field(..., description="Stop time in minutes (simulation duration)", gt=0)
    step: float = Field(..., description="Step size in minutes (SPARCED uses 0.5 min internal steps)", gt=0)
    state_values: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional species initial concentration overrides (nM)"
    )
    parameter_values: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional ratelaw parameter overrides"
    )

    @field_validator('stop')
    def stop_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('stop must be positive')
        return v

    @field_validator('step')
    def step_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('step must be positive')
        return v


class ErrorResponse(BaseModel):
    """Error response format."""
    error: str


# Create FastAPI app
app = FastAPI(
    title="SPARCED HTTP Solver",
    description="HTTP API for running SPARCED cell signaling simulations",
    version="0.1.0"
)

# Global simulator instance (initialized on startup)
simulator = None
# Global lock to serialize simulate() calls for thread safety
simulate_lock = asyncio.Lock()


@app.on_event("startup")
async def startup():
    """Initialize the SPARCED simulator on startup."""
    global simulator
    try:
        simulator = create_simulator()
        print(f"SPARCED simulator initialized successfully")
        print(f"  Model directory: {simulator.model_dir}")
        print(f"  Base directory: {simulator.base_dir}")
        print(f"  SBML file: {simulator.sbml_file}")
        print(f"  Deterministic: {simulator.deterministic}")
    except Exception as e:
        print(f"ERROR: Failed to initialize SPARCED simulator: {e}")
        raise


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "SPARCED HTTP Solver",
        "version": "0.1.0",
        "endpoints": {
            "POST /simulate": "Run a simulation",
            "GET /states": "Get default state values",
            "GET /parameters": "Get default parameter values",
            "GET /outcome_var": "Get the outcome variable name (ppAKT)",
            "GET /states/modifiable": "Get list of modifiable state names",
            "GET /parameters/modifiable": "Get list of modifiable parameter names",
            "GET /health": "Health check",
            "GET /": "API information"
        }
    }


@app.get("/health", tags=["Health"])
async def health():
    """Health check endpoint."""
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    return {
        "status": "healthy",
        "model_dir": str(simulator.model_dir),
        "deterministic": simulator.deterministic
    }


@app.get(
    "/states",
    tags=["Model Info"],
    responses={
        200: {"description": "Returns default state values"},
        503: {"model": ErrorResponse, "description": "Simulator not initialized"}
    }
)
async def get_states():
    """Get default initial values for all states (species).

    Returns a dictionary mapping state names to their default initial
    concentrations in nM. Uses AMICI's native getInitialStates() method.

    Example response:
    ```json
    {
        "Ribosome": 1900.0,
        "ppERK": 11.1,
        "ppAKT": 0.389,
        ...
    }
    ```
    """
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    return simulator.get_state_defaults()


@app.get(
    "/outcome_var",
    tags=["Model Info"],
    responses={
        200: {"description": "Returns the outcome variable name"},
        503: {"model": ErrorResponse, "description": "Simulator not initialized"}
    }
)
async def get_outcome_var():
    """Get the outcome variable name for this model.

    For SPARCED pan-cancer simulations, the outcome variable is 'ppAKT'
    (doubly phosphorylated AKT in Cytoplasm). This is the target species
    used for treatment response prediction.

    Example response:
    ```json
    "ppAKT"
    ```
    """
    return "ppAKT"


@app.get(
    "/states/modifiable",
    tags=["Model Info"],
    responses={
        200: {"description": "Returns list of modifiable state names"},
        503: {"model": ErrorResponse, "description": "Simulator not initialized"}
    }
)
async def get_modifiable_states():
    """Get list of state names that can be perturbed.

    Excludes the core AKT pathway states to conserve AKT->ppAKT behavior:
    - ppAKT (outcome variable)
    - pAKT (direct precursor)
    - AKT (direct precursor)

    All other states are modifiable, including complexes and activated species.

    Example response:
    ```json
    ["Ribosome", "HGF", "PDK1", "PTEN", "ppERK", ...]
    ```
    """
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    return simulator.get_modifiable_states()


@app.get(
    "/parameters",
    tags=["Model Info"],
    responses={
        200: {"description": "Returns default parameter values"},
        503: {"model": ErrorResponse, "description": "Simulator not initialized"}
    }
)
async def get_parameters():
    """Get default values for all fixed parameters.

    Returns a dictionary mapping parameter names to their default values.
    Uses AMICI's native getFixedParameterNames() and getFixedParameterByName() methods.

    Example response:
    ```json
    {
        "k1_1": 0.0042005,
        "k3_1": 140.7475,
        ...
    }
    ```
    """
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    return simulator.get_parameter_defaults()


@app.get(
    "/parameters/modifiable",
    tags=["Model Info"],
    responses={
        200: {"description": "Returns list of modifiable parameter names"},
        503: {"model": ErrorResponse, "description": "Simulator not initialized"}
    }
)
async def get_modifiable_parameters():
    """Get list of parameter names that can be perturbed.

    Excludes only the ppAKT dephosphorylation parameter to conserve decay rate:
    - k1806 (basal ppAKT dephosphorylation)

    All other parameters are modifiable, including AKT phosphorylation kinetics
    (k1754, k1756, k1750) and translation rates.

    Example response:
    ```json
    ["k1_1", "k3_1", "k1754", "k1756", "k105_1", ...]
    ```
    """
    if simulator is None:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    return simulator.get_modifiable_parameters()


@app.post(
    "/simulate",
    tags=["Simulation"],
    responses={
        200: {"description": "Simulation completed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Simulation error"},
        503: {"model": ErrorResponse, "description": "Simulator not initialized"}
    }
)
async def simulate(req: SimulationRequest):
    """Run a SPARCED simulation with the provided parameters.

    Returns simulation results as column-oriented JSON with time and species
    concentrations. The time is in minutes, concentrations in nM.

    Note: SPARCED uses fixed 30-second (0.5 minute) internal timesteps.
    The `step` parameter in the request is informational; all internal
    timesteps are returned in the response.

    Example request:
    ```json
    {
        "start": 0.0,
        "stop": 720.0,
        "step": 0.5,
        "state_values": {"HGF": 50.0},
        "parameter_values": {"k3_1": 150.0}
    }
    ```

    Example response:
    ```json
    {
        "time": [0.0, 0.5, 1.0, ...],
        "Ribosome": [1900.0, 1895.0, ...],
        "p53inac": [297.0, 296.5, ...],
        ...
    }
    ```
    """
    if simulator is None:
        raise HTTPException(status_code=503, detail={"error": "Simulator not initialized"})

    # Acquire lock for exclusive access to simulator
    # This ensures thread safety by serializing all simulation requests
    async with simulate_lock:
        try:
            # Validate state_values species names
            if req.state_values:
                for species_name in req.state_values.keys():
                    if species_name not in simulator.species_name_to_index:
                        raise HTTPException(
                            status_code=400,
                            detail={"error": f"Unknown species: '{species_name}'"}
                        )

            # Validate parameter_values parameter names against AMICI model
            if req.parameter_values:
                for param_name in req.parameter_values.keys():
                    if param_name not in simulator.available_parameters:
                        available = list(simulator.available_parameters)[:5]
                        raise HTTPException(
                            status_code=400,
                            detail={"error": f"Unknown parameter: '{param_name}'. Available parameters (sample): {available}"}
                        )

            # Run simulation
            results_df = simulator.simulate(
                start=req.start,
                stop=req.stop,
                step=req.step,
                state_values=req.state_values,
                parameter_values=req.parameter_values
            )

            # Return column-oriented JSON
            return results_df.to_dict(orient="list")

        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"error": str(e)})
        except Exception as e:
            raise HTTPException(status_code=500, detail={"error": str(e)})


@app.head("/simulate", tags=["Simulation"])
async def simulate_head():
    """HEAD request handler for /simulate endpoint.

    This is required by the HTTPSolver API contract - the client sends a HEAD
    request during compile() to verify the endpoint is reachable.
    """
    return Response(status_code=200)


if __name__ == "__main__":
    import uvicorn

    # Configuration from environment variables
    host = os.getenv("SPARCED_HOST", "0.0.0.0")
    port = int(os.getenv("SPARCED_PORT", "8000"))

    # Run server
    uvicorn.run(app, host=host, port=port)
