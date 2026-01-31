"""SPARCED simulation module for HTTP API.

This module provides a wrapper around the RunSPARCED function to enable
HTTP-based simulation with state and parameter overrides.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import libsbml
import importlib
import amici

# Add bin/ to path for RunSPARCED module import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))
from modules.RunSPARCED import RunSPARCED


class SPARCEDSimulator:
    """Simulator wrapper for SPARCED model via HTTP API.

    The simulator loads the AMICI model and handles temporary file copies
    for applying state and parameter overrides without modifying the
    original input files.

    Time units:
        - API input: minutes
        - SPARCED internal: hours (th), seconds (ts=30)
    """

    # Default ligand concentrations in nM: [EGF, Her, HGF, PDGF, FGF, IGF, INS]
    DEFAULT_LIGANDS = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 1721.0]

    def __init__(
        self,
        model_dir: str,
        sbml_file: str = "SPARCED.xml",
        base_dir: Optional[str] = None,
        deterministic: bool = True
    ):
        """Initialize the SPARCED simulator.

        Args:
            model_dir: Directory containing compiled AMICI model (e.g., "Demo/SPARCED")
            sbml_file: SBML model filename (relative to base_dir)
            base_dir: Base directory containing input files (default: parent of model_dir)
            deterministic: True for deterministic, False for stochastic simulation
        """
        self.model_dir = Path(model_dir).resolve()
        self.sbml_file = sbml_file
        self.deterministic = deterministic

        # Determine base_dir (parent of model_dir, or explicit)
        if base_dir is None:
            self.base_dir = self.model_dir.parent
        else:
            self.base_dir = Path(base_dir).resolve()

        # Input file paths (relative to base_dir)
        self.species_file = self.base_dir / "Species.txt"
        self.omics_data_file = self.base_dir / "OmicsData.txt"
        self.gene_reg_file = self.base_dir / "GeneReg.txt"
        self.ratelaws_file = self.base_dir / "Ratelaws.txt"
        self.sbml_path = self.base_dir / self.sbml_file

        # Load AMICI model
        self._load_model()

        # Cache species names and indices for override mapping
        self._load_species_mapping()
        self._load_ratelaw_mapping()

    def _load_model(self):
        """Load the AMICI model and create solver instance."""
        # Add model directory to Python path
        sys.path.insert(0, str(self.model_dir))

        # Import the model module
        model_name = Path(self.sbml_file).stem
        self.model_module = importlib.import_module(model_name)
        self.model = self.model_module.getModel()
        self.solver = self.model.getSolver()
        self.solver.setMaxSteps = 1e10

        # Cache state IDs
        self.state_ids = self.model.getStateIds()

    def _load_species_mapping(self):
        """Load species names and indices from Species.txt.

        Creates a mapping from species name to row index in the file.
        """
        species_df = pd.read_csv(self.species_file, sep="\t", header=0, index_col=0, encoding='latin-1')
        self.species_name_to_index = {name: idx for idx, name in enumerate(species_df.index)}

    def _load_ratelaw_mapping(self):
        """Load ratelaw and parameter mappings from Ratelaws.txt.

        Creates two mappings:
        1. ratelaw_name -> row index (for validation)
        2. parameter_name -> (row_index, column_offset) (for applying overrides)
        """
        with open(self.ratelaws_file, 'r') as f:
            lines = f.readlines()

        self.ratelaw_name_to_index = {}  # reaction name -> row index
        self.parameter_name_to_location = {}  # parameter name -> (row_index, column_offset)

        for idx, line in enumerate(lines[1:], start=1):  # Skip header
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            ratelaw_name = parts[0]
            self.ratelaw_name_to_index[ratelaw_name] = idx

            # Parse parameters from the ratelaw equation (column 2)
            ratelaw_eq = parts[2]
            # Find all parameters like kTL1_1, kbR_2, etc.
            import re
            params = re.findall(r'(\w+_\d+)', ratelaw_eq)

            # Map each parameter name to its location (row and column offset)
            for param_idx, param in enumerate(params):
                if param not in self.parameter_name_to_location:
                    # Parameters might appear in multiple ratelaws, store first occurrence
                    self.parameter_name_to_location[param] = (idx, 2 + param_idx)

    def _apply_overrides(
        self,
        temp_dir: Path,
        state_values: Optional[Dict[str, float]] = None,
        parameter_values: Optional[Dict[str, float]] = None
    ):
        """Apply state and parameter overrides to temporary copies of input files.

        Args:
            temp_dir: Temporary directory path
            state_values: Species name -> initial concentration (nM) overrides
            parameter_values: Parameter name -> value overrides
        """
        # Copy input files to temp directory
        temp_species_file = temp_dir / "Species.txt"
        temp_ratelaws_file = temp_dir / "Ratelaws.txt"
        temp_omics_file = temp_dir / "OmicsData.txt"
        temp_gene_reg_file = temp_dir / "GeneReg.txt"
        temp_sbml_file = temp_dir / self.sbml_file

        shutil.copy2(self.species_file, temp_species_file)
        shutil.copy2(self.ratelaws_file, temp_ratelaws_file)
        shutil.copy2(self.omics_data_file, temp_omics_file)
        shutil.copy2(self.gene_reg_file, temp_gene_reg_file)
        shutil.copy2(self.sbml_path, temp_sbml_file)

        # Apply state_values to Species.txt
        if state_values:
            species_df = pd.read_csv(temp_species_file, sep="\t", header=0, index_col=0, encoding='latin-1')
            for species_name, value in state_values.items():
                if species_name in species_df.index:
                    species_df.at[species_name, 'IC_Xinitialized'] = float(value)
                else:
                    raise ValueError(f"Species '{species_name}' not found in Species.txt")
            species_df.to_csv(temp_species_file, sep="\t")

        # Apply parameter_values to Ratelaws.txt
        if parameter_values:
            with open(temp_ratelaws_file, 'r') as f:
                lines = f.readlines()

            modified_lines = lines.copy()

            for param_name, param_value in parameter_values.items():
                # Look up parameter location using the pre-built mapping
                if param_name not in self.parameter_name_to_location:
                    raise ValueError(f"Parameter '{param_name}' not found in any ratelaw equation")

                row_idx, col_idx = self.parameter_name_to_location[param_name]
                parts = modified_lines[row_idx].strip().split('\t')

                if col_idx >= len(parts):
                    raise ValueError(f"Parameter column index {col_idx} out of range for parameter '{param_name}'")

                parts[col_idx] = str(float(param_value))
                modified_lines[row_idx] = '\t'.join(parts) + '\n'

            with open(temp_ratelaws_file, 'w') as f:
                f.writelines(modified_lines)

        return {
            'species_file': str(temp_species_file),
            'ratelaws_file': str(temp_ratelaws_file),
            'omics_data_file': str(temp_omics_file),
            'gene_reg_file': str(temp_gene_reg_file),
            'sbml_file': str(temp_sbml_file.name)
        }

    def simulate(
        self,
        start: float,
        stop: float,
        step: float,
        state_values: Optional[Dict[str, float]] = None,
        parameter_values: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """Run SPARCED simulation with provided parameters.

        Args:
            start: Start time in minutes (SPARCED always starts at 0, this is ignored)
            stop: Stop time in minutes (simulation duration)
            step: Step size in minutes (SPARCED uses fixed 30-second = 0.5 min internal steps)
            state_values: Optional species initial concentration overrides (nM)
            parameter_values: Optional ratelaw parameter overrides

        Returns:
            DataFrame with columns: time (minutes), species1, species2, ...
        """
        # Convert stop time from minutes to hours for SPARCED
        th = stop / 60.0

        # Note: start is ignored as SPARCED always starts at t=0
        # The step parameter is also informational; SPARCED outputs every 30 seconds

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Apply overrides to temporary files
            file_paths = self._apply_overrides(
                temp_path,
                state_values=state_values,
                parameter_values=parameter_values
            )

            # Load species initializations from temporary Species.txt
            species_sheet = np.array([
                np.array(line.strip().split("\t"))
                for line in open(file_paths['species_file'], encoding='latin-1')
            ])

            species_initializations = []
            for row in species_sheet[1:]:
                species_initializations.append(float(row[2]))
            species_initializations = np.array(species_initializations)
            species_initializations[np.argwhere(species_initializations <= 1e-6)] = 0.0

            # Update ligand concentrations (positions 155-161)
            # Note: state_values can override individual ligands
            species_initializations[155:162] = self.DEFAULT_LIGANDS
            if state_values:
                # Apply ligand overrides if specified
                ligand_names = ['EGF', 'Her', 'HGF', 'PDGF', 'FGF', 'IGF', 'INS']
                for i, ligand in enumerate(ligand_names):
                    if ligand in state_values:
                        species_initializations[155 + i] = state_values[ligand]

            # Update model's SBML file path to use temporary copy
            temp_sbml_path = file_paths['sbml_file']

            # Create a new model instance with temporary files
            # We need to temporarily change to temp directory for file loading
            original_cwd = os.getcwd()
            original_path = sys.path.copy()

            try:
                os.chdir(temp_path)

                # Set model timepoints for the 30-second internal timestep
                ts = 30  # SPARCED uses fixed 30-second internal timesteps
                self.model.setTimepoints(np.linspace(0, ts, 2))

                # Run simulation
                flagD = 1 if self.deterministic else 0
                xoutS_all, xoutG_all, tout_all = RunSPARCED(
                    flagD,
                    th,
                    species_initializations,
                    [],  # genedata - let RunPrep load from temp files
                    temp_sbml_path,
                    self.model
                )

            finally:
                os.chdir(original_cwd)
                sys.path.clear()
                sys.path.extend(original_path)

        # Convert results to DataFrame
        # tout_all is in seconds, convert to minutes for output
        time_minutes = tout_all / 60.0

        # Create DataFrame with species names as columns
        result_df = pd.DataFrame(xoutS_all, columns=self.state_ids)
        result_df.insert(0, 'time', time_minutes)

        return result_df


def create_simulator(
    model_dir: str = "Demo/SPARCED",
    sbml_file: str = "SPARCED.xml",
    base_dir: str = None,
    deterministic: bool = True
) -> SPARCEDSimulator:
    """Create a SPARCED simulator instance.

    This is a convenience function for creating a simulator with
    environment variable overrides.

    Args:
        model_dir: Path to compiled AMICI model (env: SPARCED_MODEL_DIR)
        sbml_file: SBML filename (env: SPARCED_SBML_FILE)
        base_dir: Base directory for input files (env: SPARCED_BASE_DIR)
        deterministic: True for deterministic, False for stochastic (env: SPARCED_DETERMINISTIC)

    Returns:
        Configured SPARCEDSimulator instance
    """
    model_dir = os.getenv("SPARCED_MODEL_DIR", model_dir)
    sbml_file = os.getenv("SPARCED_SBML_FILE", sbml_file)
    base_dir = os.getenv("SPARCED_BASE_DIR", base_dir)

    det_env = os.getenv("SPARCED_DETERMINISTIC")
    if det_env is not None:
        deterministic = det_env == "1"

    return SPARCEDSimulator(
        model_dir=model_dir,
        sbml_file=sbml_file,
        base_dir=base_dir,
        deterministic=deterministic
    )
