#!/usr/bin/env python3
"""
LEGACY VERSION - kept for backward compatibility with old data extraction.

This version works with the old occupation matrix handling system.
For new extractions, use gather_workchain_data.py instead.

Extract calculations from GlobalConstrainedSearchWorkChain with source tagging.

Identifies and tags calculations by their source:
- 'afm_workchain': Standard PW calculations from AFM scans  
- 'constrained_scan': ConstrainedPW calculations from OSCDFT scans

This enables separate plotting of AFM vs constrained calculations.

Usage:
    # Extract from all global workchains
    data = gather_workchain_data()
    
    # Extract from specific workchain  
    data = gather_workchain_data(workchain_pk=12345)
    
    # Filter by source for plotting
    afm_data = filter_calculations_by_source(data, 'afm_workchain')
    constrained_data = filter_calculations_by_source(data, 'constrained_scan')
"""

import json
import warnings
import numpy as np
from typing import Dict, Any, List, Union, Tuple
from pathlib import Path
from datetime import datetime
from aiida.orm import load_node, CalcJobNode, WorkChainNode
from aiida.common.exceptions import NotExistent
from aiida.plugins import CalculationFactory

# SO(N) decomposition utilities (imported only when needed)
from lordcapulet.utils.so_n_decomposition import (
    get_so_n_lie_basis,
    decompose_rho_and_fix_gauge,
    euler_angles_to_rotation
)

# Try to import alive_progress for progress bar
try:
    from alive_progress import alive_bar
    HAS_ALIVE_BAR = True
except ImportError:
    HAS_ALIVE_BAR = False
    warnings.warn("alive_progress not available. Install with: pip install alive-progress")

# Try to import custom calculation types
try:
    from lordcapulet.calculations.constrained_pw import ConstrainedPWCalculation
    HAS_CONSTRAINED_PW = True
except ImportError:
    HAS_CONSTRAINED_PW = False
    warnings.warn("Could not import ConstrainedPWCalculation. Will try to identify by process type.")

# Import unified occupation matrix utilities
try:
    from lordcapulet.utils import extract_occupations_from_calc
    HAS_UNIFIED_EXTRACTION = True
except ImportError:
    HAS_UNIFIED_EXTRACTION = False
    warnings.warn("Could not import unified occupation matrix utilities.")


# =============================================================================
# CODE ORGANIZATION PHILOSOPHY
# =============================================================================
"""
This module is organized around a clear separation of concerns:

INSIDE WorkchainExtractor Class:
================================
- User-facing interface for workchain and calculation extraction
- Configuration management (SO(N) options, debug levels, etc.)
- Node type validation with helpful error messages
- Data processing orchestration with progress tracking
- Regularization statistics collection and summarized reporting
- JSON export functionality
- Output data structure building

The class provides a clean, stateful interface that:
1. Validates input node types and gives helpful error messages
2. Manages SO(N) decomposition settings consistently across operations
3. Tracks and summarizes regularization statistics 
4. Provides progress feedback and consolidated reporting
5. Handles all user-facing concerns (validation, feedback, export)

OUTSIDE WorkchainExtractor Class (Utility Functions):
====================================================
- Low-level AiiDA node inspection and classification
- Workchain traversal and calculation discovery algorithms
- Raw data extraction from individual calculation nodes
- Legacy SO(N) decomposition implementation (for backward compatibility)
- AiiDA query builders for node discovery
- Data filtering and transformation utilities

DESIGN RATIONALE:
================
- The class handles "what the user wants" (extract workchain, debug calculation, etc.)
- The utilities handle "how AiiDA works" (node traversal, data extraction, etc.)
"""


class WorkchainExtractor:
    """
    A class to extract and analyze calculations from AiiDA workchains.
    
    Handles extraction from GlobalConstrainedSearchWorkChain, AFMScanWorkChain, 
    ConstrainedScanWorkChain, or individual calculations with optional SO(N) decomposition.
    
    Features:
    - Automatic workchain type detection and validation
    - Single calculation analysis for debugging
    - SO(N) decomposition with regularization and consolidated reporting
    - Progress tracking with summary statistics
    - JSON export functionality
    
    Methods Overview:
    ================
    
    Main extraction methods:
    - extract_from_workchain(workchain_pk) -> Extract all calculations from a workchain
    - extract_single_calculation(calc_pk) -> Extract and analyze one calculation (debugging)
    
    Step-by-step debugging methods:
    - extract_occupation_matrices(calc_pk) -> Step 1: Extract only occupation matrices
    - decompose_single_matrix(matrix, atom, spin) -> Step 2: Decompose a single matrix
    - step_through_calculation(calc_pk) -> Complete step-by-step analysis with detailed output
    
    Utility methods:
    - save_to_json(data, filename) -> Save extracted data to JSON file
    - get_extraction_summary(data) -> Get concise summary of extraction results
    - get_regularization_summary() -> Get summary of regularization statistics
    - get_regularization_details() -> Get list of specific calculations/atoms that needed regularization
    
    Usage Examples:
    ===============
    
    # Basic workchain extraction:
    extractor = WorkchainExtractor()
    data = extractor.extract_from_workchain(workchain_pk=12345)
    extractor.save_to_json(data, "results.json")
    
    # With SO(N) decomposition:
    extractor = WorkchainExtractor(perform_so_n=True, sanity_check_reconstruct_rho=True)
    data = extractor.extract_from_workchain(workchain_pk=12345)
    print(extractor.get_extraction_summary(data))
    
    # Single calculation debugging:
    debug_extractor = WorkchainExtractor(perform_so_n=True, debug=True, verbose_warnings=True)
    calc_data = debug_extractor.extract_single_calculation(calc_pk=67890)
    
    # Step-by-step debugging (new feature):
    step_extractor = WorkchainExtractor(perform_so_n=True, sanity_check_reconstruct_rho=True, debug=True)
    step_results = step_extractor.step_through_calculation(calc_pk=67890)
    
    # Or individual steps:
    occupation_matrices = step_extractor.extract_occupation_matrices(calc_pk=67890)
    matrix = occupation_matrices['1']['spin_data']['up']['occupation_matrix']
    decomp_result = step_extractor.decompose_single_matrix(matrix, atom_key='1', spin='up')
    
    # Get regularization summary after analysis:
    reg_summary = debug_extractor.get_regularization_summary()
    print(reg_summary)  # Shows overall statistics and points to details method
    
    # Get specific list of affected calculations:
    reg_details = debug_extractor.get_regularization_details()
    # Returns: [(pk, atom, success), ...] e.g., [(12345, "1_up", True), (12346, "2_down", False)]
    
    # Error handling (automatic node type validation):
    try:
        data = extractor.extract_from_workchain(calc_pk)  # Wrong method for calculation
    except ValueError as e:
        print(e)  # "PK 67890 is a calculation, not a workchain. Use extract_single_calculation() method instead."
        calc_data = extractor.extract_single_calculation(calc_pk)  # Correct method
    
    Configuration Options:
    ======================
    - perform_so_n: Enable SO(N) decomposition on occupation matrices
    - sanity_check_reconstruct_rho: Include reconstruction validation in SO(N) results
    - debug: Show detailed per-calculation processing information
    - verbose_warnings: Show individual regularization warnings (vs. summary only)
    """
    
    SUPPORTED_WORKCHAIN_TYPES = [
        'globalconstrained', 'global_constrained',
        'afmscan', 'afm_scan', 
        'constrainedscan', 'constrained_scan'
    ]
    
    def __init__(self, perform_so_n: bool = False, sanity_check_reconstruct_rho: bool = False, 
                 debug: bool = False, verbose_warnings: bool = False, include_non_converged: bool = False):
        """
        Initialize the WorkchainExtractor.
        
        Args:
            perform_so_n: Whether to perform SO(N) decomposition on occupation matrices
            sanity_check_reconstruct_rho: Whether to include reconstruction validation
            debug: Whether to print detailed debug information
            verbose_warnings: Whether to show detailed warnings (vs summary at end)
            include_non_converged: Whether to include calculations with exit status 410 (non-converged SCF)
        """
        self.perform_so_n = perform_so_n
        self.sanity_check_reconstruct_rho = sanity_check_reconstruct_rho
        self.debug = debug
        self.verbose_warnings = verbose_warnings
        self.include_non_converged = include_non_converged
        
        # Track regularization statistics
        self.regularization_stats = {
            'total_decompositions': 0,
            'regularizations_needed': 0,
            'regularizations_successful': 0,
            'regularizations_failed': 0,
            'regularization_details': []  # List of (pk, atom, success) tuples
        }
    
    def _validate_node_type(self, node_pk: int, expected_type: str = 'auto') -> Tuple[Any, str]:
        """
        Load and validate an AiiDA node, determining if it's a workchain or calculation.
        
        Args:
            node_pk: Primary key of the node to load
            expected_type: 'workchain', 'calculation', or 'auto' for automatic detection
            
        Returns:
            tuple: (loaded_node, actual_type) where actual_type is 'workchain' or 'calculation'
            
        Raises:
            NotExistent: If node cannot be loaded
            ValueError: If node type doesn't match expected type
        """
        try:
            node = load_node(node_pk)
        except Exception as e:
            raise NotExistent(f"Could not load node with PK {node_pk}: {str(e)}")
        
        # Determine actual node type
        if isinstance(node, WorkChainNode):
            actual_type = 'workchain'
            process_type = getattr(node, 'process_type', '').lower()
            
            # Check if it's a supported workchain type
            is_supported = any(wc_type in process_type for wc_type in self.SUPPORTED_WORKCHAIN_TYPES)
            if not is_supported:
                supported_list = ', '.join(self.SUPPORTED_WORKCHAIN_TYPES)
                raise ValueError(
                    f"Workchain PK {node_pk} has unsupported type '{process_type}'. "
                    f"Supported types: {supported_list}"
                )
                
        elif isinstance(node, CalcJobNode):
            actual_type = 'calculation'
            
            # Check if it's a PW calculation
            if not is_pw_calculation(node):
                raise ValueError(
                    f"Calculation PK {node_pk} is not a PW or ConstrainedPW calculation. "
                    f"Process type: {getattr(node, 'process_type', 'unknown')}"
                )
        else:
            raise ValueError(
                f"Node PK {node_pk} is neither a WorkChain nor a CalcJob. "
                f"Node type: {type(node).__name__}"
            )
        
        # Validate against expected type
        if expected_type != 'auto' and actual_type != expected_type:
            if actual_type == 'calculation' and expected_type == 'workchain':
                raise ValueError(
                    f"PK {node_pk} is a calculation, not a workchain. "
                    f"Use extract_single_calculation() method instead."
                )
            elif actual_type == 'workchain' and expected_type == 'calculation':
                raise ValueError(
                    f"PK {node_pk} is a workchain, not a calculation. "
                    f"Use extract_from_workchain() method instead."
                )
        
        return node, actual_type
    
    def extract_from_workchain(self, workchain_pk: int) -> Dict[str, Any]:
        """
        Extract all calculations from a specific workchain.
        
        Args:
            workchain_pk: Primary key of the workchain to process
            
        Returns:
            dict: Complete data dictionary with metadata, statistics, and calculations
            
        Raises:
            ValueError: If PK is not a supported workchain type
            NotExistent: If workchain cannot be loaded
        """
        # Validate that this is a workchain
        workchain_node, node_type = self._validate_node_type(workchain_pk, expected_type='workchain')
        
        process_type = getattr(workchain_node, 'process_type', '').lower()
        print(f"Extracting from workchain PK {workchain_pk}")
        print(f"Workchain type: {getattr(workchain_node, 'process_type', 'unknown')}")
        
        # Use existing extraction logic based on workchain type
        if any(wc_type in process_type for wc_type in ['globalconstrained', 'global_constrained']):
            extraction_method = "global_constrained_workchain"
            converged_calculations = extract_calculations_from_global_workchain(workchain_node, self.debug, self.include_non_converged)
            
        elif any(wc_type in process_type for wc_type in ['afmscan', 'afm_scan']):
            extraction_method = "afm_workchain"
            converged_calculations = _extract_pw_calculations_from_workchain(workchain_node, 'afm_workchain', include_non_converged=self.include_non_converged)
            
        elif any(wc_type in process_type for wc_type in ['constrainedscan', 'constrained_scan']):
            extraction_method = "constrained_workchain"
            converged_calculations = _extract_pw_calculations_from_workchain(workchain_node, 'constrained_scan', include_non_converged=self.include_non_converged)
        
        else:
            # This shouldn't happen due to validation, but just in case
            extraction_method = "unknown_workchain"
            converged_calculations = _extract_pw_calculations_from_workchain(workchain_node, 'unknown', include_non_converged=self.include_non_converged)
        
        if self.include_non_converged:
            print(f"Found {len(converged_calculations)} calculations (including non-converged)")
        else:
            print(f"Found {len(converged_calculations)} converged calculations")
        
        # Process the calculations
        calculations_data = self._process_calculations(converged_calculations)
        
        # Build the complete data structure
        return self._build_output_data(
            calculations_data=calculations_data,
            all_calculations=converged_calculations,
            extraction_method=extraction_method,
            workchain_info={
                'pk': workchain_pk,
                'process_type': getattr(workchain_node, 'process_type', 'unknown'),
                'node_type': type(workchain_node).__name__
            }
        )
    
    def extract_single_calculation(self, calc_pk: int) -> Dict[str, Any]:
        """
        Extract and analyze a single calculation (useful for debugging).
        
        Args:
            calc_pk: Primary key of the calculation to process
            
        Returns:
            dict: Calculation data with optional SO(N) decomposition
            
        Raises:
            ValueError: If PK is not a calculation
            NotExistent: If calculation cannot be loaded
        """
        # Validate that this is a calculation
        calc_node, node_type = self._validate_node_type(calc_pk, expected_type='calculation')
        
        print(f"Extracting single calculation PK {calc_pk}")
        print(f"Calculation type: {getattr(calc_node, 'process_type', 'unknown')}")
        
        # Extract the calculation data
        calc_data = self._extract_single_calc_data(calc_node)
        
        if calc_data is None:
            raise ValueError(f"Failed to extract data from calculation PK {calc_pk}")
        
        # Show regularization summary if SO(N) was performed
        if self.perform_so_n:
            self._print_regularization_summary()
        
        return calc_data
    
    def _process_calculations(self, calculation_list: List[Tuple[int, int, str, str]]) -> List[Dict[str, Any]]:
        """
        Process a list of converged calculations and extract data.
        
        Args:
            calculation_list: List of tuples (pk, exit_status, process_type, source)
            
        Returns:
            list: List of calculation data dictionaries
        """
        # Reset regularization stats for this extraction
        self.regularization_stats = {
            'total_decompositions': 0,
            'regularizations_needed': 0,
            'regularizations_successful': 0,
            'regularizations_failed': 0,
            'regularization_details': []  # List of (pk, atom, success) tuples
        }
        
        results = []
        
        if HAS_ALIVE_BAR and len(calculation_list) > 0:
            with alive_bar(len(calculation_list), title="Processing calculations") as bar:
                for pk, exit_status, process_type, source in calculation_list:
                    calc_data = self._process_single_calculation(pk, process_type, source)
                    if calc_data:
                        results.append(calc_data)
                    bar()
        else:
            # Fallback without progress bar
            print(f"Processing {len(calculation_list)} converged calculations...")
            for i, (pk, exit_status, process_type, source) in enumerate(calculation_list):
                calc_data = self._process_single_calculation(pk, process_type, source)
                if calc_data:
                    results.append(calc_data)
                
                # Print progress every 10%
                if not self.debug and i % max(1, len(calculation_list) // 10) == 0:
                    progress = (i + 1) / len(calculation_list) * 100
                    print(f"Progress: {progress:.1f}% ({i + 1}/{len(calculation_list)})")
        
        # Print regularization summary
        if self.perform_so_n:
            self._print_regularization_summary()
        
        return results
    
    def _process_single_calculation(self, pk: int, process_type: str, source: str) -> Dict[str, Any]:
        """Process a single calculation and handle errors gracefully."""
        try:
            calc_node = load_node(pk)
            calc_data = self._extract_single_calc_data(calc_node)
            
            if calc_data and self.debug:
                print(f"Processed {process_type} calculation: PK {pk} (source: {source})")
            
            return calc_data
            
        except Exception as e:
            if self.debug or self.verbose_warnings:
                print(f"Error processing calculation PK {pk}: {str(e)}")
            return None
    
    def _print_regularization_summary(self):
        """Print a summary of regularization statistics (internal method)."""
        summary = self.get_regularization_summary()
        if summary:
            print(summary)
    
    def get_regularization_summary(self) -> str:
        """
        Get a detailed summary of regularization statistics.
        
        Returns:
            str: Formatted summary string, or empty string if no decompositions performed
        """
        stats = self.regularization_stats
        if stats['total_decompositions'] == 0:
            return ""
        
        summary_lines = [
            f"\nSO(N) Decomposition Summary:",
            f"  Total decompositions attempted: {stats['total_decompositions']}",
            f"  Regularizations needed: {stats['regularizations_needed']}"
        ]
        
        if stats['regularizations_needed'] > 0:
            success_rate = stats['regularizations_successful'] / stats['regularizations_needed'] * 100
            summary_lines.append(f"  Regularizations successful: {stats['regularizations_successful']} ({success_rate:.1f}%)")
            if stats['regularizations_failed'] > 0:
                summary_lines.append(f"  Regularizations failed: {stats['regularizations_failed']}")
            
            # Add note about detailed breakdown method
            summary_lines.append(f"\nFor detailed list of affected calculations and atoms, use:")
            summary_lines.append(f"  extractor.get_regularization_details()")
        
        return '\n'.join(summary_lines)
    
    def save_to_json(self, data: Dict[str, Any], filename: str) -> None:
        """
        Save extracted data to a JSON file.
        
        Args:
            data: Data dictionary to save
            filename: Output filename
        """
        output_path = Path(filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to: {output_path.absolute()}")
    
    def get_extraction_summary(self, data: Dict[str, Any]) -> str:
        """
        Get a concise summary of extraction results.
        
        Args:
            data: Data dictionary from extraction
            
        Returns:
            str: Formatted summary string
        """
        metadata = data.get('metadata', {})
        statistics = data.get('statistics', {})
        
        summary_lines = [
            f"Extraction completed: {metadata.get('extraction_method', 'unknown')}",
            f"Total calculations: {metadata.get('total_calculations_found', 0)}",
            f"Extraction timestamp: {metadata.get('extraction_timestamp', 'unknown')}"
        ]
        
        # Add source breakdown
        sources = statistics.get('calculation_sources', {})
        if sources:
            summary_lines.append("Source breakdown:")
            for source, count in sources.items():
                summary_lines.append(f"  {source}: {count}")
        
        # add regularization summary 
        reg_summary = self.get_regularization_summary()
        summary_lines.append(reg_summary)
        
        return '\n'.join(summary_lines)
    
    def get_regularization_details(self) -> List[Tuple[int, str, bool]]:
        """
        Get detailed list of calculations and atoms that needed regularization.
        
        Returns:
            list: List of tuples (pk, atom, success) where:
                  - pk: Calculation primary key
                  - atom: Atom identifier in format "atom_key_spin" (e.g., "1_up", "2_down")
                  - success: Boolean indicating if regularization was successful
        """
        return self.regularization_stats.get('regularization_details', [])
    
    def extract_occupation_matrices(self, calc_pk: int) -> Dict[str, Any]:
        """
        Step 1: Extract only the occupation matrices from a calculation.
        
        Args:
            calc_pk: Primary key of the calculation
            
        Returns:
            dict: Occupation matrices data
        """
        calc_node, _ = self._validate_node_type(calc_pk, expected_type='calculation')
        
        try:
            if 'output_atomic_occupations' in calc_node.outputs:
                occupations = calc_node.outputs.output_atomic_occupations
                if hasattr(occupations, 'get_dict'):
                    return occupations.get_dict()
        except Exception as e:
            if self.debug:
                print(f"Error extracting occupation matrices: {e}")
        
        return {}
    
    def decompose_single_matrix(self, occ_matrix: np.ndarray, atom_key: str = "debug", spin: str = "debug") -> Dict[str, Any]:
        """
        Step 2: Decompose a single occupation matrix using the complete SO(N) approach.
        
        Args:
            occ_matrix: The occupation matrix to decompose
            atom_key: Identifier for the atom (for debugging)
            spin: Spin channel identifier (for debugging)
            
        Returns:
            dict: Complete decomposition results
        """
        if not isinstance(occ_matrix, np.ndarray):
            occ_matrix = np.array(occ_matrix)
        
        if occ_matrix.shape[0] != occ_matrix.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {occ_matrix.shape}")
        
        dim = occ_matrix.shape[0]
        generators = get_so_n_lie_basis(dim)
        
        if self.debug:
            print(f"Decomposing {dim}x{dim} matrix for {atom_key}_{spin}")
            print(f"Matrix trace: {np.trace(occ_matrix):.6f}")
        
        # Use the complete decomposition function
        eigenvalues, eigenvectors, euler_angles, need_regularization, degenerate_groups = decompose_rho_and_fix_gauge(
            occ_matrix, generators
        )
        
        result = {
            'eigenvalues': eigenvalues.tolist(),
            'eigenvectors': eigenvectors.tolist(),
            'euler_angles': euler_angles.tolist(),
            'need_regularization': bool(need_regularization),
            'degenerate_groups': degenerate_groups,
            'matrix_dimension': dim,
            'trace': float(np.trace(occ_matrix))
        }
        
        # Add reconstruction check if requested
        if self.sanity_check_reconstruct_rho:
            R_reconstructed = euler_angles_to_rotation(euler_angles, generators)
            rho_reconstructed = R_reconstructed @ np.diag(eigenvalues) @ R_reconstructed.T
            reconstruction_error = float(np.max(np.abs(rho_reconstructed - occ_matrix)))
            
            result['sanity_check'] = {
                'reconstructed_matrix': rho_reconstructed.tolist(),
                'reconstruction_error': reconstruction_error
            }
            
            if self.debug:
                print(f"Reconstruction error: {reconstruction_error:.2e}")
        
        return result
    
    def step_through_calculation(self, calc_pk: int) -> Dict[str, Any]:
        """
        Step through the SO(N) decomposition of a calculation step by step.
        
        Args:
            calc_pk: Primary key of the calculation
            
        Returns:
            dict: Complete step-by-step analysis
        """
        print(f"=== Step-by-step SO(N) analysis for calculation {calc_pk} ===")
        
        # Step 1: Extract occupation matrices
        print("Step 1: Extracting occupation matrices...")
        occupation_matrices = self.extract_occupation_matrices(calc_pk)
        
        if not occupation_matrices:
            return {'error': 'No occupation matrices found'}
        
        print(f"Found occupation data for atoms: {list(occupation_matrices.keys())}")
        
        # Step 2: Process each atom and spin
        results = {}
        for atom_key, atom_data in occupation_matrices.items():
            if not isinstance(atom_data, dict) or 'spin_data' not in atom_data:
                continue
                
            print(f"\nStep 2: Processing atom {atom_key}")
            atom_results = {}
            
            spin_data = atom_data['spin_data']
            for spin in ['up', 'down']:
                if spin not in spin_data or 'occupation_matrix' not in spin_data[spin]:
                    continue
                
                print(f"  Processing {spin} spin channel...")
                occ_matrix = np.array(spin_data[spin]['occupation_matrix'])
                
                try:
                    decomp_result = self.decompose_single_matrix(occ_matrix, atom_key, spin)
                    atom_results[f'{spin}_spin'] = decomp_result
                    
                    # Print key results
                    print(f"    ✓ Eigenvalues: {decomp_result['eigenvalues']}")
                    print(f"    ✓ Regularization needed: {decomp_result['need_regularization']}")
                    if decomp_result['degenerate_groups']:
                        print(f"    ✓ Degenerate eigenvalue groups: {decomp_result['degenerate_groups']}")
                    
                except Exception as e:
                    print(f"    ✗ Error: {e}")
                    atom_results[f'{spin}_spin'] = {'error': str(e)}
            
            if atom_results:
                results[atom_key] = atom_results
        
        print(f"\n=== Analysis complete for calculation {calc_pk} ===")
        return results
    
    def _extract_single_calc_data(self, calc_node: CalcJobNode) -> Dict[str, Any]:
        """Extract data from a single calculation node."""
        try:
            data = {
                'pk': calc_node.pk,
                'exit_status': calc_node.exit_status,
                'converged': calc_node.exit_status == 0,
                'convergence_status': 'converged' if calc_node.exit_status == 0 else ('non_converged_scf' if calc_node.exit_status == 410 else 'other_error'),
                'process_type': getattr(calc_node, 'process_type', 'unknown'),
                'calculation_source': _determine_calculation_source(calc_node),
                'inputs': {},
                'output_parameters': None,
                'output_atomic_occupations': None
            }
        except Exception as e:
            if self.debug:
                print(f"Error creating basic data structure for node {calc_node.pk}: {e}")
            return None
        
        # Extract inputs with proper AiiDA API
        try:
            if hasattr(calc_node, 'inputs'):
                incoming_links = calc_node.base.links.get_incoming()
                for link in incoming_links:
                    try:
                        key = link.link_label
                        input_node = link.node
                        
                        if hasattr(input_node, 'get_dict'):
                            data['inputs'][key] = input_node.get_dict()
                        else:
                            data['inputs'][key] = {
                                'node_type': getattr(input_node, 'node_type', 'unknown'),
                                'pk': input_node.pk,
                                'uuid': str(input_node.uuid)
                            }
                    except Exception as e:
                        if hasattr(link, 'link_label'):
                            data['inputs'][link.link_label] = f"Error extracting input '{link.link_label}': {str(e)}"
                        else:
                            data['inputs'][f'link_error_{len(data["inputs"])}'] = f"Error processing input link: {str(e)}"
            else:
                data['inputs'] = "Node has no inputs attribute"
                
        except Exception as e:
            data['inputs'] = f"Error accessing inputs for node {calc_node.pk}: {str(e)}"
        
        # Extract output_parameters
        try:
            if 'output_parameters' in calc_node.outputs:
                output_params = calc_node.outputs.output_parameters
                if hasattr(output_params, 'get_dict'):
                    data['output_parameters'] = output_params.get_dict()
        except Exception as e:
            data['output_parameters'] = f"Error extracting output_parameters: {str(e)}"
        
        # Extract output_atomic_occupations
        try:
            if 'output_atomic_occupations' in calc_node.outputs:
                occupations = calc_node.outputs.output_atomic_occupations
                if hasattr(occupations, 'get_dict'):
                    data['output_atomic_occupations'] = occupations.get_dict()
                elif hasattr(occupations, 'get_array'):
                    data['output_atomic_occupations'] = {
                        name: array.tolist() for name, array in occupations.get_arraydict().items()
                    }
        except Exception as e:
            data['output_atomic_occupations'] = f"Error extracting output_atomic_occupations: {str(e)}"
        
        # Perform SO(N) decomposition if requested
        if self.perform_so_n and data['output_atomic_occupations'] and isinstance(data['output_atomic_occupations'], dict):
            try:
                so_n_results = self._perform_so_n_analysis(data['output_atomic_occupations'], calc_pk=data['pk'])
                data['so_n_decomposition'] = so_n_results
            except Exception as e:
                data['so_n_decomposition'] = {
                    'decomposition_successful': False,
                    'error_message': f"SO(N) decomposition failed: {str(e)}"
                }
        elif self.perform_so_n:
            data['so_n_decomposition'] = {
                'decomposition_successful': False,
                'error_message': "No valid occupation matrices available for SO(N) decomposition"
            }
        
        return data
    
    def _perform_so_n_analysis(self, occupation_matrices: Dict[str, Any], calc_pk: int = None) -> Dict[str, Any]:
        """
        Perform SO(N) decomposition with regularization tracking.
        
        Args:
            occupation_matrices: Dictionary containing occupation matrices from AiiDA output
            calc_pk: Primary key of the calculation (for detailed regularization tracking)
            
        Returns:
            dict: SO(N) decomposition results
        """
        decomposition_results = {
            'atom_decompositions': {},
            'decomposition_successful': False,
            'error_message': None
        }
        
        try:
            if not occupation_matrices or not isinstance(occupation_matrices, dict):
                decomposition_results['error_message'] = "No valid occupation matrices found"
                return decomposition_results
            
            # Process each atom
            for atom_key, atom_data in occupation_matrices.items():
                if not isinstance(atom_data, dict) or 'spin_data' not in atom_data:
                    continue
                
                atom_decomp = {}
                spin_data = atom_data['spin_data']
                
                # Process up and down spin channels
                for spin in ['up', 'down']:
                    if spin not in spin_data or 'occupation_matrix' not in spin_data[spin]:
                        continue
                    
                    try:
                        # Get the occupation matrix
                        occ_matrix = np.array(spin_data[spin]['occupation_matrix'])
                        
                        # Check if matrix is square
                        if occ_matrix.shape[0] != occ_matrix.shape[1]:
                            if self.debug:
                                print(f"Skipping non-square matrix for atom {atom_key}, spin {spin}")
                            continue
                        
                        dim = occ_matrix.shape[0]
                        
                        # Generate SO(N) basis
                        generators = get_so_n_lie_basis(dim)
                        
                        # Update statistics
                        self.regularization_stats['total_decompositions'] += 1
                        
                        # Use the complete decomposition function
                        eigenvalues, eigenvectors, euler_angles, need_regularization, degenerate_groups = decompose_rho_and_fix_gauge(
                            occ_matrix, generators
                        )
                        
                        # Track regularization statistics (decompose_rho_and_fix_gauge handles the actual regularization)
                        regularization_applied = need_regularization
                        if need_regularization:
                            self.regularization_stats['regularizations_needed'] += 1
                            self.regularization_stats['regularizations_successful'] += 1  # assume success since function completed
                            
                            # Record regularization details
                            if calc_pk is not None:
                                self.regularization_stats['regularization_details'].append(
                                    (calc_pk, f"{atom_key}_{spin}", True)
                                )
                            
                            if self.verbose_warnings:
                                print(f"Regularization applied for atom {atom_key}, {spin} spin during gauge fixing")
                        
                        # Store results
                        result_data = {
                            'eigenvalues': eigenvalues.tolist(),
                            'euler_angles': euler_angles.tolist(),
                            'matrix_dimension': dim,
                            'trace': float(np.trace(occ_matrix)),
                            'need_regularization': bool(need_regularization),
                            'regularization_applied': bool(regularization_applied),
                            'degenerate_groups': degenerate_groups
                        }
                        
                        # Add sanity check reconstruction if requested
                        if self.sanity_check_reconstruct_rho:
                            # Reconstruct the rotation matrix from Euler angles
                            R_reconstructed = euler_angles_to_rotation(euler_angles, generators)
                            
                            # Reconstruct the density matrix: rho = R * diag(eigenvals) * R^T
                            rho_reconstructed = R_reconstructed @ np.diag(eigenvalues) @ R_reconstructed.T
                            
                            # Calculate reconstruction error
                            reconstruction_error = float(np.max(np.abs(rho_reconstructed - occ_matrix)))
                            
                            # Add to results
                            result_data['sanity_check'] = {
                                'reconstructed_density_matrix': rho_reconstructed.tolist(),
                                'reconstruction_error': reconstruction_error
                            }
                            
                            if self.debug:
                                print(f"  Reconstruction error: {reconstruction_error:.2e}")
                        
                        atom_decomp[f'{spin}_spin'] = result_data
                        
                        if self.debug:
                            print(f"SO(N) decomposition successful for atom {atom_key}, {spin} spin:")
                            print(f"  Eigenvalues: {eigenvalues}")
                            print(f"  Matrix trace: {np.trace(occ_matrix):.6f}")
                    
                    except Exception as e:
                        if self.debug or self.verbose_warnings:
                            print(f"SO(N) decomposition failed for atom {atom_key}, {spin} spin: {e}")
                        atom_decomp[f'{spin}_spin'] = {
                            'error': str(e),
                            'decomposition_failed': True
                        }
                
                if atom_decomp:
                    decomposition_results['atom_decompositions'][atom_key] = atom_decomp
            
            # Mark as successful if we processed at least one atom
            if decomposition_results['atom_decompositions']:
                decomposition_results['decomposition_successful'] = True
            else:
                decomposition_results['error_message'] = "No atoms could be processed"
        
        except Exception as e:
            decomposition_results['error_message'] = f"General SO(N) decomposition error: {str(e)}"
            if self.debug:
                print(f"SO(N) decomposition error: {e}")
        
        return decomposition_results
    
    def _build_output_data(self, calculations_data: List[Dict[str, Any]], 
                          all_calculations: List[Tuple[int, int, str, str]],
                          extraction_method: str, workchain_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build the complete output data structure."""
        # Build statistics from all calculations
        source_counts = {}
        calc_type_counts = {}
        convergence_counts = {'converged': 0, 'non_converged_scf': 0, 'other_error': 0}
        
        for pk, exit_status, process_type, source in all_calculations:
            source_counts[source] = source_counts.get(source, 0) + 1
            calc_type_counts[process_type] = calc_type_counts.get(process_type, 0) + 1
            
            if exit_status == 0:
                convergence_counts['converged'] += 1
            elif exit_status == 410:
                convergence_counts['non_converged_scf'] += 1
            else:
                convergence_counts['other_error'] += 1
        
        statistics = {
            'total_calculations': len(all_calculations),
            'converged_calculations': convergence_counts['converged'],
            'non_converged_scf_calculations': convergence_counts['non_converged_scf'],
            'other_error_calculations': convergence_counts['other_error'],
            'convergence_breakdown': convergence_counts,
            'calculation_types': calc_type_counts,
            'calculation_sources': source_counts
        }
        
        # Prepare metadata
        metadata = {
            'total_calculations_found': len(calculations_data),
            'extraction_method': extraction_method,
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        # Add workchain info if available
        if workchain_info:
            metadata.update(workchain_info)
        
        # Build output structure
        output_data = {
            'metadata': metadata,
            'statistics': statistics,
            'calculations': {}
        }
        
        # Store calculations with PK as key
        for calc_data in calculations_data:
            pk = calc_data['pk']
            output_data['calculations'][str(pk)] = calc_data
        
        return output_data


# =============================================================================
# UTILITY FUNCTIONS (Outside WorkchainExtractor Class)
# =============================================================================
# These functions handle low-level AiiDA operations and are used both by the
# WorkchainExtractor class and available for direct use in legacy code.
# They are stateless and focused on specific technical tasks.


def is_pw_calculation(node: CalcJobNode) -> bool:
    """
    Check if a calculation node is a PW or ConstrainedPW calculation.
    
    Args:
        node: AiiDA calculation node to check
        
    Returns:
        bool: True if node is a PW or ConstrainedPW calculation
    """
    if not isinstance(node, CalcJobNode):
        return False
    
    # Check process type string
    process_type = getattr(node, 'process_type', '')
    
    # Check for standard PW calculation
    if 'quantumespresso.pw' in process_type:
        return True
    
    # Check for ConstrainedPW calculation
    if 'lordcapulet.constrained_pw' in process_type or 'ConstrainedPW' in process_type:
        return True
    
    # Check by class if available
    if HAS_CONSTRAINED_PW:
        try:
            PwCalculation = CalculationFactory('quantumespresso.pw')
            if isinstance(node, (PwCalculation, ConstrainedPWCalculation)):
                return True
        except Exception:
            pass
    
    # Additional check by node attributes or labels
    node_type = getattr(node, 'node_type', '').lower()
    if 'pw' in node_type and 'calculation' in node_type:
        return True
        
    return False


def _determine_calculation_source(calc_node: CalcJobNode) -> str:
    """
    Determine the source of a calculation (AFM workchain vs constrained scan).
    
    Args:
        calc_node: AiiDA calculation node
        
    Returns:
        str: 'afm_workchain' for standard PW calculations (typically from AFMScanWorkChain),
             'constrained_scan' for ConstrainedPW calculations (typically from ConstrainedScanWorkChain),
             'unknown' if cannot be determined
    """
    process_type = getattr(calc_node, 'process_type', '').lower()
    
    # ConstrainedPW calculations are from constrained scans
    if 'lordcapulet.constrained_pw' in process_type or 'constrainedpw' in process_type:
        return 'constrained_scan'
    
    # Standard QE PW calculations are typically from AFM workchains
    if 'quantumespresso.pw' in process_type:
        return 'afm_workchain'
    
    # Fallback: try to infer from caller workchain by examining the node's ancestry
    try:
        # Walk up the call tree to find the parent workchain
        caller = getattr(calc_node, 'caller', None)
        if caller:
            caller_process_type = getattr(caller, 'process_type', '').lower()
            if 'afmscan' in caller_process_type or 'afm_scan' in caller_process_type:
                return 'afm_workchain'
            elif 'constrainedscan' in caller_process_type or 'constrained_scan' in caller_process_type:
                return 'constrained_scan'
            elif 'globalconstrained' in caller_process_type or 'global_constrained' in caller_process_type:
                # For global search, need to check the calculation type itself
                if 'lordcapulet.constrained_pw' in process_type:
                    return 'constrained_scan'
                else:
                    return 'afm_workchain'
    except Exception:
        # If we can't access caller information, continue with other methods
        pass
    
    # Additional heuristic: check input parameters for OSCDFT-specific keys
    try:
        if hasattr(calc_node, 'inputs'):
            inputs = calc_node.inputs
            # Look for OSCDFT-specific inputs that indicate constrained calculations
            oscdft_indicators = ['oscdft_card', 'target_matrix', 'occupation_matrix']
            for key in inputs.keys():
                if any(indicator in key.lower() for indicator in oscdft_indicators):
                    return 'constrained_scan'
            
            # Check parameters for OSCDFT-related settings
            if 'parameters' in inputs:
                params = inputs.parameters
                if hasattr(params, 'get_dict'):
                    param_dict = params.get_dict()
                    # Look for OSCDFT namelist or parameters
                    if 'OSCDFT' in param_dict or any('oscdft' in str(v).lower() for v in param_dict.values()):
                        return 'constrained_scan'
    except Exception:
        pass
    
    return 'unknown'





def extract_calculation_data(calc_node: CalcJobNode, perform_so_n: bool = False, sanity_check_reconstruct_rho: bool = False) -> Dict[str, Any]:
    """
    Extract relevant data from a converged PW/ConstrainedPW calculation.
    
    Args:
        calc_node: AiiDA calculation node
        perform_so_n: Whether to perform SO(N) decomposition on occupation matrices
        
    Returns:
        dict: Dictionary containing extracted data with keys:
              - pk: Primary key of the calculation
              - exit_status: Exit status of the calculation
              - inputs: Dictionary of input parameters
              - output_parameters: Output parameters if available
              - output_atomic_occupations: Atomic occupations if available
              - process_type: Type of the calculation process
              - calculation_source: Tag indicating if from AFM workchain or constrained scan
              - so_n_decomposition: SO(N) analysis results (if perform_so_n=True)
    """
    try:
        data = {
            'pk': calc_node.pk,
            'exit_status': calc_node.exit_status,
            'converged': calc_node.exit_status == 0,
            'convergence_status': 'converged' if calc_node.exit_status == 0 else ('non_converged_scf' if calc_node.exit_status == 410 else 'other_error'),
            'process_type': getattr(calc_node, 'process_type', 'unknown'),
            'calculation_source': _determine_calculation_source(calc_node),
            'inputs': {},
            'output_parameters': None,
            'output_atomic_occupations': None
        }
    except Exception as e:
        print(f"Error creating basic data structure for node {calc_node.pk}: {e}")
        return None
    
    # Extract inputs with proper AiiDA API
    try:
        # Check if inputs exist and are accessible
        if hasattr(calc_node, 'inputs'):
            # Use the new API: base.links.get_incoming()
            incoming_links = calc_node.base.links.get_incoming()
            for link in incoming_links:
                try:
                    key = link.link_label
                    input_node = link.node
                    
                    # Try to get dictionary representation for Dict nodes
                    if hasattr(input_node, 'get_dict'):
                        data['inputs'][key] = input_node.get_dict()
                    # For other node types, store basic info
                    else:
                        data['inputs'][key] = {
                            'node_type': getattr(input_node, 'node_type', 'unknown'),
                            'pk': input_node.pk,
                            'uuid': str(input_node.uuid)
                        }
                except Exception as e:
                    if hasattr(link, 'link_label'):
                        data['inputs'][link.link_label] = f"Error extracting input '{link.link_label}': {str(e)}"
                    else:
                        data['inputs'][f'link_error_{len(data["inputs"])}'] = f"Error processing input link: {str(e)}"
        else:
            data['inputs'] = "Node has no inputs attribute"
                
    except Exception as e:
        data['inputs'] = f"Error accessing inputs for node {calc_node.pk}: {str(e)}"
    
    # Extract output_parameters
    try:
        if 'output_parameters' in calc_node.outputs:
            output_params = calc_node.outputs.output_parameters
            if hasattr(output_params, 'get_dict'):
                data['output_parameters'] = output_params.get_dict()
    except Exception as e:
        data['output_parameters'] = f"Error extracting output_parameters: {str(e)}"
    
    # Extract output_atomic_occupations
    try:
        if 'output_atomic_occupations' in calc_node.outputs:
            occupations = calc_node.outputs.output_atomic_occupations
            if hasattr(occupations, 'get_dict'):
                data['output_atomic_occupations'] = occupations.get_dict()
            elif hasattr(occupations, 'get_array'):
                # For ArrayData nodes
                data['output_atomic_occupations'] = {
                    name: array.tolist() for name, array in occupations.get_arraydict().items()
                }
        else:
            # Fallback: try to use unified extraction method (for new AiiDA-QE API)
            if HAS_UNIFIED_EXTRACTION:
                try:
                    unified_data = extract_occupations_from_calc(calc_node)
                    # Convert unified format to legacy format for backward compatibility
                    data['output_atomic_occupations'] = unified_data.to_legacy_dict()
                    data['unified_occupation_data'] = unified_data.as_dict()  # Also store unified format
                except Exception as unified_e:
                    data['output_atomic_occupations'] = f"Unified extraction failed: {str(unified_e)}"
    except Exception as e:
        data['output_atomic_occupations'] = f"Error extracting output_atomic_occupations: {str(e)}"
    
    # Perform SO(N) decomposition if requested and occupation matrices are available
    if perform_so_n and data['output_atomic_occupations'] and isinstance(data['output_atomic_occupations'], dict):
        try:
            # Create a temporary extractor instance to use the class method
            temp_extractor = WorkchainExtractor(perform_so_n=True, sanity_check_reconstruct_rho=sanity_check_reconstruct_rho)
            so_n_results = temp_extractor._perform_so_n_analysis(data['output_atomic_occupations'])
            data['so_n_decomposition'] = so_n_results
        except Exception as e:
            data['so_n_decomposition'] = {
                'decomposition_successful': False,
                'error_message': f"SO(N) decomposition failed: {str(e)}"
            }
    elif perform_so_n:
        data['so_n_decomposition'] = {
            'decomposition_successful': False,
            'error_message': "No valid occupation matrices available for SO(N) decomposition"
        }
    
    return data


def discover_global_workchains(group_name: str = None, max_results: int = None) -> List:
    """
    Discover GlobalConstrainedSearchWorkChain workchains as starting points.
    
    Args:
        group_name: Name of the AiiDA group to search in (optional)
        max_results: Maximum number of workchains to return (optional)
        
    Returns:
        list: List of GlobalConstrainedSearchWorkChain nodes
    """
    from aiida.orm import QueryBuilder, Group, WorkChainNode
    
    qb = QueryBuilder()
    
    if group_name:
        try:
            group = Group.get(label=group_name)
            qb.append(Group, filters={'id': group.id}, tag='group')
            qb.append(WorkChainNode, with_group='group', tag='workchain')
        except Exception as e:
            print(f"Could not find group '{group_name}': {e}")
            return []
    else:
        qb.append(WorkChainNode, tag='workchain')
    
    # Filter for GlobalConstrainedSearchWorkChain
    qb.add_filter('workchain', {
        'process_type': {'like': '%GlobalConstrainedSearchWorkChain%'}
    })
    
    qb.order_by({'workchain': {'ctime': 'desc'}})
    
    if max_results:
        qb.limit(max_results)
    
    return [result[0] for result in qb.all()]


def extract_calculations_from_global_workchain(global_wc, debug: bool = False, include_non_converged: bool = False) -> List[Tuple[int, int, str, str]]:
    """
    Extract calculations from a GlobalConstrainedSearchWorkChain with proper source tagging.
    Top-down approach: AFM workchain -> AFM calculations, Constrained workchain -> Constrained calculations.
    
    Args:
        global_wc: GlobalConstrainedSearchWorkChain node
        debug: If True, print detailed debug information
        include_non_converged: If True, also include calculations with exit status 410
        
    Returns:
        list: List of tuples (pk, exit_status, process_type, source) for PW calculations
    """
    calculations = []
    
    try:
        # Process called workchains to find AFM and constrained scans
        for called_wc in global_wc.called:
            process_type = getattr(called_wc, 'process_type', '').lower()
            
            if 'afmscan' in process_type or 'afm_scan' in process_type:
                # This is an AFM scan workchain
                afm_calculations = _extract_pw_calculations_from_workchain(called_wc, 'afm_workchain', include_non_converged=include_non_converged)
                calculations.extend(afm_calculations)
                if debug:
                    print(f"Found {len(afm_calculations)} AFM calculations from workchain {called_wc.pk}")
            
            elif 'constrainedscan' in process_type or 'constrained_scan' in process_type:
                # This is a constrained scan workchain
                constrained_calculations = _extract_pw_calculations_from_workchain(called_wc, 'constrained_scan', include_non_converged=include_non_converged)
                calculations.extend(constrained_calculations)
                if debug:
                    print(f"Found {len(constrained_calculations)} constrained calculations from workchain {called_wc.pk}")
                        
    except Exception as e:
        print(f"Error extracting calculations from global workchain {global_wc.pk}: {e}")
    
    return calculations


def _extract_pw_calculations_from_workchain(workchain, source_tag: str, visited: set = None, max_depth: int = 10, include_non_converged: bool = False) -> List[Tuple[int, int, str, str]]:
    """
    Extract PW calculations from a workchain with a specific source tag.
    
    Args:
        workchain: The workchain to extract calculations from
        source_tag: Tag to assign to calculations ('afm_workchain' or 'constrained_scan')
        visited: Set of visited node PKs to avoid cycles
        max_depth: Maximum recursion depth
        include_non_converged: If True, also include calculations with exit status 410
        
    Returns:
        list: List of tuples (pk, exit_status, process_type, source) for PW calculations
    """
    if visited is None:
        visited = set()
    
    if workchain.pk in visited or max_depth <= 0:
        return []
    
    visited.add(workchain.pk)
    calculations = []
    
    try:
        # Check all called nodes
        for called_node in workchain.called:
            if is_pw_calculation(called_node):
                # Include converged calculations (exit_status == 0)
                # Optionally include non-converged calculations (exit_status == 410)
                if called_node.exit_status == 0 or (include_non_converged and called_node.exit_status == 410):
                    calculations.append((
                        called_node.pk, 
                        called_node.exit_status, 
                        getattr(called_node, 'process_type', 'unknown'),
                        source_tag
                    ))
            else:
                # Recursively check sub-workchains
                sub_calculations = _extract_pw_calculations_from_workchain(
                    called_node, source_tag, visited, max_depth - 1, include_non_converged
                )
                calculations.extend(sub_calculations)
                
    except Exception as e:
        print(f"Error processing workchain {workchain.pk}: {e}")
    
    return calculations


def discover_pw_calculations(group_name: str = None, max_results: int = None, debug: bool = False, include_non_converged: bool = False) -> List[Tuple[int, int, str, str]]:
    """
    Discover PW calculations using a top-down approach starting from GlobalConstrainedSearchWorkChain.
    This is more efficient than bottom-up searching as it directly targets the relevant workchains.
    
    Args:
        group_name: Name of the AiiDA group to search in (optional)
        max_results: Maximum number of global workchains to process (optional)
        debug: If True, print detailed information
        include_non_converged: If True, also include calculations with exit status 410
        
    Returns:
        list: List of tuples (pk, exit_status, process_type, source) for PW/ConstrainedPW calculations
    """
    global_workchains = discover_global_workchains(group_name, max_results)
    
    if debug:
        print(f"Found {len(global_workchains)} GlobalConstrainedSearchWorkChain instances")
    
    all_calculations = []
    for i, global_wc in enumerate(global_workchains):
        if debug:
            print(f"Processing global workchain {i+1}/{len(global_workchains)}: {global_wc.pk}")
        
        calculations = extract_calculations_from_global_workchain(global_wc, debug, include_non_converged)
        all_calculations.extend(calculations)
        
        if debug:
            afm_count = sum(1 for calc in calculations if calc[3] == 'afm_workchain')
            constrained_count = sum(1 for calc in calculations if calc[3] == 'constrained_scan')
            print(f"  -> Found {afm_count} AFM calculations, {constrained_count} constrained calculations")
    
    if debug:
        print(f"Total: {len(all_calculations)} calculations from {len(global_workchains)} global workchains")
    
    return all_calculations


def discover_all_pw_calculations_for_stats(node, visited: set = None, depth: int = 0, max_depth: int = 50) -> Tuple[int, int, Dict[str, int], Dict[str, int], Dict[str, int], List[Dict[str, Any]]]:
    """
    Recursively discover ALL PW/ConstrainedPW calculations for statistics purposes.
    
    Args:
        node: Starting AiiDA node (workchain or calculation)
        visited: Set of already visited node PKs to avoid infinite loops
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent infinite recursion
        
    Returns:
        tuple: (total_calcs, converged_calcs, exit_status_counts, calc_type_counts, source_counts, non_converged_details)
    """
    if visited is None:
        visited = set()
    
    if depth > max_depth:
        return 0, 0, {}, {}, {}, []
    
    if node.pk in visited:
        return 0, 0, {}, {}, {}, []
    
    visited.add(node.pk)
    
    total_calcs = 0
    converged_calcs = 0
    exit_status_counts = {}
    calc_type_counts = {}
    source_counts = {}
    non_converged_details = []
    
    # If current node is a PW/ConstrainedPW calculation, count it
    if isinstance(node, CalcJobNode) and is_pw_calculation(node):
        process_type = getattr(node, 'process_type', 'unknown')
        exit_status = getattr(node, 'exit_status', None)
        source = _determine_calculation_source(node)
        
        total_calcs = 1
        calc_type_counts[process_type] = 1
        exit_status_counts[str(exit_status)] = 1
        source_counts[source] = 1
        
        if exit_status == 0:
            converged_calcs = 1
        else:
            non_converged_details.append({
                'pk': node.pk,
                'exit_status': exit_status,
                'process_type': process_type,
                'source': source
            })
        
        return total_calcs, converged_calcs, exit_status_counts, calc_type_counts, source_counts, non_converged_details
    
    # If it's a workchain, traverse its called links
    if hasattr(node, 'called'):
        try:
            called_nodes = node.called
            
            for called_node in called_nodes:
                child_total, child_conv, child_exit, child_types, child_sources, child_non_conv = discover_all_pw_calculations_for_stats(
                    called_node, visited, depth + 1, max_depth)
                
                total_calcs += child_total
                converged_calcs += child_conv
                non_converged_details.extend(child_non_conv)
                
                # Merge dictionaries
                for status, count in child_exit.items():
                    exit_status_counts[status] = exit_status_counts.get(status, 0) + count
                
                for calc_type, count in child_types.items():
                    calc_type_counts[calc_type] = calc_type_counts.get(calc_type, 0) + count
                
                for source, count in child_sources.items():
                    source_counts[source] = source_counts.get(source, 0) + count
                
        except Exception:
            pass  # Silently ignore errors in stats collection
    
    return total_calcs, converged_calcs, exit_status_counts, calc_type_counts, source_counts, non_converged_details


def process_calculations(calculation_list: List[Tuple[int, int, str, str]], debug: bool = False, perform_so_n: bool = False, sanity_check_reconstruct_rho: bool = False) -> List[Dict[str, Any]]:
    """
    Process a list of converged calculations and extract data.
    
    Args:
        calculation_list: List of tuples (pk, exit_status, process_type, source) - all should be converged
        debug: If True, print detailed processing information
        perform_so_n: If True, perform SO(N) decomposition on occupation matrices
        
    Returns:
        list: List of calculation data dictionaries
    """
    results = []
    
    # Create a WorkchainExtractor instance for consistent SO(N) processing
    temp_extractor = WorkchainExtractor(perform_so_n=perform_so_n, sanity_check_reconstruct_rho=sanity_check_reconstruct_rho, debug=debug)
    
    if HAS_ALIVE_BAR and len(calculation_list) > 0:
        with alive_bar(len(calculation_list), title="Processing calculations") as bar:
            for pk, exit_status, process_type, source in calculation_list:
                try:
                    calc_node = load_node(pk)
                    calc_data = temp_extractor._extract_single_calc_data(calc_node)
                    if calc_data is not None:
                        results.append(calc_data)
                        
                        if debug:
                            status_desc = "converged" if exit_status == 0 else ("non-converged SCF" if exit_status == 410 else f"error (exit {exit_status})")
                            print(f"Processed {status_desc} {process_type} calculation: PK {pk} (source: {source})")
                    else:
                        if debug:
                            print(f"Skipped calculation PK {pk} due to extraction error")
                        
                except Exception as e:
                    if debug:
                        print(f"Error processing calculation PK {pk}: {str(e)}")
                
                bar()
    else:
        # Fallback without progress bar
        # Check if we have non-converged calculations in the list
        has_non_converged = any(exit_status != 0 for _, exit_status, _, _ in calculation_list)
        calc_desc = "calculations (including non-converged)" if has_non_converged else "converged calculations"
        print(f"Processing {len(calculation_list)} {calc_desc}...")
        
        for i, (pk, exit_status, process_type, source) in enumerate(calculation_list):
            try:
                calc_node = load_node(pk)
                calc_data = temp_extractor._extract_single_calc_data(calc_node)
                if calc_data is not None:
                    results.append(calc_data)
                else:
                    print(f"Skipped calculation PK {pk} due to extraction error")
                
                if debug:
                    status_desc = "converged" if exit_status == 0 else ("non-converged SCF" if exit_status == 410 else f"error (exit {exit_status})")
                    print(f"Processed {status_desc} {process_type} calculation: PK {pk} (source: {source})")
                elif i % max(1, len(calculation_list) // 10) == 0:  # Print progress every 10%
                    progress = (i + 1) / len(calculation_list) * 100
                    print(f"Progress: {progress:.1f}% ({i + 1}/{len(calculation_list)})")
                    
            except Exception as e:
                if debug:
                    print(f"Error processing calculation PK {pk}: {str(e)}")
    
    return results
def gather_workchain_data(workchain_pk: int = None, group_name: str = None, output_filename: str = None, max_results: int = None, debug: bool = False, perform_so_n: bool = False, sanity_check_reconstruct_rho: bool = False, include_non_converged: bool = False) -> Dict[str, Any]:
    """
    Main function to gather data from workchains with intelligent workchain type detection.
    
    - If workchain_pk is a GlobalConstrainedSearchWorkChain: uses top-down approach
    - If workchain_pk is AFMScanWorkChain: extracts AFM calculations only
    - If workchain_pk is ConstrainedScanWorkChain: extracts constrained calculations only
    - If no workchain_pk: searches for GlobalConstrainedSearchWorkChain instances
    
    Args:
        workchain_pk: Primary key of a specific workchain (optional)
        group_name: Name of AiiDA group to search for GlobalConstrainedSearchWorkChain instances (optional)
        output_filename: Name of the output JSON file (optional)
        max_results: Maximum number of global workchains to process (optional)
        debug: If True, print detailed information (default: False)
        perform_so_n: If True, perform SO(N) decomposition on occupation matrices (default: False)
        sanity_check_reconstruct_rho: If True, include reconstructed density matrix and error for sanity checking (default: False)
        include_non_converged: If True, also include calculations with exit status 410 (non-converged SCF) (default: False)
        
    Returns:
        dict: Complete data dictionary containing all extracted information
        
    Raises:
        ValueError: If neither workchain_pk nor group_name is provided
        NotExistent: If the workchain node cannot be loaded
        Exception: For other errors during data extraction or file writing
    """
    try:
        extraction_method = "unknown"
        processed_workchain_info = None
        
        if workchain_pk is not None:
            # Load and identify workchain type
            root_node = load_node(workchain_pk)
            process_type = getattr(root_node, 'process_type', '').lower()
            
            print(f"Loading workchain: PK {workchain_pk}, type: {type(root_node).__name__}")
            print(f"Process type: {getattr(root_node, 'process_type', 'unknown')}")
            
            processed_workchain_info = {
                'pk': workchain_pk,
                'process_type': getattr(root_node, 'process_type', 'unknown'),
                'node_type': type(root_node).__name__
            }
            
            if 'globalconstrained' in process_type or 'global_constrained' in process_type:
                # GlobalConstrainedSearchWorkChain - use top-down approach
                print("Detected GlobalConstrainedSearchWorkChain - using top-down approach")
                extraction_method = "top_down_global_search"
                converged_calculation_list = extract_calculations_from_global_workchain(root_node, debug, include_non_converged)
                
            elif 'afmscan' in process_type or 'afm_scan' in process_type:
                # AFMScanWorkChain - extract AFM calculations only
                print("Detected AFMScanWorkChain - extracting AFM calculations")
                extraction_method = "afm_workchain_direct"
                converged_calculation_list = _extract_pw_calculations_from_workchain(root_node, 'afm_workchain', include_non_converged=include_non_converged)
                
            elif 'constrainedscan' in process_type or 'constrained_scan' in process_type:
                # ConstrainedScanWorkChain - extract constrained calculations only  
                print("Detected ConstrainedScanWorkChain - extracting constrained calculations")
                extraction_method = "constrained_workchain_direct"
                converged_calculation_list = _extract_pw_calculations_from_workchain(root_node, 'constrained_scan', include_non_converged=include_non_converged)
                
            else:
                # Unknown workchain type - try generic approach
                print(f"Unknown workchain type, trying generic extraction approach")
                extraction_method = "generic_workchain"
                converged_calculation_list = _extract_pw_calculations_from_workchain(root_node, 'unknown', include_non_converged=include_non_converged)
            
        elif group_name is not None:
            # Process all GlobalConstrainedSearchWorkChain instances from a group or globally
            print("Phase 1: Discovering GlobalConstrainedSearchWorkChain instances...")
            converged_calculation_list = discover_pw_calculations(group_name=group_name, max_results=max_results, debug=debug, include_non_converged=include_non_converged)
            
        else:
            # Process all GlobalConstrainedSearchWorkChain instances globally
            print("Phase 1: Discovering all GlobalConstrainedSearchWorkChain instances...")
            converged_calculation_list = discover_pw_calculations(max_results=max_results, debug=debug, include_non_converged=include_non_converged)
        
        if include_non_converged:
            print(f"Found {len(converged_calculation_list)} calculations (including non-converged)")
        else:
            print(f"Found {len(converged_calculation_list)} converged calculations")
        
        # Process the calculations and extract data
        calc_type_desc = "calculations (including non-converged)" if include_non_converged else "converged calculations"
        if perform_so_n:
            print(f"Phase 2: Processing {calc_type_desc} and extracting data (with SO(N) decomposition)...")
        else:
            print(f"Phase 2: Processing {calc_type_desc} and extracting data...")
        calculations_data = process_calculations(converged_calculation_list, debug=debug, perform_so_n=perform_so_n, sanity_check_reconstruct_rho=sanity_check_reconstruct_rho)
        
        # Build statistics from all calculations
        source_counts = {}
        calc_type_counts = {}
        convergence_counts = {'converged': 0, 'non_converged_scf': 0, 'other_error': 0}
        
        for pk, exit_status, process_type, source in converged_calculation_list:
            source_counts[source] = source_counts.get(source, 0) + 1
            calc_type_counts[process_type] = calc_type_counts.get(process_type, 0) + 1
            
            if exit_status == 0:
                convergence_counts['converged'] += 1
            elif exit_status == 410:
                convergence_counts['non_converged_scf'] += 1
            else:
                convergence_counts['other_error'] += 1
        
        statistics = {
            'total_calculations': len(converged_calculation_list),
            'converged_calculations': convergence_counts['converged'],
            'non_converged_scf_calculations': convergence_counts['non_converged_scf'],
            'other_error_calculations': convergence_counts['other_error'],
            'convergence_breakdown': convergence_counts,
            'calculation_types': calc_type_counts,
            'calculation_sources': source_counts
        }
        
        # Print statistics
        print("\n" + "="*50)
        print("WORKCHAIN STATISTICS (Top-down approach):")
        print("="*50)
        print(f"Total calculations processed: {statistics['total_calculations']}")
        print(f"Converged calculations: {statistics['converged_calculations']}")
        if include_non_converged:
            print(f"Non-converged SCF calculations (exit 410): {statistics['non_converged_scf_calculations']}")
            if statistics['other_error_calculations'] > 0:
                print(f"Other error calculations: {statistics['other_error_calculations']}")
        
        print("\nCalculation types:")
        for calc_type, count in statistics['calculation_types'].items():
            print(f"  {calc_type}: {count}")
        
        print("\nCalculation sources:")
        for source, count in statistics['calculation_sources'].items():
            print(f"  {source}: {count}")
        
        if include_non_converged:
            print(f"\nNote: This extraction includes both converged (exit 0) and non-converged SCF (exit 410) calculations.")
        else:
            print(f"\nNote: This extraction includes only converged calculations from GlobalConstrainedSearchWorkChain instances.")
        print("="*50)
        
        # Prepare final data structure
        metadata = {
            'total_calculations_found': len(calculations_data),
            'extraction_method': extraction_method,
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        # Add specific metadata based on extraction method
        if processed_workchain_info:
            metadata.update({
                'workchain_pk': processed_workchain_info['pk'],
                'workchain_process_type': processed_workchain_info['process_type'],
                'workchain_node_type': processed_workchain_info['node_type']
            })
        else:
            metadata.update({
                'group_name': group_name,
                'max_results': max_results,
                'global_workchains_processed': len(discover_global_workchains(group_name, max_results)) if not processed_workchain_info else None
            })
        
        output_data = {
            'metadata': metadata,
            'statistics': statistics,
            'calculations': {}
        }
        
        # Store calculations data with PK as key
        for calc_data in calculations_data:
            pk = calc_data['pk']
            output_data['calculations'][str(pk)] = calc_data
        
        # Save to JSON file if filename provided
        if output_filename:
            output_path = Path(output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Data saved to: {output_path.absolute()}")
        
        print(f"\nData extraction completed successfully!")
        print(f"Total calculations processed: {statistics['total_calculations']}")
        print(f"Converged calculations: {statistics['converged_calculations']}")
        if include_non_converged and statistics['non_converged_scf_calculations'] > 0:
            print(f"Non-converged SCF calculations: {statistics['non_converged_scf_calculations']}")
        print(f"AFM workchain calculations: {statistics['calculation_sources'].get('afm_workchain', 0)}")
        print(f"Constrained scan calculations: {statistics['calculation_sources'].get('constrained_scan', 0)}")
        
        return output_data
        
    except NotExistent:
        if workchain_pk:
            raise NotExistent(f"Could not load node with PK {workchain_pk}. Please check that the PK is correct.")
        else:
            raise NotExistent("Could not find the specified group or workchain nodes.")
    
    except Exception as e:
        print(f"Error during data extraction: {str(e)}")
        raise


def filter_calculations_by_source(data: Dict[str, Any], source_filter: str) -> Dict[str, Any]:
    """
    Filter calculations from gathered data by their source.
    
    Args:
        data: Data dictionary from gather_workchain_data
        source_filter: Source to filter by ('afm_workchain', 'constrained_scan', or 'unknown')
        
    Returns:
        dict: Filtered calculations dictionary with same structure but only matching calculations
    """
    filtered_calculations = {}
    
    for pk, calc_data in data['calculations'].items():
        if calc_data.get('calculation_source') == source_filter:
            filtered_calculations[pk] = calc_data
    
    # Create filtered data structure
    filtered_data = {
        'metadata': data['metadata'].copy(),
        'statistics': data['statistics'].copy(),
        'calculations': filtered_calculations
    }
    
    # Update metadata to reflect filtering
    filtered_data['metadata']['total_calculations_found'] = len(filtered_calculations)
    filtered_data['metadata']['filtered_by_source'] = source_filter
    
    return filtered_data


def get_calculation_sources_summary(data: Dict[str, Any]) -> Dict[str, int]:
    """
    Get a summary of calculation sources from gathered data.
    
    Args:
        data: Data dictionary from gather_workchain_data
        
    Returns:
        dict: Dictionary with source names as keys and counts as values
    """
    source_counts = {}
    
    for calc_data in data['calculations'].values():
        source = calc_data.get('calculation_source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    return source_counts


