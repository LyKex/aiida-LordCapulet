#!/usr/bin/env python3
"""
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
    rotation_to_euler_angles,
    decompose_rho_and_fix_gauge
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


def perform_so_n_decomposition(occupation_matrices: Dict[str, Any], debug: bool = False, sanity_check_reconstruct_rho: bool = False) -> Dict[str, Any]:
    """
    Perform SO(N) decomposition on occupation matrices to extract eigenvalues, 
    Euler angles, and reflection information.
    
    Args:
        occupation_matrices: Dictionary containing occupation matrices from AiiDA output
        debug: Whether to print debug information
        sanity_check_reconstruct_rho: Whether to include reconstructed density matrix and reconstruction error
        
    Returns:
        dict: Dictionary containing SO(N) decomposition results with keys:
              - atom_decompositions: Dict with atom keys containing:
                - up_spin: {eigenvalues, euler_angles, has_reflection}
                - down_spin: {eigenvalues, euler_angles, has_reflection}
              - decomposition_successful: bool indicating if decomposition was successful
              - error_message: str with error details if decomposition failed
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
                        if debug:
                            print(f"Skipping non-square matrix for atom {atom_key}, spin {spin}")
                        continue
                    
                    dim = occ_matrix.shape[0]
                    
                    # Generate SO(N) basis
                    generators = get_so_n_lie_basis(dim)
                    
                    # Diagonalize to get eigenvalues and eigenvectors
                    eigenvalues, eigenvectors = np.linalg.eigh(occ_matrix)
                    
                    # Check if eigenvectors form an orthogonal matrix
                    if not np.allclose(eigenvectors @ eigenvectors.T, np.eye(dim), atol=1e-10):
                        if debug:
                            print(f"Warning: eigenvectors not orthogonal for atom {atom_key}, spin {spin}")
                    
                    # Extract Euler angles and reflection information
                    euler_angles, has_reflection, need_regularization = rotation_to_euler_angles(
                        eigenvectors, generators, check_orthogonal=True
                    )
                    
                    # Handle regularization if needed
                    regularization_applied = False
                    if need_regularization:
                        if debug:
                            print(f"Regularization needed for atom {atom_key}, {spin} spin - applying 1e-7 to density matrix")
                        
                        # Apply regularization to density matrix and try again
                        regularized_occ_matrix = occ_matrix + 1e-7
                        eigenvalues_reg, eigenvectors_reg = np.linalg.eigh(regularized_occ_matrix)
                        
                        # Try SO(N) decomposition again with regularized matrix
                        euler_angles_reg, has_reflection_reg, need_regularization_reg = rotation_to_euler_angles(
                            eigenvectors_reg, generators, check_orthogonal=False
                        )
                        
                        if not need_regularization_reg:
                            # Regularization worked - use regularized results
                            eigenvalues = eigenvalues_reg
                            eigenvectors = eigenvectors_reg
                            euler_angles = euler_angles_reg
                            has_reflection = has_reflection_reg
                            regularization_applied = True
                            if debug:
                                print(f"Regularization successful for atom {atom_key}, {spin} spin")
                        else:
                            # Regularization didn't help - use original results but flag the issue
                            regularization_applied = False
                            if debug:
                                print(f"Regularization failed for atom {atom_key}, {spin} spin - using original results")
                    
                    # Store results
                    result_data = {
                        'eigenvalues': eigenvalues.tolist(),
                        'euler_angles': euler_angles.tolist(),
                        'has_reflection': bool(has_reflection),
                        'matrix_dimension': dim,
                        'trace': float(np.trace(occ_matrix)),
                        'need_regularization': bool(need_regularization),
                        'regularization_applied': bool(regularization_applied)
                    }
                    
                    # Add sanity check reconstruction if requested
                    if sanity_check_reconstruct_rho:
                        from lordcapulet.utils.so_n_decomposition import euler_angles_to_rotation
                        
                        # Reconstruct the rotation matrix from Euler angles
                        R_reconstructed = euler_angles_to_rotation(euler_angles, generators, reflection=has_reflection)
                        
                        # Reconstruct the density matrix: rho = R * diag(eigenvals) * R^T
                        rho_reconstructed = R_reconstructed @ np.diag(eigenvalues) @ R_reconstructed.T
                        
                        # Calculate reconstruction error
                        reconstruction_error = float(np.max(np.abs(rho_reconstructed - occ_matrix)))
                        
                        # Add to results
                        result_data['sanity_check'] = {
                            'reconstructed_density_matrix': rho_reconstructed.tolist(),
                            'reconstruction_error': reconstruction_error
                        }
                        
                        if debug:
                            print(f"  Reconstruction error: {reconstruction_error:.2e}")
                    
                    atom_decomp[f'{spin}_spin'] = result_data
                    
                    if debug:
                        print(f"SO(N) decomposition successful for atom {atom_key}, {spin} spin:")
                        print(f"  Eigenvalues: {eigenvalues}")
                        print(f"  Has reflection: {has_reflection}")
                        print(f"  Matrix trace: {np.trace(occ_matrix):.6f}")
                
                except Exception as e:
                    if debug:
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
        if debug:
            print(f"SO(N) decomposition error: {e}")
    
    return decomposition_results


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
    except Exception as e:
        data['output_atomic_occupations'] = f"Error extracting output_atomic_occupations: {str(e)}"
    
    # Perform SO(N) decomposition if requested and occupation matrices are available
    if perform_so_n and data['output_atomic_occupations'] and isinstance(data['output_atomic_occupations'], dict):
        try:
            so_n_results = perform_so_n_decomposition(data['output_atomic_occupations'], sanity_check_reconstruct_rho=sanity_check_reconstruct_rho)
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


def extract_calculations_from_global_workchain(global_wc, debug: bool = False) -> List[Tuple[int, int, str, str]]:
    """
    Extract calculations from a GlobalConstrainedSearchWorkChain with proper source tagging.
    Top-down approach: AFM workchain -> AFM calculations, Constrained workchain -> Constrained calculations.
    
    Args:
        global_wc: GlobalConstrainedSearchWorkChain node
        
    Returns:
        list: List of tuples (pk, exit_status, process_type, source) for converged calculations
    """
    calculations = []
    
    try:
        # Process called workchains to find AFM and constrained scans
        for called_wc in global_wc.called:
            process_type = getattr(called_wc, 'process_type', '').lower()
            
            if 'afmscan' in process_type or 'afm_scan' in process_type:
                # This is an AFM scan workchain
                afm_calculations = _extract_pw_calculations_from_workchain(called_wc, 'afm_workchain')
                calculations.extend(afm_calculations)
                if debug:
                    print(f"Found {len(afm_calculations)} AFM calculations from workchain {called_wc.pk}")
            
            elif 'constrainedscan' in process_type or 'constrained_scan' in process_type:
                # This is a constrained scan workchain
                constrained_calculations = _extract_pw_calculations_from_workchain(called_wc, 'constrained_scan')
                calculations.extend(constrained_calculations)
                if debug:
                    print(f"Found {len(constrained_calculations)} constrained calculations from workchain {called_wc.pk}")
                        
    except Exception as e:
        print(f"Error extracting calculations from global workchain {global_wc.pk}: {e}")
    
    return calculations


def _extract_pw_calculations_from_workchain(workchain, source_tag: str, visited: set = None, max_depth: int = 10) -> List[Tuple[int, int, str, str]]:
    """
    Extract all converged PW calculations from a workchain with a specific source tag.
    
    Args:
        workchain: The workchain to extract calculations from
        source_tag: Tag to assign to calculations ('afm_workchain' or 'constrained_scan')
        visited: Set of visited node PKs to avoid cycles
        max_depth: Maximum recursion depth
        
    Returns:
        list: List of tuples (pk, exit_status, process_type, source) for converged calculations
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
            if is_pw_calculation(called_node) and called_node.exit_status == 0:
                calculations.append((
                    called_node.pk, 
                    called_node.exit_status, 
                    getattr(called_node, 'process_type', 'unknown'),
                    source_tag
                ))
            else:
                # Recursively check sub-workchains
                sub_calculations = _extract_pw_calculations_from_workchain(
                    called_node, source_tag, visited, max_depth - 1
                )
                calculations.extend(sub_calculations)
                
    except Exception as e:
        print(f"Error processing workchain {workchain.pk}: {e}")
    
    return calculations


def discover_pw_calculations(group_name: str = None, max_results: int = None, debug: bool = False) -> List[Tuple[int, int, str, str]]:
    """
    Discover PW calculations using a top-down approach starting from GlobalConstrainedSearchWorkChain.
    This is more efficient than bottom-up searching as it directly targets the relevant workchains.
    
    Args:
        group_name: Name of the AiiDA group to search in (optional)
        max_results: Maximum number of global workchains to process (optional)
        debug: If True, print detailed information
        
    Returns:
        list: List of tuples (pk, exit_status, process_type, source) for converged PW/ConstrainedPW calculations
    """
    global_workchains = discover_global_workchains(group_name, max_results)
    
    if debug:
        print(f"Found {len(global_workchains)} GlobalConstrainedSearchWorkChain instances")
    
    all_calculations = []
    for i, global_wc in enumerate(global_workchains):
        if debug:
            print(f"Processing global workchain {i+1}/{len(global_workchains)}: {global_wc.pk}")
        
        calculations = extract_calculations_from_global_workchain(global_wc, debug)
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
    
    if HAS_ALIVE_BAR and len(calculation_list) > 0:
        with alive_bar(len(calculation_list), title="Processing calculations") as bar:
            for pk, exit_status, process_type, source in calculation_list:
                try:
                    calc_node = load_node(pk)
                    calc_data = extract_calculation_data(calc_node, perform_so_n=perform_so_n, sanity_check_reconstruct_rho=sanity_check_reconstruct_rho)
                    if calc_data is not None:
                        results.append(calc_data)
                        
                        if debug:
                            print(f"Processed converged {process_type} calculation: PK {pk} (source: {source})")
                    else:
                        if debug:
                            print(f"Skipped calculation PK {pk} due to extraction error")
                        
                except Exception as e:
                    if debug:
                        print(f"Error processing calculation PK {pk}: {str(e)}")
                
                bar()
    else:
        # Fallback without progress bar
        print(f"Processing {len(calculation_list)} converged calculations...")
        for i, (pk, exit_status, process_type, source) in enumerate(calculation_list):
            try:
                calc_node = load_node(pk)
                calc_data = extract_calculation_data(calc_node, perform_so_n=perform_so_n, sanity_check_reconstruct_rho=sanity_check_reconstruct_rho)
                if calc_data is not None:
                    results.append(calc_data)
                else:
                    print(f"Skipped calculation PK {pk} due to extraction error")
                
                if debug:
                    print(f"Processed converged {process_type} calculation: PK {pk} (source: {source})")
                elif i % max(1, len(calculation_list) // 10) == 0:  # Print progress every 10%
                    progress = (i + 1) / len(calculation_list) * 100
                    print(f"Progress: {progress:.1f}% ({i + 1}/{len(calculation_list)})")
                    
            except Exception as e:
                if debug:
                    print(f"Error processing calculation PK {pk}: {str(e)}")
    
    return results
def gather_workchain_data(workchain_pk: int = None, group_name: str = None, output_filename: str = None, max_results: int = None, debug: bool = False, perform_so_n: bool = False, sanity_check_reconstruct_rho: bool = False) -> Dict[str, Any]:
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
                converged_calculation_list = extract_calculations_from_global_workchain(root_node, debug)
                
            elif 'afmscan' in process_type or 'afm_scan' in process_type:
                # AFMScanWorkChain - extract AFM calculations only
                print("Detected AFMScanWorkChain - extracting AFM calculations")
                extraction_method = "afm_workchain_direct"
                converged_calculation_list = _extract_pw_calculations_from_workchain(root_node, 'afm_workchain')
                
            elif 'constrainedscan' in process_type or 'constrained_scan' in process_type:
                # ConstrainedScanWorkChain - extract constrained calculations only  
                print("Detected ConstrainedScanWorkChain - extracting constrained calculations")
                extraction_method = "constrained_workchain_direct"
                converged_calculation_list = _extract_pw_calculations_from_workchain(root_node, 'constrained_scan')
                
            else:
                # Unknown workchain type - try generic approach
                print(f"Unknown workchain type, trying generic extraction approach")
                extraction_method = "generic_workchain"
                converged_calculation_list = _extract_pw_calculations_from_workchain(root_node, 'unknown')
            
        elif group_name is not None:
            # Process all GlobalConstrainedSearchWorkChain instances from a group or globally
            print("Phase 1: Discovering GlobalConstrainedSearchWorkChain instances...")
            converged_calculation_list = discover_pw_calculations(group_name=group_name, max_results=max_results, debug=debug)
            
        else:
            # Process all GlobalConstrainedSearchWorkChain instances globally
            print("Phase 1: Discovering all GlobalConstrainedSearchWorkChain instances...")
            converged_calculation_list = discover_pw_calculations(max_results=max_results, debug=debug)
        
        print(f"Found {len(converged_calculation_list)} converged calculations")
        
        # Process the converged calculations and extract data
        if perform_so_n:
            print("Phase 2: Processing converged calculations and extracting data (with SO(N) decomposition)...")
        else:
            print("Phase 2: Processing converged calculations and extracting data...")
        calculations_data = process_calculations(converged_calculation_list, debug=debug, perform_so_n=perform_so_n, sanity_check_reconstruct_rho=sanity_check_reconstruct_rho)
        
        # Build statistics from the converged calculations
        source_counts = {}
        calc_type_counts = {}
        
        for pk, exit_status, process_type, source in converged_calculation_list:
            source_counts[source] = source_counts.get(source, 0) + 1
            calc_type_counts[process_type] = calc_type_counts.get(process_type, 0) + 1
        
        total_converged = len(converged_calculation_list)
        
        statistics = {
            'converged_calculations': total_converged,
            'calculation_types': calc_type_counts,
            'calculation_sources': source_counts
        }
        
        # Print statistics
        print("\n" + "="*50)
        print("WORKCHAIN STATISTICS (Top-down approach):")
        print("="*50)
        print(f"Converged calculations processed: {statistics['converged_calculations']}")
        
        print("\nCalculation types:")
        for calc_type, count in statistics['calculation_types'].items():
            print(f"  {calc_type}: {count}")
        
        print("\nCalculation sources:")
        for source, count in statistics['calculation_sources'].items():
            print(f"  {source}: {count}")
        
        print("\nNote: This top-down approach processes only converged calculations from GlobalConstrainedSearchWorkChain instances.")
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
        print(f"Total converged calculations processed: {statistics['converged_calculations']}")
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract data from converged PW/ConstrainedPW calculations in a workchain"
    )
    parser.add_argument("pk", type=int, help="Primary key of the root workchain")
    parser.add_argument("output", help="Output JSON filename")
    parser.add_argument("--max-depth", type=int, default=50, 
                       help="Maximum recursion depth (default: 50)")
    parser.add_argument("--debug", action="store_true", 
                       help="Print detailed traversal information")
    parser.add_argument("--filter-source", choices=['afm_workchain', 'constrained_scan', 'unknown'],
                       help="Filter calculations by source type")
    parser.add_argument("--so-n", action="store_true",
                       help="Perform SO(N) decomposition on occupation matrices")
    
    args = parser.parse_args()
    
    try:
        data = gather_workchain_data(args.pk, args.output, debug=args.debug, perform_so_n=args.so_n)
        
        # If filtering requested, create filtered file
        if args.filter_source:
            filtered_data = filter_calculations_by_source(data, args.filter_source)
            filtered_filename = args.output.replace('.json', f'_filtered_{args.filter_source}.json')
            
            with open(filtered_filename, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)
            
            print(f"\nFiltered data ({args.filter_source}) saved to: {filtered_filename}")
            print(f"Filtered calculations count: {len(filtered_data['calculations'])}")
            
        # Print source summary
        source_summary = get_calculation_sources_summary(data)
        print(f"\nCalculation sources summary:")
        for source, count in source_summary.items():
            print(f"  {source}: {count}")
            
    except Exception as e:
        print(f"Failed to extract data: {e}")
        exit(1)
