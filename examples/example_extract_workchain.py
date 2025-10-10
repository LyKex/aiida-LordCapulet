#%%
import json
import aiida
from lordcapulet.utils.postprocessing.gather_workchain_data import gather_workchain_data
# aiida profile load
aiida.load_profile()


def extract_workchain_to_json(workchain_pk, output_file=None, perform_so_n=False, sanity_check_reconstruct_rho=False):
    """
    Extract all calculations from a workchain and save to JSON.
    
    Args:
        workchain_pk (int): Primary key of the workchain to process
        output_file (str, optional): Output JSON filename. If None, uses workchain_pk.json
        perform_so_n (bool): Whether to perform SO(N) decomposition on occupation matrices
        sanity_check_reconstruct_rho (bool): Whether to include reconstructed density matrix and error for sanity checking
        
    Returns:
        dict: Extracted data with calculations, metadata, and statistics
    """
    if output_file is None:
        output_file = f"workchain_{workchain_pk}_data.json"
    
    if perform_so_n:
        print(f"Extracting calculations from workchain {workchain_pk} with SO(N) decomposition...")
    else:
        print(f"Extracting calculations from workchain {workchain_pk}...")
    
    # Extract data from the workchain
    data = gather_workchain_data(workchain_pk=workchain_pk, output_filename=output_file, debug=True, perform_so_n=perform_so_n, sanity_check_reconstruct_rho=sanity_check_reconstruct_rho)
    
    # Print summary
    print(f"\nExtraction complete!")
    
    # Debug: print available metadata keys
    print(f"Available metadata keys: {list(data['metadata'].keys())}")
    
    # Print workchain info if available
    if 'workchain_process_type' in data['metadata']:
        print(f"Workchain type: {data['metadata']['workchain_process_type']}")
    else:
        print(f"Workchain process type not available in metadata")
    
    print(f"Total calculations found: {data['metadata']['total_calculations_found']}")
    print(f"Extraction method: {data['metadata']['extraction_method']}")
    
    print(f"Source breakdown:")
    for source, count in data['statistics']['calculation_sources'].items():
        print(f"  {source}: {count}")
    print(f"Data saved to: {output_file}")
    
    return data
#%%

    # workchain_pk = 19232 # FeO
    # workchain_pk = 24930 # CoO
    # workchain_pk = 25730 # CuO
    # workchain_pk = 36461 # VO
    # workchain_pk = 59855 # UO2
# data = extract_workchain_to_json(49395)
# data = extract_workchain_to_json(24930, output_file = "CoO_scan_data.json")
# data = extract_workchain_to_json(19232, output_file = "FeO_scan_data.json")

global_chain = 64716 # UO2
afm_chain = 64717

feo_so_n_global_chain = 71770

data = extract_workchain_to_json(feo_so_n_global_chain, output_file = "FeO_so_n_trial_scan_data.json", perform_so_n=True, sanity_check_reconstruct_rho=True)

#%%
# Simple debug for PK 64856 - atom 1 SO(N) decomposition
import numpy as np
from lordcapulet.utils.so_n_decomposition import get_so_n_lie_basis, rotation_to_euler_angles
from aiida.orm import load_node
import aiida
from scipy.linalg import expm, logm
aiida.load_profile()

# calc_node = load_node(64856)

# calc_node = load_node(65688)
calc_node = load_node(65087)
# calc_node = load_node(64793)
occupations = calc_node.outputs.output_atomic_occupations.get_dict()

# Check only a single atom 
iatom = 2
atom_data = occupations[str(iatom)]
spin_data = atom_data['spin_data']

# Focus on up spin only
spin = 'up'
print(f"Atom {iatom}, {spin} spin:")
occ_matrix = np.array(spin_data[spin]['occupation_matrix'])
eigenvals, eigenvecs = np.linalg.eigh(occ_matrix)
det = np.linalg.det(eigenvecs)

print(f"  det = {det:.6f}")

# Check eigenvalues of the rotation matrix (eigenvecs)
rotation_eigenvals = np.linalg.eigvals(eigenvecs)
print(f"  rotation matrix eigenvalues: {rotation_eigenvals}")
dim = occ_matrix.shape[0]
generators = get_so_n_lie_basis(dim)
euler_angles, has_reflection, reg = rotation_to_euler_angles(eigenvecs, generators, check_orthogonal=False)
#%%
# Check if any eigenvalues are close to -1 (branch cut issue)
close_to_minus_one = np.abs(rotation_eigenvals + 1.0) < 1e-10
if np.any(close_to_minus_one):
    print(f"  WARNING: Found eigenvalues close to -1: {rotation_eigenvals[close_to_minus_one]}")

# SO(N) decomposition
dim = occ_matrix.shape[0]
generators = get_so_n_lie_basis(dim)
euler_angles, has_reflection, reg = rotation_to_euler_angles(eigenvecs, generators, check_orthogonal=False)



#%%
print(f"  has_reflection = {has_reflection}")
print(f"  euler_angles range: [{np.min(euler_angles):.3f}, {np.max(euler_angles):.3f}]")

# Verify reconstruction: rebuild the rotation matrix from Euler angles
from lordcapulet.utils.so_n_decomposition import euler_angles_to_rotation
R_reconstructed = euler_angles_to_rotation(euler_angles, generators, reflection=has_reflection)

# Check if we get back the original eigenvectors
reconstruction_error = np.max(np.abs(R_reconstructed - eigenvecs))
print(f"  Reconstruction error: {reconstruction_error:.2e}")

# Rebuild the density matrix: rho = R * diag(eigenvals) * R^T
rho_reconstructed = R_reconstructed @ np.diag(eigenvals) @ R_reconstructed.T
density_error = np.max(np.abs(rho_reconstructed - occ_matrix))
print(f"  Density matrix error: {density_error:.2e}")

if reconstruction_error < 1e-6 and density_error < 1e-6:
    print("  ✓ Reconstruction successful!")
    print(f"  Original eigenvals: {eigenvals}")
    print(f"  Original matrix trace: {np.trace(occ_matrix):.6f}")
    print(f"  Reconstructed trace: {np.trace(rho_reconstructed):.6f}")
else:
    print("  ✗ Reconstruction failed!")
    print(f"  Original eigenvals: {eigenvals}")
    print(f"  Original matrix trace: {np.trace(occ_matrix):.6f}")
    print(f"  Reconstructed trace: {np.trace(rho_reconstructed):.6f}")

# %%
