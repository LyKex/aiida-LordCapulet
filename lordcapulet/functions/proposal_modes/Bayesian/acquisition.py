"""
Custom acquisition functions with physics-based constraints.

This module provides acquisition functions that incorporate physical
constraints on occupation matrices, such as:
- Trace constraints (target electron counts)
- Eigenvalue constraints (positive semi-definite matrices)
- Minor determinant constraints (principal minor conditions)
"""

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform


# def compute_eigenvalue_preference(matrix, k=5.0):
#     """
#     Computes a preference score (0-1) for a matrix using a smooth
#     sigmoid-based penalty on eigenvalues.
    
#     This enforces that all eigenvalues should be in the range [0, 1],
#     which is a physical constraint for occupation matrices.
    
#     Args:
#         matrix: The input matrix (must be real symmetric or Hermitian)
#         k: A "stiffness" parameter. Higher k = steeper penalty (default: 5.0)
           
#     Returns:
#         Preference score from 0 to 1 (1 = all eigenvalues in [0,1])
#     """
#     try:
#         # Use .eigh() for real symmetric/Hermitian matrices
#         eigenvalues = torch.linalg.eigh(matrix).eigenvalues
#     except torch.linalg.LinAlgError:
#         # If the decomposition fails (e.g., NaNs), this is a bad point
#         return torch.tensor(0.0, device=matrix.device, dtype=matrix.dtype)

#     # Penalty for eigenvalues < 0 (pushes eig > 0)
#     pref_low = torch.sigmoid(k * eigenvalues)

#     # Penalty for eigenvalues > 1 (pushes eig < 1)
#     pref_high = torch.sigmoid(k * (1.0 - eigenvalues))

#     # Total score is the product of all individual scores
#     # This ensures all eigenvalues must be in the [0, 1] range
#     total_pref = torch.prod(pref_low) * torch.prod(pref_high) 
#     return total_pref


def compute_minor_preference_offdiag_only(matrix, k=20.0):
    """
    Computes a preference score (0-1) for a matrix based *only* on its
    2x2 principal minors, using a smooth sigmoid penalty.
    
    This enforces: p_ii*p_jj - p_ij^2 >= 0 (for all i < j)
    
    It assumes the diagonal (p_ii) is already handled by the optimizer's 
    bounds (e.g., [0, 1]).
    
    Args:
        matrix: The input matrix (assumed symmetric)
        k: A "stiffness" parameter. Higher k = steeper penalty (default: 20.0)
        
    Returns:
        Preference score from 0 to 1
    """
    # Get the diagonal elements, p_ii
    diag = torch.diag(matrix)
    
    # Create a matrix of p_ii * p_jj products
    diag_outer = torch.outer(diag, diag)
    
    # Create a matrix of p_ij^2
    p_ij_squared = matrix**2
    
    # violation_matrix[i, j] = p_ii*p_jj - p_ij^2
    violation_matrix = diag_outer - p_ij_squared
    
    # We only care about the upper triangle (where i < j)
    n = matrix.shape[0]
    triu_indices = torch.triu_indices(n, n, offset=1, device=matrix.device)
    
    # Get the violations for just the i < j pairs
    off_diag_violations = violation_matrix[triu_indices[0], triu_indices[1]]
    
    # Calculate preference for all off-diagonal pairs
    pref_off_diag = torch.sigmoid(k * off_diag_violations)
    
    # Combine scores. All must be good.
    total_pref_off_diag = torch.prod(pref_off_diag)
    
    return total_pref_off_diag


def compute_trace_preference(trace_val, target, sigma):
    """
    Helper function to compute the Gaussian preference for a single trace.
    
    This creates a soft constraint that the trace should be near the target value.
    
    Args:
        trace_val: The computed trace value
        target: The desired trace value
        sigma: The width of the Gaussian (smaller = tighter constraint)
        
    Returns:
        Preference score from 0 to 1 (1 = trace exactly at target)
    """
    return torch.exp(-((trace_val - target)**2) / (2 * sigma**2))


def compute_total_preference(X_batch, databank, atom_ids, trace_target, trace_sigma, eig_k):
    """
    Calculates a combined preference score (0-1) for a BATCH of X vectors,
    considering BOTH trace and eigenvalue/minor constraints.
    
    Args:
        X_batch: Batch of flattened occupation matrices [batch_size, num_features]
        databank: DataBank instance
        atom_ids: List of atom IDs to constrain
        trace_target: Target trace value (electron count)
        trace_sigma: Width of trace preference Gaussian
        eig_k: Stiffness parameter for eigenvalue/minor preference
        
    Returns:
        Tensor of preference scores, one per batch element
    """
    import torch
    scores = []
    
    for x in X_batch:
        total_pref_trace = torch.tensor(1.0, device=X_batch.device, dtype=X_batch.dtype)
        total_pref_eig = torch.tensor(1.0, device=X_batch.device, dtype=X_batch.dtype)
        
        try:
            # Unflatten to get the matrices for this 'x'
            occ_data = databank.from_pytorch(x.unsqueeze(0), atom_ids=atom_ids, spins=['up', 'down'])[0]
            
            for atom_id in atom_ids:
                # Get matrices as numpy arrays, then convert to torch
                up_matrix = torch.tensor(occ_data.get_occupation_matrix(atom_id, 'up'), 
                                        device=X_batch.device, dtype=X_batch.dtype)
                down_matrix = torch.tensor(occ_data.get_occupation_matrix(atom_id, 'down'), 
                                          device=X_batch.device, dtype=X_batch.dtype)
                
                # 1. Trace Preference
                trace = torch.trace(up_matrix) + torch.trace(down_matrix)
                pref_trace = compute_trace_preference(trace, trace_target, trace_sigma)
                total_pref_trace *= pref_trace
                
                # 2. Eigenvalue/Minor Preference
                pref_eig_up = compute_minor_preference_offdiag_only(up_matrix, k=eig_k)
                pref_eig_down = compute_minor_preference_offdiag_only(down_matrix, k=eig_k)
                total_pref_eig *= pref_eig_up * pref_eig_down
            
            # 3. Final Combined Score for this 'x'
            final_score = total_pref_trace * total_pref_eig
            scores.append(final_score)
            
        except Exception:
            # If unflattening or computation fails, it's a bad point
            scores.append(torch.tensor(0.0, device=X_batch.device, dtype=X_batch.dtype))
        
    return torch.stack(scores)


class AnalyticCustomPreference(AnalyticAcquisitionFunction):
    """
    Multiplies a base acquisition function (e.g., LCB) by a custom
    preference score (from 0 to 1) calculated from X.
    
    This allows incorporating physics constraints into the acquisition
    function, guiding the optimizer toward physically valid regions.
    
    Args:
        model: The GP model
        base_acqf: The base acquisition function (e.g., UpperConfidenceBound)
        compute_preference_func: Function that computes preference scores
                                 Should accept X_batch and return scores
    """
    
    def __init__(self, model, base_acqf, compute_preference_func):
        super().__init__(model=model)
        self.base_acqf = base_acqf
        self.compute_pref = compute_preference_func

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        """
        Compute the constrained acquisition function value.
        
        Args:
            X: Input tensor [batch_size, 1, num_features]
            
        Returns:
            Acquisition values multiplied by preference scores
        """
        # X has shape [batch_size, 1, num_features]
        X_squeezed = X.squeeze(-2)  # Shape [batch_size, num_features]
        
        # 1. Get energy score from base acquisition function
        energy_score = self.base_acqf(X)
        
        # 2. Get preference score (a value from 0 to 1)
        pref_score = self.compute_pref(X_squeezed)
        
        # 3. Combine: The preference score "gates" the energy score
        return energy_score * pref_score
