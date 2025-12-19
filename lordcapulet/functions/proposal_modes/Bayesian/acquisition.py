"""
Custom acquisition functions with physics-based constraints.

This module provides acquisition functions that incorporate physical
constraints on occupation matrices, such as:
- Trace constraints (target electron counts)
- Principal minor constraints (positive semi-definite condition via 2x2 minors)
"""

import torch
import numpy as np
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform


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


def compute_total_preference(X_batch, databank, atom_ids, trace_target, trace_sigma, 
                           use_minor_preference=False, eig_k=20.0):
    """
    Calculates a combined preference score (0-1) for a BATCH of X vectors,
    considering BOTH trace and principal minor constraints.
    
    Args:
        X_batch: Batch of flattened occupation matrices [batch_size, num_features]
        databank: DataBank instance
        atom_ids: List of atom IDs to constrain
        trace_target: Target trace value (electron count)
        trace_sigma: Width of trace preference Gaussian
        use_minor_preference: If True, apply principal minor constraint
        eig_k: Stiffness parameter for principal minor preference (default: 20.0)
        
    Returns:
        Tensor of preference scores, one per batch element
    """
    scores = []
    
    for x in X_batch:
        try:
            # Unflatten to get the matrices for this 'x'
            occ_data = databank.from_pytorch(x.unsqueeze(0), atom_ids=atom_ids, spins=['up', 'down'])[0]
            
            total_pref_trace = torch.tensor(1.0, device=X_batch.device, dtype=X_batch.dtype)
            total_pref_minor = torch.tensor(1.0, device=X_batch.device, dtype=X_batch.dtype)
            
            for atom_id in atom_ids:
                # Get matrices as numpy arrays, then convert to torch
                up_matrix_np = occ_data.get_occupation_matrix(atom_id, 'up')
                down_matrix_np = occ_data.get_occupation_matrix(atom_id, 'down')
                
                up_matrix = torch.tensor(up_matrix_np, device=X_batch.device, dtype=X_batch.dtype)
                down_matrix = torch.tensor(down_matrix_np, device=X_batch.device, dtype=X_batch.dtype)
                
                # 1. Trace Preference
                trace = torch.trace(up_matrix) + torch.trace(down_matrix)
                pref_trace = compute_trace_preference(trace, trace_target, trace_sigma)
                total_pref_trace *= pref_trace
                
                # 2. Principal Minor Preference (only if explicitly enabled)
                if use_minor_preference:
                    pref_minor_up = compute_minor_preference_offdiag_only(up_matrix, k=eig_k)
                    pref_minor_down = compute_minor_preference_offdiag_only(down_matrix, k=eig_k)
                    total_pref_minor *= pref_minor_up * pref_minor_down
            
            # 3. Final Combined Score for this 'x'
            final_score = total_pref_trace * total_pref_minor
            scores.append(final_score)
            
        except Exception as e:
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
            X: Input tensor [batch_size, q, num_features] or [batch_size, 1, num_features]
            
        Returns:
            Acquisition values multiplied by preference scores
        """
        # X has shape [batch_size, q, num_features] where q is number of candidates
        # For batch optimization (q>1), we need to handle each candidate
        
        # 1. Get energy score from base acquisition function
        energy_score = self.base_acqf(X)
        
        # 2. Get preference scores for each candidate in the batch
        # Reshape X to [batch_size * q, num_features] for preference computation
        batch_size, q, num_features = X.shape
        X_flat = X.reshape(-1, num_features)  # [batch_size * q, num_features]
        
        pref_scores_flat = self.compute_pref(X_flat)  # [batch_size * q]
        
        # For q>1 batch optimization, we need the minimum preference across all q candidates
        # because a batch is only as good as its worst element
        pref_scores = pref_scores_flat.reshape(batch_size, q)  # [batch_size, q]
        pref_score = pref_scores.min(dim=-1, keepdim=True)[0]  # [batch_size, 1]
        
        # 3. Combine: The preference score "gates" the energy score
        return energy_score * pref_score.squeeze(-1)
