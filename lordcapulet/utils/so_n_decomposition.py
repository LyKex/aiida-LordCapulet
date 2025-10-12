#%%
import numpy as np
from scipy.linalg import expm, logm, qr

# So for a collinear calculation the density matrix of one atomic shell
# is a norb x norb matrix, which is assumed to be symmetric and positive definite

# one can decompose an orthogonal matrix into a diagonal part and an orthogonal part
# M =  R  V R^T where V is the matrix of the eigenvalues and it is unique up to permutation
# and R is some rotation matrix in 5 dimensional space.

# The space of all orthogonal matrices is O(norb) and can be represented from its
# exponential map from the Lie algebra so(norb) which is the space of all antisymmetric
# matrices of size norb x norb, on top of the exponential map, one could
# also need an reflection to account for improper rotations,
# for the case of O(N) with N odd, the reflection can just be - identity
# 
# For instance for d orbitals norb = 5 and so(5) has dimension 10, here are the first 3 
# matrices of the Lie basis
# 
# L1 = [[ 0, 1, 0, 0, 0], L2 = [[ 0, 0, 1, 0, 0], L3 = [[ 0, 0, 0, 1, 0],
#       [-1, 0, 0, 0, 0],       [ 0, 0, 0, 0, 0],      [ 0, 0, 0, 0, 0],
#       [ 0, 0, 0, 0, 0],       [-1, 0, 0, 0, 0],      [ 0, 0, 0, 0, 0],
#       [ 0, 0, 0, 0, 0],       [ 0, 0, 0, 0, 0],      [-1, 0, 0, 0, 0],
#       [ 0, 0, 0, 0, 0]]       [ 0, 0, 0, 0, 0]]      [ 0, 0, 0, 0, 0]]

# The exponential map is given by the matrix exponential over a linear combination
# of the generators, the parameters that multiply these generators have the shape
# of an angle and are called generalized Euler angles

# R = expm(A) = expm( a1 L1 + a2 L2 + ... + a10 L10 )

# When can invert this map using the matrix logarithm
# A = logm(R) = a1 L1 + a2 L2 + ... + a10 L10
# and then extract the angles by projecting A onto the basis of generators, the projection
# in this case is easy because for instance for a1 we can just take the (1,2) entry of A
# otherwise one would need to take a trace with the generator

#
def get_so_n_lie_basis(norb):
    """Generate the basis for the Lie algebra so(norb).

    Args:
        norb (int): The dimension of the space (number of orbitals).
    Returns: 
        L_list: A list of numpy arrays representing the basis matrices.
    """
    L_list = []
    
    # Generate all pairs (i,j) with i < j
    for i in range(norb):
        for j in range(i + 1, norb):
            # Create antisymmetric matrix with 1 at (i,j) and -1 at (j,i)
            L = np.zeros((norb, norb))
            L[i, j] = 1.0
            L[j, i] = -1.0
            L_list.append(L)
    
    return L_list


def euler_angles_to_rotation(euler_angles, generators, reflection=False):
    """Construct a rotation matrix from Euler angles and generators.
    
    Args:
        euler_angles (array-like): The generalized Euler angles.
        generators (list): List of basis matrices for so(norb).
        reflection (bool): If True, apply a reflection (-I) for O(N) matrices with det = -1.
    
    Returns:
        numpy.ndarray: The rotation matrix R = expm(sum(ai * Li)) or -R for reflections.
    """
    euler_angles = np.array(euler_angles)
    
    if len(euler_angles) != len(generators):
        raise ValueError(f"Number of Euler angles ({len(euler_angles)}) must match "
                        f"number of generators ({len(generators)})")
    
    # Construct the antisymmetric matrix A = sum(ai * Li)
    A = np.zeros_like(generators[0])
    for angle, gen in zip(euler_angles, generators):
        A += angle * gen
    
    # Compute the matrix exponential
    R = expm(A)
    
    # Apply reflection if requested (for O(N) matrices with det = -1)
    if reflection:
        norb = generators[0].shape[0]
        if norb % 2 == 0:
            raise ValueError("Reflection using -I only valid for odd N in O(N)")
        R = -R
    
    return R


def rotation_to_euler_angles(R, generators, check_orthogonal=True, debug=False):
    """Extract Euler angles from an orthogonal matrix using matrix logarithm.
    
    Handles both SO(N) and O(N) matrices. For O(N) matrices with det = -1,
    decomposes into rotation and reflection parts.
    
    Args:
        R (numpy.ndarray): The orthogonal matrix (SO(norb) or O(norb)).
        generators (list): List of basis matrices for so(norb).
        check_orthogonal (bool): Whether to check if R is orthogonal.
    
    Returns:
        tuple: (euler_angles, has_reflection, need_regularization)
            - euler_angles (numpy.ndarray): The extracted Euler angles.
            - has_reflection (bool): True if the matrix has det = -1 (requires reflection).
            - need_regularization (bool): True if eigenvalues close to -1 detected (branch cut issue).
    
    Raises:
        ValueError: If check_orthogonal is True and R is not orthogonal.
    """
    R = np.array(R)
    norb = R.shape[0]
    
    if check_orthogonal:
        # Check if matrix is orthogonal: R @ R.T should be identity
        identity_check = np.allclose(R @ R.T, np.eye(norb))
        if not identity_check:
            raise ValueError("Matrix is not orthogonal")
    
    # Get the determinant
    det_R = np.linalg.det(R)

    if det_R < 0:
        raise ValueError("Matrix has det = -1, cannot decompose without reflection handling")
    
    # Check if determinant is close to ±1 (more robust approach)
    abs_det = np.abs(det_R)
    if not np.allclose(abs_det, 1.0, atol=1e-10):
        raise ValueError(f"Matrix determinant is {det_R:.6f} (|det|={abs_det:.6f}), expected ±1 for orthogonal matrix")
    
    R_rotation = R
    
    # Check for eigenvalues close to -1 (branch cut issue)
    rotation_eigenvals = np.linalg.eigvals(R_rotation)
    close_to_minus_one = np.abs(rotation_eigenvals + 1.0) < 1e-10
    need_regularization = np.any(close_to_minus_one)
    
    if need_regularization and debug:
        print(f"WARNING - rotation_to_euler_angles:\n")
        print(f"Found eigenvalues close to -1: {rotation_eigenvals[close_to_minus_one]}")
    
    # Compute the matrix logarithm of the rotation part
    A = logm(R_rotation)
    
    # Always use real part for angle extraction
    A = A.real
    
    # Extract angles by projecting onto the basis
    euler_angles = []
    for gen in generators:
        # For antisymmetric matrices, the projection is simply the Frobenius inner product
        # divided by the norm squared of the generator
        # Since our generators have norm^2 = 2, the angle is (A * gen).sum() / 2
        angle = np.sum(A * gen) / 2.0
        euler_angles.append(angle)
    
    return np.array(euler_angles), need_regularization


def canonicalize_angles(tentative_angles, generators):
    """
    Projects a set of SO(N) parameters onto the canonical principal branch.
    
    This function takes any set of Lie algebra coefficients ("angles"),
    constructs the corresponding rotation matrix, and then decomposes it back
    to find the unique, "shortest path" set of coefficients.

    This is needed because it is not sufficient to check that all your
    generalized Euler angles are in the range [-pi, pi]. Even if they are,
    they might represent a "long path" rotation, because the principal branch
    of the matrix log does not depend on the angles, but on the eigenvalues
    of the resulting Lie algebra decomposition.
    So even if individually the rotations are withing [-pi, pi], the combined
    rotation might not be and traces a "longer path" on the manifold of SO(N). 

    Args:
        tentative_angles (np.ndarray): A set of coefficients that might be
                                     outside the principal branch (a "long path").
        generators (list): The list of Lie algebra basis generators for SO(N).

    Returns:
        np.ndarray: The canonical set of angles for the same rotation,
                    guaranteed to be in the principal branch.
    """
    # Step 1: Construct the rotation matrix from the given angles.
    # This is a "many-to-one" mapping. Any set of angles that differ by a
    # generalized 2*pi rotation (a "long path") will produce the exact
    # same final orientation matrix R. This step effectively "forgets" the path.
    R = euler_angles_to_rotation(tentative_angles, generators)

    # Step 2: Decompose the matrix back into angles using the matrix logarithm.
    # This is a "one-to-one" mapping (within the principal branch). The `logm`
    # function is defined to always return the generator matrix X corresponding
    # to the unique "shortest path" rotation.
    canonical_angles, _ = rotation_to_euler_angles(R, generators)
    
    # The result is the unique, canonical representation for that rotation.
    return canonical_angles

def check_degenerate_eigenvalues(eigenvalues, tol=1e-6):
    """
    Check for degenerate eigenvalues and group them into sets.
    
    Args:
        eigenvalues (array-like): The eigenvalues to check for degeneracy.
        tol (float): Tolerance for considering eigenvalues as degenerate.
    
    Returns:
        list: List of lists, where each list contains indices of degenerate eigenvalues.
    """
    eigenvalues = np.array(eigenvalues)
    
    # Round eigenvalues to tolerance precision to group similar values
    rounded_eigenvals = np.round(eigenvalues / tol) * tol
    unique_vals = np.unique(rounded_eigenvals)
    degenerate_groups = []
    
    for val in unique_vals:
        # Find all indices with this rounded value
        indices = np.where(rounded_eigenvals == val)[0].tolist()
        
        # Only include groups with actual degeneracy (more than one eigenvalue)
        if len(indices) > 1:
            degenerate_groups.append(indices)
    
    return degenerate_groups

def decompose_rho_and_fix_gauge(rho, generators, deg_eig_tol=1e-6):
    """
    Decomposes a density matrix into its eigenvalues and eigenvectors,
    and fixes the gauge of the eigenvectors in the case there are degenerate eigenvalues
    using QR decomposition.

    Basically for the gauge fixing:
        1) Identify degenerate eigenvalue groups.
        2) For each group, extract the corresponding eigenvectors.
        3) Apply QR decomposition to the subspace of degenerate eigenvectors.
        4) Replace the original degenerate eigenvectors with the columns of Q from QR decomposition.
    
        
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!! I need to understand better why this works!
    !!! and if it is the best approach            !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Args:
        rho (np.ndarray): The density matrix to decompose (norb x norb).
        generators (list): The list of Lie algebra basis generators for SO(N).
        tol (float): Tolerance for checking degeneracy and orthogonality.

    Returns:
        tuple: (eigenvalues, gauge_fixed_eigenvectors, canonical_angles, has_reflection)
            - eigenvalues (np.ndarray): The eigenvalues of the density matrix.
            - gauge_fixed_eigenvectors (np.ndarray): The eigenvectors with fixed gauge.
            - canonical_angles (np.ndarray): The canonical Euler angles.
            - has_reflection (bool): Whether the transformation includes a reflection.
    """
    # Step 1: Diagonalize the density matrix
    eigenvalues, eigenvectors = np.linalg.eigh(rho)

    # Step 2: Check for degenerate eigenvalues 
    degenerate_groups = check_degenerate_eigenvalues(eigenvalues, tol=deg_eig_tol)

    # Step 3: Fix the gauge of the eigenvectors in each degenerate subspace
    gauge_fixed_eigenvectors = eigenvectors.copy()
    

    # handling of degenerate eigenvalues
    for group in degenerate_groups:
        if len(group) > 1:
            # Extract the subspace of degenerate eigenvectors
            subspace = eigenvectors[:, group]

            # this does not work, but should sketch how to approach this...

            # project to [1,0,0,...] to fix one degree of freedom
            # and sub the projection with the first eigenvector
            # if the projection gives the 0 vector, project
            # to [0,1,0,...], and so on
            # this fixes the first vector and the QR decomposition
            # will fix the rest in the orthogonal complement
            # for i in range(subspace.shape[1]):
            #     proj = np.zeros(subspace.shape[0])
            #     proj[i] = 1.0
            #     dot = np.dot(subspace.T, proj)
            #     if np.linalg.norm(dot) > 1e-6:
            #         newvec =  subspace @ dot
            #         subspace[:,0] = newvec / np.linalg.norm(newvec)
            #         break
                    
            
            # Apply QR decomposition to fix the gauge
            Q, R = qr(subspace, mode='economic')
            
            # Ensure positive diagonal elements in R for consistent gauge
            # (This removes the sign ambiguity in QR decomposition)
            signs = np.sign(np.diag(R))
            signs[signs == 0] = 1  # Handle zero diagonal elements
            Q = Q @ np.diag(signs)
            
            # Replace the degenerate eigenvectors with the gauge-fixed ones
            gauge_fixed_eigenvectors[:, group] = Q
    

    # handling of sign ambiguities in the definition of the R matrix
    # Any sign flip to any of the eigenvector
    # is a valid orthogonal matrix for the decomposition,
    # we fix this freedom by requiring that for the first
    # N-1 eigenvectors, it's biggest component determines
    # univocally its sign, while the N-th eigenvector's sign
    # is there to ensue that the matrix lies in SO(N), instead 
    # of the set of improper (det=-1) rotations

    for ivector in range(gauge_fixed_eigenvectors.shape[1]-1):
        vector = gauge_fixed_eigenvectors[:, ivector]
        max_comp = np.argmax(np.abs(vector))
        # print(f"{max_comp=}")
        if np.sign(vector[max_comp]) < 0:
            vector *= -1
    
    # now the last vector sign depends on the overall determinant
    temporary_det = np.linalg.det(gauge_fixed_eigenvectors)
    if temporary_det < 0:
        gauge_fixed_eigenvectors[:, -1] *= -1
        


    # Step 4: Extract canonical angles from the gauge-fixed eigenvectors
    canonical_angles, need_regularization = rotation_to_euler_angles(gauge_fixed_eigenvectors, generators, check_orthogonal=False)
    
    return eigenvalues, gauge_fixed_eigenvectors, canonical_angles, need_regularization, degenerate_groups

# Example usage:
# 
# # Generate basis for 5D (d orbitals)
# generators = get_so_n_lie_basis(5)
# 
# # Create rotation matrix from Euler angles
# angles = np.random.uniform(-np.pi, np.pi, len(generators))
# R = euler_angles_to_rotation(angles, generators)
# 
# # Extract angles from matrix (handles both SO(N) and O(N))
# extracted_angles, has_reflection = rotation_to_euler_angles(R, generators)
# 
# # For O(N) matrices with det = -1, use reflection=True
# R_with_reflection = euler_angles_to_rotation(angles, generators, reflection=True)


#%% Note on non-unique angle representations
# 
# Multiple angle sets can produce the same rotation matrix - this is expected behavior.
# Test case demonstrating this:
# 
# generators = get_so_n_lie_basis(5)
# np.random.seed(42)  # Reproducible case that shows the issue
# original_angles = np.random.uniform(-np.pi, np.pi, len(generators))
# R = euler_angles_to_rotation(original_angles, generators)
# extracted_angles, _ = rotation_to_euler_angles(R, generators)

# # original_angles != extracted_angles, but both give identical matrices:
# R1 = euler_angles_to_rotation(original_angles, generators)
# R2 = euler_angles_to_rotation(extracted_angles, generators)
# np.allclose(R1, R2)  # True - identical matrices from different angles

# # Use canonicalize_angles to get the unique principal branch representation:
# canonical_angles = canonicalize_angles(original_angles, generators)
# # extracted_angles should match canonical_angles (both are principal branch)
# np.allclose(extracted_angles, canonical_angles)  # True

# with np.printoptions(precision=3, suppress=True):
#     print("Original angles:")
#     print(original_angles)
#     print("Extracted angles (canonical):")
#     print(extracted_angles)
#     print("Canonicalized angles:")
#     print(canonical_angles)
#     print("Difference (original - canonical):")
#     print(original_angles - canonical_angles)
#%% test degeneracy detection
# # Test the improved degeneracy detection
# test_eigenvals = np.array([1.0, 1.0, 0.5, 0.5, 0.5, 0.3])
# test_groups = check_degenerate_eigenvalues(test_eigenvals, tol=1e-3)
# print(f"Test eigenvalues: {test_eigenvals}")
# print(f"Degenerate groups: {test_groups}")
# # Expected: [[0, 1], [2, 3, 4]] (indices 0,1 are degenerate at 1.0, indices 2,3,4 at 0.5)

# #%% test density matrices with non-unique eigenvalues


# # Example d-orbital density matrix
# spin_up_density_matrix = np.array([
# [ 0.575,  0.054,  0.054, -0.0,   0.108],
# [ 0.054,  0.962,  0.013,  0.094, -0.013],
# [ 0.054,  0.013,  0.962, -0.094, -0.013],
# [-0.0,    0.094, -0.094,  0.575, -0.0],
# [ 0.108, -0.013, -0.013, -0.0,   0.962]
# ])

# # Diagonalize to get eigenvectors (orthogonal matrix)
# eigenvalues, eigenvectors = np.linalg.eigh(spin_up_density_matrix)
# generators = get_so_n_lie_basis(5)
# eigenvalues, eigenvectors, _,_,deg = decompose_rho_and_fix_gauge(spin_up_density_matrix, generators=generators)



# # restore the density matrix
# restored_rho = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

# with np.printoptions(precision=3, suppress=True):
#     print("Original density matrix:")
#     print(spin_up_density_matrix)
#     print("Restored density matrix from eigen decomposition:")
#     print(restored_rho)
#     print("Difference (original - restored):")
#     print(spin_up_density_matrix - restored_rho)

# #%%
# # Test decomposition
# generators = get_so_n_lie_basis(5)
# angles, has_reflection, _ = rotation_to_euler_angles(eigenvectors, generators)


# # Test the improved functions
# degenerate_groups = check_degenerate_eigenvalues(eigenvalues)
# print(f"Degenerate groups: {degenerate_groups}")

# # Use the complete function
# eigenvals, gauge_fixed_eigenvectors, canonical_angles, has_reflection = decompose_rho_and_fix_gauge(
#     spin_up_density_matrix, generators)

# # Reconstruct the density matrix
# rho_reconstructed = gauge_fixed_eigenvectors @ np.diag(eigenvals) @ gauge_fixed_eigenvectors.T

# with np.printoptions(precision=3, suppress=True):
#     print("Original density matrix:")
#     print(spin_up_density_matrix)
#     print("Reconstructed density matrix:")
#     print(rho_reconstructed)
#     print("Difference (original - reconstructed):")
#     print(spin_up_density_matrix - rho_reconstructed)


        
# # %%
# uo2_matrix =  np.array([
#     [ 0.051, 0.0, 0.0, 0.002, 0.0, 0.0, 0.0 ],
#     [ 0.0, 0.964, 0.0, 0.0, 0.0, -0.135, 0.0 ],
#     [ 0.0, 0.0, 0.317, 0.0, 0.0, 0.0, 0.425 ],
#     [ 0.002, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0 ],
#     [ 0.0, 0.0, 0.0, 0.0, 0.139, 0.0, 0.0 ],
#     [ 0.0, -0.135, 0.0, 0.0, 0.0, 0.066, 0.0 ],
#     [ 0.0, 0.0, 0.425, 0.0, 0.0, 0.0, 0.709 ]
# ])

# gen_so7 = get_so_n_lie_basis(7)
# eigenvalues, eigenvectors = np.linalg.eigh(uo2_matrix)

# eigenvalues, gauge_fixed_eigenvectors, angles, reg, groups = decompose_rho_and_fix_gauge(eigenvectors, gen_so7)
