#%%
import numpy as np
from scipy.linalg import expm, logm

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


def rotation_to_euler_angles(R, generators, check_orthogonal=True):
    """Extract Euler angles from an orthogonal matrix using matrix logarithm.
    
    Handles both SO(N) and O(N) matrices. For O(N) matrices with det = -1,
    decomposes into rotation and reflection parts.
    
    Args:
        R (numpy.ndarray): The orthogonal matrix (SO(norb) or O(norb)).
        generators (list): List of basis matrices for so(norb).
        check_orthogonal (bool): Whether to check if R is orthogonal.
    
    Returns:
        tuple: (euler_angles, has_reflection)
            - euler_angles (numpy.ndarray): The extracted Euler angles.
            - has_reflection (bool): True if the matrix has det = -1 (requires reflection).
    
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
    has_reflection = False
    
    # Handle O(N) matrices with det = -1
    if np.allclose(det_R, -1.0):
        if norb % 2 == 0:
            raise ValueError("For even N, reflection decomposition using -I is not valid. "
                           "Use a more general reflection matrix.")
        # For odd N, we can use -I as the reflection
        # Decompose R = -I * R_rotation, so R_rotation = -R
        R_rotation = -R
        has_reflection = True
    elif np.allclose(det_R, 1.0):
        # Standard SO(N) case
        R_rotation = R
        has_reflection = False
    else:
        raise ValueError(f"Matrix determinant is {det_R:.6f}, expected ±1 for orthogonal matrix")
    
    # Compute the matrix logarithm of the rotation part
    A = logm(R_rotation)
    
    # Make sure A is real (logm can sometimes return complex with tiny imaginary parts)
    if np.allclose(A.imag, 0):
        A = A.real
    else:
        raise ValueError("Matrix logarithm has significant imaginary components")
    
    # Extract angles by projecting onto the basis
    euler_angles = []
    for gen in generators:
        # For antisymmetric matrices, the projection is simply the Frobenius inner product
        # divided by the norm squared of the generator
        # Since our generators have norm^2 = 2, the angle is (A * gen).sum() / 2
        angle = np.sum(A * gen) / 2.0
        euler_angles.append(angle)
    
    return np.array(euler_angles), has_reflection


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
#%% Test with O(N) matrix with det = -1 (reflection case)