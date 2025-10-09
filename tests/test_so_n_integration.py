"""Simple integration tests for SO(N) decomposition that can be run directly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from lordcapulet.utils.so_n_decomposition import (
    get_so_n_lie_basis,
    euler_angles_to_rotation, 
    rotation_to_euler_angles,
    canonicalize_angles
)


def test_basic_functionality():
    """Basic functionality test that can be run without pytest."""
    print("Testing SO(N) decomposition basic functionality...")
    
    # Test SO(5) - d orbitals
    norb = 5
    generators = get_so_n_lie_basis(norb)
    
    # Test 1: Check basis dimension
    expected_dim = norb * (norb - 1) // 2
    assert len(generators) == expected_dim, f"Expected {expected_dim} generators, got {len(generators)}"
    print(f"Correct number of generators: {len(generators)} (^_^)")
    
    # Test 2: SO(N) round-trip
    np.random.seed(42)
    angles = np.random.uniform(-np.pi, np.pi, len(generators))
    R = euler_angles_to_rotation(angles, generators)
    
    # Check SO(N) properties
    assert np.allclose(R @ R.T, np.eye(norb)), "Matrix not orthogonal"
    assert np.allclose(np.linalg.det(R), 1.0), "Determinant not 1"
    print("SO(N) matrix properties verified (^_^)")
    
    # Extract angles
    extracted_angles, has_reflection = rotation_to_euler_angles(R, generators)
    assert not has_reflection, "Should not have reflection for SO(N)"
    
    # Check matrix reconstruction (more important than exact angle matching)
    R_reconstructed = euler_angles_to_rotation(extracted_angles, generators)
    assert np.allclose(R, R_reconstructed, atol=1e-12), "Matrix reconstruction failed"
    print("SO(N) round-trip successful (^_^)")
    
    # Test 3: O(N) with reflection
    R_reflection = euler_angles_to_rotation(angles, generators, reflection=True)
    assert np.allclose(np.linalg.det(R_reflection), -1.0), "Reflection matrix det should be -1"
    
    extracted_refl_angles, has_reflection = rotation_to_euler_angles(R_reflection, generators)
    assert has_reflection, "Should detect reflection"
    
    # Check matrix reconstruction (more important than exact angle matching)
    R_refl_reconstructed = euler_angles_to_rotation(extracted_refl_angles, generators, reflection=True)
    assert np.allclose(R_reflection, R_refl_reconstructed, atol=1e-12), "Reflection matrix reconstruction failed"
    print("O(N) reflection case successful (^_^)")
    
    print("All basic tests passed!")


def test_canonicalize_angles():
    """Test angle canonicalization."""
    print("\nTesting angle canonicalization...")
    
    generators = get_so_n_lie_basis(3)
    np.random.seed(42)  # Known case with different representations
    
    original_angles = np.random.uniform(-np.pi, np.pi, len(generators))
    canonical_angles = canonicalize_angles(original_angles, generators)
    
    # Both should give same matrix
    R1 = euler_angles_to_rotation(original_angles, generators)
    R2 = euler_angles_to_rotation(canonical_angles, generators)
    assert np.allclose(R1, R2, atol=1e-12), "Canonicalization failed"
    
    # Canonical should match extracted
    extracted_angles, _ = rotation_to_euler_angles(R1, generators)
    assert np.allclose(canonical_angles, extracted_angles, atol=1e-12), "Canonical != extracted"
    
    print("Angle canonicalization successful (^_^)")


def test_quantum_espresso_example():
    """Test with realistic quantum espresso data."""
    print("\nTesting with Quantum Espresso example...")
    
    # Realistic d-orbital density matrix
    density_matrix = np.array([
        [ 0.575,  0.054,  0.054, -0.0,   0.108],
        [ 0.054,  0.962,  0.013,  0.094, -0.013],
        [ 0.054,  0.013,  0.962, -0.094, -0.013],
        [-0.0,    0.094, -0.094,  0.575, -0.0],
        [ 0.108, -0.013, -0.013, -0.0,   0.962]
    ])
    
    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvector determinant: {np.linalg.det(eigenvectors):.6f}")
    
    # Decompose
    generators = get_so_n_lie_basis(5)
    angles, has_reflection = rotation_to_euler_angles(eigenvectors, generators)
    
    print(f"Matrix type: {'O(N)' if has_reflection else 'SO(N)'}")
    print(f"Number of Euler angles extracted: {len(angles)}")
    
    # Verify reconstruction
    eigenvectors_reconstructed = euler_angles_to_rotation(
        angles, generators, reflection=has_reflection
    )
    reconstruction_error = np.max(np.abs(eigenvectors - eigenvectors_reconstructed))
    print(f"Eigenvector reconstruction error: {reconstruction_error:.2e}")
    
    # Full density matrix reconstruction
    density_reconstructed = (
        eigenvectors_reconstructed @ np.diag(eigenvalues) @ eigenvectors_reconstructed.T
    )
    density_error = np.max(np.abs(density_matrix - density_reconstructed))
    print(f"Full density matrix reconstruction error: {density_error:.2e}")
    
    assert reconstruction_error < 1e-12, "Eigenvector reconstruction failed"
    assert density_error < 1e-12, "Density matrix reconstruction failed"
    print("Quantum Espresso example successful (^_^)")


if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_canonicalize_angles()
        test_quantum_espresso_example()
        print("\nAll integration tests passed! (^_^)")
    except AssertionError as e:
        print(f"\nAssertion failed: {e} (>_<)")
        raise
    except Exception as e:
        print(f"\nUnexpected error: {e} (>_<)")
        raise