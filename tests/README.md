# LordCapulet Tests

This directory contains the test suite for the LordCapulet AiiDA plugin.

## Test Structure

```
tests/
├── __init__.py                     # Test package initialization
├── conftest.py                     # Pytest configuration
├── test_occupation_matrix.py       # Tests for OccupationMatrixData
├── test_proposal_modes.py          # Tests for proposal generation (random, random_so_n)
├── test_so_n_decomposition.py      # Tests for SO(N) decomposition utilities
├── test_so_n_integration.py        # Integration tests for SO(N) decomposition
└── utils/                          # Tests for utility functions
    ├── __init__.py                 # Utils test package init
    └── test_rotation_matrices.py   # Tests for rotation matrix utilities
```

## Running Tests

### Simple Test Runner (No Dependencies)

Use the simple test runner that doesn't require pytest:

```bash
python run_tests.py
```

### With Pytest (Recommended)

If you have pytest installed, you can run the full test suite:

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/utils/test_rotation_matrices.py -v
```

You can also use the pytest runner script:

```bash
python pytest_runner.py
```

## Test Coverage

### Occupation Matrix Data (`tests/test_occupation_matrix.py`)

Tests for the unified `OccupationMatrixData` structure:

- ✅ Initialization (empty and with data)
- ✅ Conversion from AiiDA-QE format (`from_aiida_qe_occupations`)
- ✅ Conversion from legacy format (`from_legacy_dict`)
- ✅ Conversion from constrained matrix format (`from_constrained_matrix_format`)
- ✅ Conversion to constrained matrix format (`to_constrained_matrix_format`)
- ✅ Conversion to legacy format (`to_legacy_dict`)
- ✅ Getter methods (`get_atom_labels`, `get_atom_species`, `get_occupation_matrix`)
- ✅ Round-trip conversions (AiiDA-QE, legacy, constrained)
- ✅ JSON serializability
- ✅ Realistic 5x5 d-orbital matrices
- ✅ 1D array reshaping (flattened matrices)
- ✅ Numpy array to list conversion

### Proposal Modes (`tests/test_proposal_modes.py`)

Tests for proposal generation functions:

#### Metadata Preservation
- ✅ Random mode preserves specie and shell metadata
- ✅ Random SO(N) mode preserves specie and shell metadata

#### Random Mode (`propose_random_constraints`)
- ✅ Basic functionality
- ✅ Matrix properties (dimensions, hermiticity, real values)
- ✅ Target traces parameter
- ✅ Multiple atoms support
- ✅ Randomness verification

#### Random SO(N) Mode (`propose_random_so_n_constraints`)
- ✅ Basic functionality
- ✅ SO(N) matrix properties
- ✅ Target traces parameter
- ✅ Multiple atoms support
- ✅ Randomness verification

#### Consistency Between Modes
- ✅ Same output structure
- ✅ Atom count preservation

### SO(N) Decomposition (`tests/test_so_n_decomposition.py`)

Tests for SO(N) and O(N) matrix decomposition:

- ✅ Lie basis dimension
- ✅ SO(N) round-trip (angles → matrix → angles)
- ✅ O(N) reflection case (det = -1)
- ✅ Even dimension reflection error handling
- ✅ Invalid matrix error handling
- ✅ Angle/generator mismatch error handling
- ✅ Quantum Espresso example (realistic d-orbital)
- ✅ Basis orthogonality properties
- ✅ Angle canonicalization
- ✅ Canonicalization with long paths
- ✅ Canonicalization idempotency

### SO(N) Integration (`tests/test_so_n_integration.py`)

Integration tests that can run without pytest:

- ✅ Basic functionality test
- ✅ Angle canonicalization
- ✅ Quantum Espresso example with density matrix reconstruction

### Rotation Matrices (`tests/utils/test_rotation_matrices.py`)

Tests for the `spherical_to_cubic_rotation` function:

- ✅ Function existence and basic functionality
- ✅ Input validation (invalid conventions and dimensions)
- ✅ Matrix mathematical properties (unitarity)
- ✅ Correct matrix elements for QE convention
- ✅ Simple density matrix rotation (Y₂⁰ → r²-3z²)
- ✅ Complex density matrix rotation with multiple orbitals
- ✅ Hermiticity preservation after rotation

## Adding New Tests

### For New Utility Functions

1. Create test files in the appropriate subdirectory under `tests/`
2. Follow the naming convention: `test_<module_name>.py`
3. Use the `TestClassName` pattern for test classes
4. Make tests compatible with both pytest and standalone execution

### Example Test Structure

```python
"""Tests for new_module utilities."""

import numpy as np
from lordcapulet.utils.new_module import new_function

class TestNewFunction:
    """Test suite for new_function."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        result = new_function()
        assert result is not None
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Add your tests here
        pass

if __name__ == "__main__":
    # Standalone execution code
    pass
```

## Test Guidelines

1. **Test Independence**: Each test should be independent and not rely on the state from other tests
2. **Error Testing**: Always test both success and failure cases
3. **Mathematical Properties**: For numerical functions, test mathematical properties (unitarity, Hermiticity, etc.)
4. **Edge Cases**: Test boundary conditions and edge cases
5. **Documentation**: Add clear docstrings explaining what each test verifies
6. **Reproducibility**: Use fixed random seeds when testing with random data

## Future Test Areas

Areas that could benefit from additional testing:

- **Calculations**: Tests for `ConstrainedPWCalculation` (acceptance of JsonableData vs Dict)
- **Workflows**: Tests for `AFMScanWorkChain`, `ConstrainedScanWorkChain`, `GlobalConstrainedSearchWorkChain`
- **Functions**: Tests for `aiida_propose_occ_matrices_from_results` (AiiDA calcfunction wrapper)
- **Proposal Modes**: Tests for 'read' mode (once refactored to use `OccupationMatrixData`)
- **Integration**: End-to-end workflow tests with AiiDA database
- **Performance**: Benchmarking for large systems
- **Backward Compatibility**: Tests ensuring old Dict format still works

