"""Tests for OccupationMatrixData unified data structure."""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lordcapulet.utils.occupation_matrix import OccupationMatrixData


class TestOccupationMatrixData:
    """Test suite for OccupationMatrixData class."""
    
    def test_initialization_empty(self):
        """Test creating empty OccupationMatrixData."""
        occ_data = OccupationMatrixData()
        assert occ_data.data == {}
        assert occ_data.as_dict() == {}
    
    def test_initialization_with_data(self):
        """Test creating OccupationMatrixData with initial data."""
        test_data = {
            'Atom_1': {
                'specie': 'Fe1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.0], [0.0, 1.0]],
                    'down': [[0.5, 0.0], [0.0, 0.5]]
                }
            }
        }
        occ_data = OccupationMatrixData(test_data)
        assert occ_data.data == test_data
        assert occ_data.as_dict() == test_data
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        test_data = {
            'Atom_1': {
                'specie': 'Ni1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.0], [0.0, 0.0]],
                    'down': [[0.0, 0.0], [0.0, 1.0]]
                }
            }
        }
        occ_data = OccupationMatrixData.from_dict(test_data)
        assert occ_data.data == test_data
    
    def test_from_aiida_qe_occupations_simple(self):
        """Test conversion from AiiDA-QE get_occupations() format."""
        aiida_format = [
            {
                'atom_label': 'Atom_1',
                'atom_specie': 'Fe1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': np.array([[1.0, 0.1], [0.1, 0.5]]),
                    'down': np.array([[0.5, 0.05], [0.05, 1.0]])
                }
            }
        ]
        
        occ_data = OccupationMatrixData.from_aiida_qe_occupations(aiida_format)
        
        assert 'Atom_1' in occ_data.data
        assert occ_data.data['Atom_1']['specie'] == 'Fe1'
        assert occ_data.data['Atom_1']['shell'] == '3d'
        
        # Check matrices are converted to lists
        assert isinstance(occ_data.data['Atom_1']['occupation_matrix']['up'], list)
        assert isinstance(occ_data.data['Atom_1']['occupation_matrix']['down'], list)
        
        # Check values
        np.testing.assert_allclose(
            occ_data.data['Atom_1']['occupation_matrix']['up'],
            [[1.0, 0.1], [0.1, 0.5]]
        )
    
    def test_from_aiida_qe_occupations_multiple_atoms(self):
        """Test conversion with multiple atoms."""
        aiida_format = [
            {
                'atom_label': 'Atom_1',
                'atom_specie': 'Fe1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': np.eye(5),
                    'down': np.eye(5) * 0.5
                }
            },
            {
                'atom_label': 'Atom_2',
                'atom_specie': 'Fe2',
                'shell': '3d',
                'occupation_matrix': {
                    'up': np.eye(5) * 0.8,
                    'down': np.eye(5) * 0.2
                }
            }
        ]
        
        occ_data = OccupationMatrixData.from_aiida_qe_occupations(aiida_format)
        
        assert len(occ_data.data) == 2
        assert 'Atom_1' in occ_data.data
        assert 'Atom_2' in occ_data.data
        assert occ_data.data['Atom_2']['specie'] == 'Fe2'
    
    def test_from_aiida_qe_occupations_1d_reshape(self):
        """Test conversion when matrices are given as 1D arrays."""
        # Some versions of AiiDA-QE might return flattened matrices
        aiida_format = [
            {
                'atom_label': 'Atom_1',
                'atom_specie': 'Co1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [1.0, 0.0, 0.0, 1.0],  # Flattened 2x2 matrix
                    'down': [0.5, 0.0, 0.0, 0.5]
                }
            }
        ]
        
        occ_data = OccupationMatrixData.from_aiida_qe_occupations(aiida_format)
        
        # Should be reshaped to 2x2
        assert len(occ_data.data['Atom_1']['occupation_matrix']['up']) == 2
        assert len(occ_data.data['Atom_1']['occupation_matrix']['up'][0]) == 2
        assert occ_data.data['Atom_1']['occupation_matrix']['up'] == [[1.0, 0.0], [0.0, 1.0]]
    
    def test_from_legacy_dict_spin_data_format(self):
        """Test conversion from legacy format with spin_data."""
        legacy_format = {
            '1': {
                'specie': 'Fe1',
                'spin_data': {
                    'up': {
                        'occupation_matrix': [[1.0, 0.0], [0.0, 0.5]]
                    },
                    'down': {
                        'occupation_matrix': [[0.5, 0.0], [0.0, 1.0]]
                    }
                }
            }
        }
        
        occ_data = OccupationMatrixData.from_legacy_dict(legacy_format)
        
        assert '1' in occ_data.data
        assert occ_data.data['1']['specie'] == 'Fe1'
        assert occ_data.data['1']['occupation_matrix']['up'] == [[1.0, 0.0], [0.0, 0.5]]
    
    def test_from_legacy_dict_simple_format(self):
        """Test conversion from simple legacy format."""
        legacy_format = {
            '1': {
                'specie': 'Ni1',
                'occupation_matrix': {
                    'up': [[1.0, 0.1], [0.1, 0.9]],
                    'down': [[0.9, 0.05], [0.05, 1.0]]
                }
            }
        }
        
        occ_data = OccupationMatrixData.from_legacy_dict(legacy_format)
        
        assert '1' in occ_data.data
        assert occ_data.data['1']['specie'] == 'Ni1'
        assert occ_data.data['1']['shell'] == 'UNKNOWN'  # Should set default
    
    def test_from_legacy_dict_numpy_arrays(self):
        """Test conversion when legacy format contains numpy arrays."""
        legacy_format = {
            '1': {
                'specie': 'Cu1',
                'occupation_matrix': {
                    'up': np.array([[1.0, 0.0], [0.0, 1.0]]),
                    'down': np.array([[0.5, 0.0], [0.0, 0.5]])
                }
            }
        }
        
        occ_data = OccupationMatrixData.from_legacy_dict(legacy_format)
        
        # Should be converted to lists
        assert isinstance(occ_data.data['1']['occupation_matrix']['up'], list)
        assert occ_data.data['1']['occupation_matrix']['up'] == [[1.0, 0.0], [0.0, 1.0]]
    
    def test_from_constrained_matrix_format(self):
        """Test conversion from ConstrainedPW matrix format."""
        matrix_format = {
            'matrix': [
                [  # Atom 1
                    [[1.0, 0.0], [0.0, 0.5]],  # up
                    [[0.5, 0.0], [0.0, 1.0]]   # down
                ],
                [  # Atom 2
                    [[0.8, 0.1], [0.1, 0.9]],  # up
                    [[0.9, 0.05], [0.05, 0.8]] # down
                ]
            ]
        }
        atom_species = ['Fe1', 'Fe2']
        
        occ_data = OccupationMatrixData.from_constrained_matrix_format(matrix_format, atom_species)
        
        assert len(occ_data.data) == 2
        assert 1 in occ_data.data
        assert 2 in occ_data.data
        assert occ_data.data[1]['specie'] == 'Fe1'
        assert occ_data.data[2]['specie'] == 'Fe2'
        assert occ_data.data[1]['occupation_matrix']['up'] == [[1.0, 0.0], [0.0, 0.5]]
    
    def test_to_constrained_matrix_format(self):
        """Test conversion to ConstrainedPW matrix format."""
        test_data = {
            'Atom_1': {
                'specie': 'Fe1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.0], [0.0, 0.5]],
                    'down': [[0.5, 0.0], [0.0, 1.0]]
                }
            },
            'Atom_2': {
                'specie': 'Fe2',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[0.8, 0.1], [0.1, 0.9]],
                    'down': [[0.9, 0.05], [0.05, 0.8]]
                }
            }
        }
        
        occ_data = OccupationMatrixData(test_data)
        matrix_format = occ_data.to_constrained_matrix_format()
        
        assert 'matrix' in matrix_format
        assert len(matrix_format['matrix']) == 2
        assert matrix_format['matrix'][0][0] == [[1.0, 0.0], [0.0, 0.5]]  # Atom 1 up
        assert matrix_format['matrix'][0][1] == [[0.5, 0.0], [0.0, 1.0]]  # Atom 1 down
    
    def test_to_legacy_dict(self):
        """Test conversion to legacy format."""
        test_data = {
            'Atom_1': {
                'specie': 'Fe1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.0], [0.0, 0.5]],
                    'down': [[0.5, 0.0], [0.0, 1.0]]
                }
            }
        }
        
        occ_data = OccupationMatrixData(test_data)
        legacy_format = occ_data.to_legacy_dict()
        
        assert '1' in legacy_format
        assert legacy_format['1']['specie'] == 'Fe1'
        assert 'spin_data' in legacy_format['1']
        assert legacy_format['1']['spin_data']['up']['occupation_matrix'] == [[1.0, 0.0], [0.0, 0.5]]
    
    def test_get_atom_labels(self):
        """Test getting atom labels."""
        test_data = {
            'Atom_1': {'specie': 'Fe1', 'shell': '3d', 'occupation_matrix': {'up': [], 'down': []}},
            'Atom_2': {'specie': 'Fe2', 'shell': '3d', 'occupation_matrix': {'up': [], 'down': []}}
        }
        
        occ_data = OccupationMatrixData(test_data)
        labels = occ_data.get_atom_labels()
        
        assert len(labels) == 2
        assert 'Atom_1' in labels
        assert 'Atom_2' in labels
    
    def test_get_atom_species(self):
        """Test getting atom species."""
        test_data = {
            'Atom_1': {'specie': 'Fe1', 'shell': '3d', 'occupation_matrix': {'up': [], 'down': []}},
            'Atom_2': {'specie': 'Ni1', 'shell': '3d', 'occupation_matrix': {'up': [], 'down': []}}
        }
        
        occ_data = OccupationMatrixData(test_data)
        species = occ_data.get_atom_species()
        
        assert len(species) == 2
        assert 'Fe1' in species
        assert 'Ni1' in species
    
    def test_get_occupation_matrix(self):
        """Test getting specific occupation matrix."""
        test_data = {
            'Atom_1': {
                'specie': 'Fe1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.1], [0.1, 0.5]],
                    'down': [[0.5, 0.05], [0.05, 1.0]]
                }
            }
        }
        
        occ_data = OccupationMatrixData(test_data)
        
        up_matrix = occ_data.get_occupation_matrix('Atom_1', 'up')
        down_matrix = occ_data.get_occupation_matrix('Atom_1', 'down')
        
        assert up_matrix == [[1.0, 0.1], [0.1, 0.5]]
        assert down_matrix == [[0.5, 0.05], [0.05, 1.0]]
    
    def test_round_trip_aiida_qe(self):
        """Test round-trip conversion: AiiDA-QE → OccupationMatrixData → AiiDA format."""
        aiida_format = [
            {
                'atom_label': 'Atom_1',
                'atom_specie': 'Fe1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': np.eye(5) * 0.9,
                    'down': np.eye(5) * 0.1
                }
            }
        ]
        
        # Forward conversion
        occ_data = OccupationMatrixData.from_aiida_qe_occupations(aiida_format)
        
        # Verify data preserved
        assert occ_data.data['Atom_1']['specie'] == 'Fe1'
        assert occ_data.data['Atom_1']['shell'] == '3d'
        
        # Verify matrix content
        up_matrix = np.array(occ_data.data['Atom_1']['occupation_matrix']['up'])
        expected = np.eye(5) * 0.9
        np.testing.assert_allclose(up_matrix, expected, atol=1e-10)
    
    def test_round_trip_legacy(self):
        """Test round-trip conversion: legacy → OccupationMatrixData → legacy."""
        legacy_format = {
            '1': {
                'specie': 'Fe1',
                'spin_data': {
                    'up': {'occupation_matrix': [[1.0, 0.0], [0.0, 0.5]]},
                    'down': {'occupation_matrix': [[0.5, 0.0], [0.0, 1.0]]}
                }
            }
        }
        
        # Forward conversion
        occ_data = OccupationMatrixData.from_legacy_dict(legacy_format)
        
        # Backward conversion
        legacy_back = occ_data.to_legacy_dict()
        
        # Verify matrices match
        assert legacy_back['1']['spin_data']['up']['occupation_matrix'] == [[1.0, 0.0], [0.0, 0.5]]
        assert legacy_back['1']['spin_data']['down']['occupation_matrix'] == [[0.5, 0.0], [0.0, 1.0]]
    
    
    def test_json_serializability(self):
        """Test that data can be JSON serialized."""
        import json
        
        test_data = {
            'Atom_1': {
                'specie': 'Fe1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.0], [0.0, 0.5]],
                    'down': [[0.5, 0.0], [0.0, 1.0]]
                }
            }
        }
        
        occ_data = OccupationMatrixData(test_data)
        
        # Should be JSON serializable
        json_str = json.dumps(occ_data.as_dict())
        
        # Should be able to load it back
        loaded_data = json.loads(json_str)
        assert loaded_data == test_data
    
    def test_complex_5x5_d_orbital_matrix(self):
        """Test with realistic 5x5 d-orbital occupation matrix."""
        # Realistic d-orbital matrix from DFT calculation
        d_orbital_up = [
            [0.575, 0.054, 0.054, 0.0, 0.108],
            [0.054, 0.962, 0.013, 0.094, -0.013],
            [0.054, 0.013, 0.962, -0.094, -0.013],
            [0.0, 0.094, -0.094, 0.575, 0.0],
            [0.108, -0.013, -0.013, 0.0, 0.962]
        ]
        
        d_orbital_down = [
            [0.425, 0.042, 0.042, 0.0, 0.085],
            [0.042, 0.900, 0.010, 0.070, -0.010],
            [0.042, 0.010, 0.900, -0.070, -0.010],
            [0.0, 0.070, -0.070, 0.425, 0.0],
            [0.085, -0.010, -0.010, 0.0, 0.900]
        ]
        
        test_data = {
            'Atom_1': {
                'specie': 'Fe1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': d_orbital_up,
                    'down': d_orbital_down
                }
            }
        }
        
        occ_data = OccupationMatrixData(test_data)
        
        # Test conversion to constrained format
        constrained = occ_data.to_constrained_matrix_format()
        assert len(constrained['matrix'][0][0]) == 5
        assert len(constrained['matrix'][0][0][0]) == 5
        
        # Verify trace is preserved
        trace_up_original = sum(d_orbital_up[i][i] for i in range(5))
        trace_up_converted = sum(constrained['matrix'][0][0][i][i] for i in range(5))
        np.testing.assert_allclose(trace_up_original, trace_up_converted, atol=1e-10)


if __name__ == "__main__":
    # Run tests if executed directly
    test_suite = TestOccupationMatrixData()
    
    print("Running OccupationMatrixData tests...")
    
    test_suite.test_initialization_empty()
    print("Empty initialization test passed (^_^)")
    
    test_suite.test_initialization_with_data()
    print("Initialization with data test passed (^_^)")
    
    test_suite.test_from_dict()
    print("from_dict test passed (^_^)")
    
    test_suite.test_from_aiida_qe_occupations_simple()
    print("AiiDA-QE conversion (simple) test passed (^_^)")
    
    test_suite.test_from_aiida_qe_occupations_multiple_atoms()
    print("AiiDA-QE conversion (multiple atoms) test passed (^_^)")
    
    test_suite.test_from_aiida_qe_occupations_1d_reshape()
    print("AiiDA-QE 1D reshape test passed (^_^)")
    
    test_suite.test_from_legacy_dict_spin_data_format()
    print("Legacy format (spin_data) test passed (^_^)")
    
    test_suite.test_from_legacy_dict_simple_format()
    print("Legacy format (simple) test passed (^_^)")
    
    test_suite.test_from_constrained_matrix_format()
    print("Constrained format conversion test passed (^_^)")
    
    test_suite.test_to_constrained_matrix_format()
    print("to_constrained_matrix_format test passed (^_^)")
    
    test_suite.test_to_legacy_dict()
    print("to_legacy_dict test passed (^_^)")
    
    test_suite.test_get_atom_labels()
    print("get_atom_labels test passed (^_^)")
    
    test_suite.test_get_atom_species()
    print("get_atom_species test passed (^_^)")
    
    test_suite.test_get_occupation_matrix()
    print("get_occupation_matrix test passed (^_^)")
    
    test_suite.test_round_trip_aiida_qe()
    print("Round-trip AiiDA-QE test passed (^_^)")
    
    test_suite.test_round_trip_legacy()
    print("Round-trip legacy test passed (^_^)")
    
    test_suite.test_round_trip_constrained()
    print("Round-trip constrained test passed (^_^)")
    
    test_suite.test_json_serializability()
    print("JSON serializability test passed (^_^)")
    
    test_suite.test_complex_5x5_d_orbital_matrix()
    print("Complex 5x5 d-orbital test passed (^_^)")
    
    print("\nAll OccupationMatrixData tests passed! (^_^)")
