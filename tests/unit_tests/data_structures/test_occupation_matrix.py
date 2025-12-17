"""Tests for OccupationMatrixData - essential functionality only."""

import numpy as np
import pytest

from lordcapulet.data_structures.occupation_matrix import OccupationMatrixData


class TestOccupationMatrixData:
    """Test suite for OccupationMatrixData class - core features only."""
    
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
    
    def test_get_trace_up(self):
        """Test getting trace of spin-up matrix."""
        test_data = {
            'Atom_1': {
                'specie': 'Fe1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.1], [0.1, 2.0]],
                    'down': [[0.5, 0.0], [0.0, 0.5]]
                }
            }
        }
        occ_data = OccupationMatrixData(test_data)
        trace_up = occ_data.get_trace_up('Atom_1')
        assert trace_up == pytest.approx(3.0)  # 1.0 + 2.0
    
    def test_get_trace_down(self):
        """Test getting trace of spin-down matrix."""
        test_data = {
            'Atom_1': {
                'specie': 'Fe1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.1], [0.1, 2.0]],
                    'down': [[0.5, 0.0], [0.0, 0.5]]
                }
            }
        }
        occ_data = OccupationMatrixData(test_data)
        trace_down = occ_data.get_trace_down('Atom_1')
        assert trace_down == pytest.approx(1.0)  # 0.5 + 0.5
    
    def test_get_electron_number(self):
        """Test getting total electron number (trace_up + trace_down)."""
        test_data = {
            'Atom_1': {
                'specie': 'Fe1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.1], [0.1, 2.0]],
                    'down': [[0.5, 0.0], [0.0, 0.5]]
                }
            }
        }
        occ_data = OccupationMatrixData(test_data)
        electron_num = occ_data.get_electron_number('Atom_1')
        assert electron_num == pytest.approx(4.0)  # 3.0 + 1.0
    
    def test_get_magnetic_moment(self):
        """Test getting magnetic moment (trace_up - trace_down)."""
        test_data = {
            'Atom_1': {
                'specie': 'Fe1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': [[1.0, 0.1], [0.1, 2.0]],
                    'down': [[0.5, 0.0], [0.0, 0.5]]
                }
            }
        }
        occ_data = OccupationMatrixData(test_data)
        moment = occ_data.get_magnetic_moment('Atom_1')
        assert moment == pytest.approx(2.0)  # 3.0 - 1.0
    
    def test_from_aiida_qe_occupations_simple(self):
        """Test conversion from AiiDA-QE get_occupations() format."""
        aiida_format = [
            {
                'atom_index': 'Atom_1',
                'kind_name': 'Fe1',
                'manifold': '3d',
                'occupations': {
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
        np.testing.assert_allclose(
            occ_data.data['Atom_1']['occupation_matrix']['up'],
            [[1.0, 0.1], [0.1, 0.5]]
        )
    
    def test_from_aiida_qe_occupations_multiple_atoms(self):
        """Test conversion with multiple atoms."""
        aiida_format = [
            {
                'atom_index': 'Atom_1',
                'kind_name': 'Fe1',
                'manifold': '3d',
                'occupations': {
                    'up': np.eye(5),
                    'down': np.eye(5) * 0.5
                }
            },
            {
                'atom_index': 'Atom_2',
                'kind_name': 'Fe2',
                'manifold': '3d',
                'occupations': {
                    'up': np.eye(5) * 0.8,
                    'down': np.eye(5) * 0.2
                }
            }
        ]
        
        occ_data = OccupationMatrixData.from_aiida_qe_occupations(aiida_format)
        
        assert len(occ_data.data) == 2
        assert 'Atom_1' in occ_data.data
        assert 'Atom_2' in occ_data.data
    
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
        
        assert up_matrix == [[1.0, 0.1], [0.1, 0.5]]
    
    def test_get_occupation_matrix_as_numpy(self):
        """Test getting occupation matrix as numpy array."""
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
        up_matrix = occ_data.get_occupation_matrix_as_numpy('Atom_1', 'up')
        
        assert isinstance(up_matrix, np.ndarray)
        np.testing.assert_allclose(up_matrix, [[1.0, 0.1], [0.1, 0.5]])
    
    def test_set_occupation_matrix(self):
        """Test setting occupation matrix."""
        occ_data = OccupationMatrixData()
        
        matrix = [[1.0, 0.0], [0.0, 0.8]]
        occ_data.set_occupation_matrix('Atom_1', 'up', matrix)
        
        assert 'Atom_1' in occ_data.data
        assert occ_data.data['Atom_1']['occupation_matrix']['up'] == matrix
    
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
        json_str = json.dumps(occ_data.as_dict())
        loaded_data = json.loads(json_str)
        
        assert loaded_data == test_data
    
    def test_realistic_5x5_d_orbital(self):
        """Test with realistic 5x5 d-orbital occupation matrix."""
        d_orbital_up = [
            [0.575, 0.054, 0.054, 0.0, 0.108],
            [0.054, 0.962, 0.013, 0.094, -0.013],
            [0.054, 0.013, 0.962, -0.094, -0.013],
            [0.0, 0.094, -0.094, 0.575, 0.0],
            [0.108, -0.013, -0.013, 0.0, 0.962]
        ]
        
        test_data = {
            'Atom_1': {
                'specie': 'Fe1',
                'shell': '3d',
                'occupation_matrix': {
                    'up': d_orbital_up,
                    'down': d_orbital_up  # Reuse for simplicity
                }
            }
        }
        
        occ_data = OccupationMatrixData(test_data)
        up_matrix = occ_data.get_occupation_matrix_as_numpy('Atom_1', 'up')
        
        # Verify trace
        trace_up = np.trace(up_matrix)
        expected_trace = sum(d_orbital_up[i][i] for i in range(5))
        np.testing.assert_allclose(trace_up, expected_trace, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
