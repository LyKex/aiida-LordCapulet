"""Tests for DataBank class - essential functionality only."""

import numpy as np
import pytest
import json
import tempfile
from pathlib import Path

from lordcapulet.data_structures import DataBank, OccupationMatrixData


@pytest.fixture
def sample_json_data():
    """Create sample JSON data mimicking gather_workchain_data output."""
    return {
        "metadata": {
            "total_calculations_found": 3,
            "extraction_timestamp": "2025-12-15T12:00:00",
            "pk": 12345,
            "process_type": "test.workchain",
            "node_type": "WorkChainNode"
        },
        "statistics": {
            "total_calculations": 3,
            "converged_calculations": 2,
            "non_converged_calculations": 1
        },
        "calculations": {
            "100": {
                "pk": 100,
                "exit_status": 0,
                "converged": True,
                "process_type": "test.calculation",
                "calculation_source": "test",
                "output_parameters": {
                    "energy": -100.5,
                    "energy_hubbard": -10.2
                },
                "occupation_matrices": {
                    "Atom_1": {
                        "specie": "Fe1",
                        "shell": "3d",
                        "occupation_matrix": {
                            "up": [[1.0, 0.0], [0.0, 0.8]],
                            "down": [[0.5, 0.0], [0.0, 0.6]]
                        }
                    },
                    "Atom_2": {
                        "specie": "Fe2",
                        "shell": "3d",
                        "occupation_matrix": {
                            "up": [[0.9, 0.0], [0.0, 0.7]],
                            "down": [[0.6, 0.0], [0.0, 0.5]]
                        }
                    }
                }
            },
            "101": {
                "pk": 101,
                "exit_status": 0,
                "converged": True,
                "process_type": "test.calculation",
                "calculation_source": "test",
                "output_parameters": {
                    "energy": -105.3,
                    "energy_hubbard": -12.1
                },
                "occupation_matrices": {
                    "Atom_1": {
                        "specie": "Fe1",
                        "shell": "3d",
                        "occupation_matrix": {
                            "up": [[0.95, 0.0], [0.0, 0.85]],
                            "down": [[0.55, 0.0], [0.0, 0.65]]
                        }
                    },
                    "Atom_2": {
                        "specie": "Fe2",
                        "shell": "3d",
                        "occupation_matrix": {
                            "up": [[0.88, 0.0], [0.0, 0.72]],
                            "down": [[0.62, 0.0], [0.0, 0.52]]
                        }
                    }
                }
            },
            "102": {
                "pk": 102,
                "exit_status": 410,
                "converged": False,
                "process_type": "test.calculation",
                "calculation_source": "test",
                "output_parameters": {
                    "energy": -98.2,
                    "energy_hubbard": -9.5
                },
                "occupation_matrices": {
                    "Atom_1": {
                        "specie": "Fe1",
                        "shell": "3d",
                        "occupation_matrix": {
                            "up": [[1.0, 0.0], [0.0, 0.9]],
                            "down": [[0.4, 0.0], [0.0, 0.7]]
                        }
                    },
                    "Atom_2": {
                        "specie": "Fe2",
                        "shell": "3d",
                        "occupation_matrix": {
                            "up": [[0.85, 0.0], [0.0, 0.75]],
                            "down": [[0.65, 0.0], [0.0, 0.55]]
                        }
                    }
                }
            }
        }
    }


@pytest.fixture
def sample_json_file(sample_json_data, tmp_path):
    """Create a temporary JSON file with sample data."""
    json_file = tmp_path / "test_data.json"
    with open(json_file, 'w') as f:
        json.dump(sample_json_data, f)
    return json_file


class TestDataBankBasics:
    """Test basic DataBank functionality."""
    
    def test_from_json_loading(self, sample_json_file):
        """Test loading from JSON file."""
        db = DataBank.from_json(sample_json_file)
        
        # Should load only converged by default
        assert len(db) == 2
        assert all(db.converged)
    
    def test_from_json_include_non_converged(self, sample_json_file):
        """Test loading all calculations including non-converged."""
        db = DataBank.from_json(sample_json_file, only_converged=False)
        
        assert len(db) == 3
        assert sum(db.converged) == 2
        assert sum(~db.converged) == 1
    
    def test_basic_properties(self, sample_json_file):
        """Test basic property access."""
        db = DataBank.from_json(sample_json_file)
        
        # Check properties
        assert len(db.pks) == 2
        assert len(db.energies) == 2
        assert len(db.energy_uncertainties) == 2
        assert all(db.energy_uncertainties == 0.0)  # Temporary placeholder
        
        # Check atom structure
        assert len(db.atom_ids) == 2
        assert 'Atom_1' in db.atom_ids
        assert 'Atom_2' in db.atom_ids
    
    def test_get_n_orbitals(self, sample_json_file):
        """Test getting orbital counts."""
        db = DataBank.from_json(sample_json_file)
        
        # By string label
        assert db.get_n_orbitals('Atom_1') == 2
        assert db.get_n_orbitals('Atom_2') == 2
        
        # By integer index
        assert db.get_n_orbitals(0) == 2
        assert db.get_n_orbitals(1) == 2
        
        # n_orbitals_dict property
        n_orb_dict = db.n_orbitals_dict
        assert n_orb_dict['Atom_1'] == 2
        assert n_orb_dict['Atom_2'] == 2


class TestDataBankFiltering:
    """Test filtering operations."""
    
    def test_filter_converged(self, sample_json_file):
        """Test filtering by convergence status."""
        db = DataBank.from_json(sample_json_file, only_converged=False)
        
        # Filter for converged
        db_conv = db.filter_converged(True)
        assert len(db_conv) == 2
        assert all(db_conv.converged)
        
        # Filter for non-converged
        db_nonconv = db.filter_converged(False)
        assert len(db_nonconv) == 1
        assert not any(db_nonconv.converged)
    
    def test_filter_energy_range(self, sample_json_file):
        """Test filtering by energy range."""
        db = DataBank.from_json(sample_json_file)
        
        # Filter to specific range
        db_filtered = db.filter_energy_range(min_energy=-106, max_energy=-100)
        assert len(db_filtered) == 2
        assert all(db_filtered.energies >= -106)
        assert all(db_filtered.energies <= -100)
        
        # Filter with only max
        db_low = db.filter_energy_range(max_energy=-104)
        assert len(db_low) == 1
        assert db_low.energies[0] == -105.3
    
    def test_filter_atoms(self, sample_json_file):
        """Test filtering by atom presence."""
        db = DataBank.from_json(sample_json_file)
        
        # All calculations have both atoms, so should be unchanged
        db_filtered = db.filter_atoms(['Atom_1', 'Atom_2'])
        assert len(db_filtered) == len(db)


class TestDataBankIndexing:
    """Test indexing and slicing operations."""
    
    def test_integer_indexing(self, sample_json_file):
        """Test single integer indexing."""
        db = DataBank.from_json(sample_json_file)
        
        db_single = db[0]
        assert len(db_single) == 1
        assert isinstance(db_single, DataBank)
    
    def test_slice_indexing(self, sample_json_file):
        """Test slicing."""
        db = DataBank.from_json(sample_json_file, only_converged=False)
        
        db_slice = db[0:2]
        assert len(db_slice) == 2
        assert isinstance(db_slice, DataBank)
    
    def test_array_indexing(self, sample_json_file):
        """Test array/list indexing."""
        db = DataBank.from_json(sample_json_file, only_converged=False)
        
        db_selected = db[[0, 2]]
        assert len(db_selected) == 2


class TestDataBankSorting:
    """Test sorting operations."""
    
    def test_sort_by_energy(self, sample_json_file):
        """Test sorting by energy."""
        db = DataBank.from_json(sample_json_file)
        
        # Sort ascending
        db_sorted = db.sort_by_energy(ascending=True)
        energies = db_sorted.energies
        assert all(energies[i] <= energies[i+1] for i in range(len(energies)-1))
        
        # Sort descending
        db_sorted_desc = db.sort_by_energy(ascending=False)
        energies_desc = db_sorted_desc.energies
        assert all(energies_desc[i] >= energies_desc[i+1] for i in range(len(energies_desc)-1))
    
    def test_sort_by_pk(self, sample_json_file):
        """Test sorting by PK."""
        db = DataBank.from_json(sample_json_file)
        
        db_sorted = db.sort_by_pk(ascending=True)
        pks = db_sorted.pks
        assert all(pks[i] <= pks[i+1] for i in range(len(pks)-1))


class TestDataBankModification:
    """Test append and remove operations."""
    
    def test_append_databank(self, sample_json_file):
        """Test appending another DataBank."""
        db1 = DataBank.from_json(sample_json_file)
        db2 = DataBank.from_json(sample_json_file)
        
        db_combined = db1.append(db2)
        assert len(db_combined) == len(db1) + len(db2)
    
    def test_remove_by_index(self, sample_json_file):
        """Test removing by index."""
        db = DataBank.from_json(sample_json_file)
        original_len = len(db)
        
        db_removed = db.remove(0)
        assert len(db_removed) == original_len - 1
    
    def test_remove_by_pk(self, sample_json_file):
        """Test removing by PK."""
        db = DataBank.from_json(sample_json_file)
        original_len = len(db)
        pk_to_remove = db.pks[0]
        
        db_removed = db.remove_by_pk(pk_to_remove)
        assert len(db_removed) == original_len - 1
        assert pk_to_remove not in db_removed.pks


class TestDataBankPyTorchConversion:
    """Test PyTorch/numpy conversion - core functionality."""
    
    def test_to_numpy_basic(self, sample_json_file):
        """Test conversion to numpy arrays."""
        db = DataBank.from_json(sample_json_file)
        
        # Convert to numpy
        matrices = db.to_numpy()
        
        assert isinstance(matrices, np.ndarray)
        assert matrices.shape[0] == len(db)
        assert matrices.shape[1] > 0  # Has some features
    
    def test_to_numpy_with_energies(self, sample_json_file):
        """Test getting matrices and energies together."""
        db = DataBank.from_json(sample_json_file)
        
        matrices, energies = db.to_numpy(include_energies=True)
        
        assert matrices.shape[0] == energies.shape[0]
        assert len(energies) == len(db)
    
    def test_to_numpy_atom_selection(self, sample_json_file):
        """Test selecting specific atoms for flattening."""
        db = DataBank.from_json(sample_json_file)
        
        # All atoms
        matrices_all = db.to_numpy()
        
        # Single atom
        matrices_single = db.to_numpy(atom_ids=['Atom_1'])
        
        # Single atom should have fewer features
        assert matrices_single.shape[1] < matrices_all.shape[1]
    
    def test_to_numpy_spin_selection(self, sample_json_file):
        """Test selecting specific spins for flattening."""
        db = DataBank.from_json(sample_json_file)
        
        # Both spins
        matrices_both = db.to_numpy(spins=['up', 'down'])
        
        # Single spin
        matrices_up = db.to_numpy(spins=['up'])
        
        # Single spin should have half the features (roughly)
        assert matrices_up.shape[1] < matrices_both.shape[1]
    
    def test_from_numpy_reconstruction(self, sample_json_file):
        """Test reconstructing OccupationMatrixData from numpy."""
        db = DataBank.from_json(sample_json_file)
        
        # Convert to numpy
        matrices = db.to_numpy()
        
        # Reconstruct
        occ_data_list = db.from_numpy(matrices)
        
        assert len(occ_data_list) == len(db)
        assert all(isinstance(occ, OccupationMatrixData) for occ in occ_data_list)
        
        # Check structure is preserved
        original_occ = db.get_occ_data(0)
        reconstructed_occ = occ_data_list[0]
        
        assert set(original_occ.get_atom_labels()) == set(reconstructed_occ.get_atom_labels())
    
    def test_round_trip_numpy(self, sample_json_file):
        """Test round-trip conversion preserves data."""
        db = DataBank.from_json(sample_json_file)
        
        # Get original data
        original_matrices = db.to_numpy()
        
        # Round trip
        occ_data_list = db.from_numpy(original_matrices)
        
        # Create new databank and flatten again
        # Note: we need to manually create records since from_numpy returns just occ_data
        # This tests the mathematical reconstruction, not full record preservation
        for i, occ_data in enumerate(occ_data_list):
            atom_1_up = occ_data.get_occupation_matrix('Atom_1', 'up')
            orig_occ_data = db.get_occ_data(i)
            orig_atom_1_up = orig_occ_data.get_occupation_matrix('Atom_1', 'up')
            
            # Check matrices match (accounting for symmetry filling)
            np.testing.assert_allclose(atom_1_up, orig_atom_1_up, atol=1e-10)


class TestDataBankConvenience:
    """Test convenience methods."""
    
    def test_as_dict(self, sample_json_file):
        """Test exporting as dictionary."""
        db = DataBank.from_json(sample_json_file)
        
        records = db.as_dict()
        
        assert isinstance(records, list)
        assert len(records) == len(db)
        assert all('pk' in r for r in records)
        assert all('energy' in r for r in records)
        assert all('occ_data' in r for r in records)
        # occ_data should be dict, not OccupationMatrixData object
        assert all(isinstance(r['occ_data'], dict) for r in records)
    
    def test_summary(self, sample_json_file):
        """Test summary string generation."""
        db = DataBank.from_json(sample_json_file)
        
        summary = db.summary()
        
        assert isinstance(summary, str)
        assert 'Total calculations' in summary
        assert 'Converged' in summary
        assert 'Energy range' in summary
    
    def test_get_record(self, sample_json_file):
        """Test getting individual records."""
        db = DataBank.from_json(sample_json_file)
        
        record = db.get_record(0)
        
        assert isinstance(record, dict)
        assert 'pk' in record
        assert 'energy' in record
        assert 'occ_data' in record
        assert isinstance(record['occ_data'], OccupationMatrixData)
    
    def test_get_occ_data(self, sample_json_file):
        """Test getting OccupationMatrixData directly."""
        db = DataBank.from_json(sample_json_file)
        
        occ_data = db.get_occ_data(0)
        
        assert isinstance(occ_data, OccupationMatrixData)
        assert len(occ_data.get_atom_labels()) > 0


# Skip torch tests if torch not available
try:
    import torch
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestDataBankPyTorch:
    """Test PyTorch-specific functionality."""
    
    def test_to_pytorch_basic(self, sample_json_file):
        """Test conversion to PyTorch tensors."""
        db = DataBank.from_json(sample_json_file)
        
        matrices, energies = db.to_pytorch(include_energies=True)
        
        assert torch.is_tensor(matrices)
        assert torch.is_tensor(energies)
        assert matrices.shape[0] == len(db)
        assert energies.shape[0] == len(db)
    
    def test_to_pytorch_device(self, sample_json_file):
        """Test PyTorch tensor device placement."""
        db = DataBank.from_json(sample_json_file)
        
        matrices, energies = db.to_pytorch(device='cpu')
        
        assert matrices.device.type == 'cpu'
        assert energies.device.type == 'cpu'
    
    def test_from_pytorch_reconstruction(self, sample_json_file):
        """Test reconstructing from PyTorch tensors."""
        db = DataBank.from_json(sample_json_file)
        
        matrices, _ = db.to_pytorch(include_energies=True)
        occ_data_list = db.from_pytorch(matrices)
        
        assert len(occ_data_list) == len(db)
        assert all(isinstance(occ, OccupationMatrixData) for occ in occ_data_list)


class TestElectronNumberAndMoment:
    """Test electron number and magnetic moment calculations."""
    
    def test_get_magnetic_moment_single_atom(self, sample_json_file):
        """Test getting magnetic moment for a specific atom."""
        db = DataBank.from_json(sample_json_file)
        moments = db.get_magnetic_moment('Atom_1')
        
        assert isinstance(moments, np.ndarray)
        assert len(moments) == 2
        # For Atom_1 in first calc: 1.8 - 1.1 = 0.7
        assert moments[0] == pytest.approx(0.7)
        # For Atom_1 in second calc: 1.8 - 1.2 = 0.6
        assert moments[1] == pytest.approx(0.6)
        
        # Test with calc_index
        moment_0 = db.get_magnetic_moment('Atom_1', calc_index=0)
        assert isinstance(moment_0, float)
        assert moment_0 == pytest.approx(0.7)
    
    def test_get_electron_number_single_atom(self, sample_json_file):
        """Test getting electron number for a specific atom."""
        db = DataBank.from_json(sample_json_file)
        electron_nums = db.get_electron_number('Atom_1')
        
        assert isinstance(electron_nums, np.ndarray)
        assert len(electron_nums) == 2
        # For Atom_1 in first calc: 1.8 + 1.1 = 2.9
        assert electron_nums[0] == pytest.approx(2.9)
        # For Atom_1 in second calc: 1.8 + 1.2 = 3.0
        assert electron_nums[1] == pytest.approx(3.0)
        
        # Test with calc_index
        electron_0 = db.get_electron_number('Atom_1', calc_index=0)
        assert isinstance(electron_0, float)
        assert electron_0 == pytest.approx(2.9)
    
    def test_get_all_atoms(self, sample_json_file):
        """Test getting electron numbers and moments for all atoms."""
        db = DataBank.from_json(sample_json_file)
        
        # Test electron numbers
        electron_nums = db.get_electron_number()
        assert isinstance(electron_nums, dict)
        assert set(electron_nums.keys()) == {'Atom_1', 'Atom_2'}
        assert len(electron_nums['Atom_1']) == 2
        assert len(electron_nums['Atom_2']) == 2
        
        # Test magnetic moments
        moments = db.get_magnetic_moment()
        assert isinstance(moments, dict)
        assert set(moments.keys()) == {'Atom_1', 'Atom_2'}
        assert len(moments['Atom_1']) == 2
        assert len(moments['Atom_2']) == 2
        
        # Test with calc_index - should return dict
        electron_nums_0 = db.get_electron_number(calc_index=0)
        assert isinstance(electron_nums_0, dict)
        assert set(electron_nums_0.keys()) == {'Atom_1', 'Atom_2'}
        
        moments_0 = db.get_magnetic_moment(calc_index=0)
        assert isinstance(moments_0, dict)
        assert set(moments_0.keys()) == {'Atom_1', 'Atom_2'}
    
    def test_to_dataframe_with_precomputed(self, sample_json_file):
        """Test to_dataframe with precomputed electron numbers and moments."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")
        
        # Create DataBank with precomputed values
        db = DataBank.from_json(sample_json_file, include_electron_number=True, include_moment=True)
        df = db.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert 'electron_number_Atom_1' in df.columns
        assert 'electron_number_Atom_2' in df.columns
        assert 'moment_Atom_1' in df.columns
        assert 'moment_Atom_2' in df.columns
        assert len(df) == 2
        
        # Test without precomputed values
        db_no_precompute = DataBank.from_json(sample_json_file)
        df_no_precompute = db_no_precompute.to_dataframe()
        
        assert 'electron_number_Atom_1' not in df_no_precompute.columns
        assert 'moment_Atom_1' not in df_no_precompute.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
