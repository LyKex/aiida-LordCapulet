"""
Utility modules for LordCapulet.
"""

# Import utility functions when created
# from .structure_utils import tag_and_list_atoms
# from .magnetism_utils import create_magnetic_configs

from .occupation_matrix import (
    OccupationMatrixData, 
    OccupationMatrixAiidaData,
    extract_occupations_from_calc,
    filter_atoms_by_species
)

__all__ = [
    'OccupationMatrixData', 
    'OccupationMatrixAiidaData',
    'extract_occupations_from_calc',
    'filter_atoms_by_species'
]
