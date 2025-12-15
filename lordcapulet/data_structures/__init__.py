"""
Data structures for LordCapulet.

This module provides core data structures for handling occupation matrices
and collections of calculation data.
"""

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
    'filter_atoms_by_species',
]
