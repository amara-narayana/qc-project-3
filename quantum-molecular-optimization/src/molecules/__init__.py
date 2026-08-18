"""Molecular system definitions for quantum chemistry calculations.

This module provides classes and functions to define molecular systems
and compute their properties using classical methods (PySCF) as a
reference for VQE calculations.
"""

from src.molecules.molecule import Molecule, create_molecule

__all__ = ["Molecule", "create_molecule"]