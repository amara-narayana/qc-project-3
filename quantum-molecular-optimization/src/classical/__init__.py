"""Classical reference energy calculations using PySCF.

This module provides functions to compute reference energies for molecular
systems using classical quantum chemistry methods. These reference energies
are used to validate VQE results.
"""

from src.classical.reference_energy import (
    compute_hartree_fock_energy,
    compute_fci_energy,
    compute_cisd_energy,
    compute_reference_energy,
)

__all__ = [
    "compute_hartree_fock_energy",
    "compute_fci_energy",
    "compute_cisd_energy",
    "compute_reference_energy",
]

