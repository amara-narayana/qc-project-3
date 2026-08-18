"""Molecular system definitions for quantum chemistry calculations.

This module provides classes and functions to define molecular systems
and compute their properties using classical methods (PySCF) as a
reference for VQE calculations.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Molecule:
    """Represents a molecular system for quantum chemistry calculations.
    
    Attributes:
        name: Molecular formula (e.g., "H2", "LiH", "BeH2").
        atoms: List of (element, coordinates) tuples.
        charge: Total molecular charge.
        multiplicity: Spin multiplicity (2S+1 where S is total spin).
        basis: Basis set name (e.g., "sto-3g", "6-31g").
    """
    
    name: str
    atoms: list[tuple[str, tuple[float, float, float]]] = field(default_factory=list)
    charge: int = 0
    multiplicity: int = 1
    basis: str = "sto-3g"
    
    def __post_init__(self):
        """Validate molecule after initialization."""
        if not self.atoms:
            raise ValueError(f"Molecule {self.name} must have at least one atom")
    
    @classmethod
    def h2(cls, bond_distance: float = 0.74, basis: str = "sto-3g") -> 'Molecule':
        """Create H2 molecule with specified bond distance.
        
        Args:
            bond_distance: H-H bond distance in Angstroms.
            basis: Basis set name.
            
        Returns:
            H2 Molecule instance.
            
        Note:
            The default bond distance of 0.74 Å is close to the 
            experimental equilibrium bond length of H2.
        """
        # Place H atoms along z-axis, centered at origin
        half_dist = bond_distance / 2.0
        atoms = [
            ("H", (0.0, 0.0, -half_dist)),
            ("H", (0.0, 0.0, half_dist)),
        ]
        return cls(
            name="H2",
            atoms=atoms,
            charge=0,
            multiplicity=1,  # Singlet ground state
            basis=basis,
        )
    
    @classmethod
    def lih(cls, bond_distance: float = 1.60, basis: str = "sto-3g") -> 'Molecule':
        """Create LiH molecule with specified bond distance.
        
        Args:
            bond_distance: Li-H bond distance in Angstroms.
            basis: Basis set name.
            
        Returns:
            LiH Molecule instance.
            
        Note:
            The default bond distance of 1.60 Å is close to the 
            experimental equilibrium bond length of LiH (~1.595 Å).
        """
        # Place Li at origin, H along z-axis
        atoms = [
            ("Li", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, bond_distance)),
        ]
        return cls(
            name="LiH",
            atoms=atoms,
            charge=0,
            multiplicity=1,  # Singlet ground state
            basis=basis,
        )
    
    @classmethod
    def beh2(cls, bond_distance: float = 1.33, basis: str = "sto-3g") -> 'Molecule':
        """Create BeH2 molecule with specified bond distance.
        
        Args:
            bond_distance: Be-H bond distance in Angstroms.
            basis: Basis set name.
            
        Returns:
            BeH2 Molecule instance.
            
        Note:
            BeH2 is linear in its ground state. The default bond 
            distance of 1.33 Å is close to the experimental value.
            WARNING: This molecule requires more qubits and may be
            computationally expensive for VQE simulation.
        """
        # Linear molecule: H-Be-H along z-axis
        atoms = [
            ("H", (0.0, 0.0, -bond_distance)),
            ("Be", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, bond_distance)),
        ]
        return cls(
            name="BeH2",
            atoms=atoms,
            charge=0,
            multiplicity=1,  # Singlet ground state
            basis=basis,
        )
    
    @property
    def num_atoms(self) -> int:
        """Return number of atoms in molecule."""
        return len(self.atoms)
    
    @property
    def num_electrons(self) -> int:
        """Return total number of electrons.
        
        Returns:
            Total electron count based on atomic numbers minus charge.
        """
        atomic_numbers = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, 
                         "N": 7, "O": 8, "F": 9, "Ne": 10}
        total = sum(atomic_numbers.get(element, 0) for element, _ in self.atoms)
        return total - self.charge
    
    def to_pyscf_geometry(self) -> list[tuple[str, tuple[float, float, float]]]:
        """Convert to PySCF-compatible geometry format.
        
        Returns:
            List of (element, (x, y, z)) tuples for PySCF.
        """
        return self.atoms.copy()
    
    def __str__(self) -> str:
        atoms_str = ", ".join([f"{elem}" for elem, _ in self.atoms])
        return f"{self.name}({self.basis}): [{atoms_str}], charge={self.charge}, mult={self.multiplicity}"


def create_molecule(name: str, bond_distance: float = 0.74, basis: str = "sto-3g") -> Molecule:
    """Factory function to create molecules by name.
    
    Args:
        name: Molecule name ("H2", "LiH", "BeH2").
        bond_distance: Bond distance in Angstroms.
        basis: Basis set name.
        
    Returns:
        Molecule instance.
        
    Raises:
        ValueError: If molecule name is not recognized.
    """
    molecules = {
        "H2": Molecule.h2,
        "LIH": Molecule.lih,
        "BEH2": Molecule.beh2,
    }
    
    name_upper = name.upper()
    if name_upper not in molecules:
        raise ValueError(
            f"Unknown molecule: {name}. Available: {list(molecules.keys())}"
        )
    
    return molecules[name_upper](bond_distance=bond_distance, basis=basis)
