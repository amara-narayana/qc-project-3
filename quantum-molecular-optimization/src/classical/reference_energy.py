"""Classical reference energy calculations using PySCF.

This module provides functions to compute reference energies for molecular
systems using classical quantum chemistry methods. These reference energies
are used to validate VQE results.

Methods implemented:
- Hartree-Fock (HF): Mean-field approximation
- Full Configuration Interaction (FCI): Exact solution within basis set
- Configuration Interaction Singles Doubles (CISD): Approximate correlation
"""

import numpy as np
from pyscf import gto, scf, fci, ci


def compute_hartree_fock_energy(
    atoms: list[tuple[str, tuple[float, float, float]]],
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
) -> float:
    """Compute Hartree-Fock energy for a molecular system.
    
    Hartree-Fock theory is a mean-field approximation that treats each
    electron as moving in the average field created by all other electrons.
    It neglects electron correlation (instantaneous electron-electron
    interactions), so HF energy is typically higher than the exact energy.
    
    Args:
        atoms: List of (element, (x, y, z)) tuples defining molecular geometry.
        basis: Basis set name (e.g., "sto-3g", "6-31g").
        charge: Total molecular charge.
        multiplicity: Spin multiplicity (2S+1).
        
    Returns:
        Hartree-Fock energy in Hartree (atomic units).
        
    Note:
        1 Hartree ≈ 27.2114 eV ≈ 627.51 kcal/mol
    """
    # Create PySCF molecule object
    mol = gto.Mole()
    mol.atom = atoms
    mol.basis = basis
    mol.charge = charge
    mol.spin = multiplicity - 1  # PySCF uses 2S, not 2S+1
    mol.build()
    
    # Run Hartree-Fock calculation
    mf = scf.RHF(mol)
    mf.verbose = 0
    energy = mf.kernel()
    
    return float(energy)


def compute_fci_energy(
    atoms: list[tuple[str, tuple[float, float, float]]],
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
) -> float:
    """Compute Full Configuration Interaction energy.
    
    FCI provides the exact solution to the electronic Schrödinger equation
    within the given basis set. It includes all possible electron excitations
    from occupied to virtual orbitals, capturing all electron correlation.
    
    WARNING: FCI scales factorially with system size and is only feasible
    for very small molecules (typically < 16 orbitals).
    
    Args:
        atoms: List of (element, (x, y, z)) tuples.
        basis: Basis set name.
        charge: Total molecular charge.
        multiplicity: Spin multiplicity.
        
    Returns:
        FCI energy in Hartree.
    """
    # Create PySCF molecule object
    mol = gto.Mole()
    mol.atom = atoms
    mol.basis = basis
    mol.charge = charge
    mol.spin = multiplicity - 1
    mol.build()
    
    # Run Hartree-Fock first to get orbitals
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    
    # Run FCI
    ci_solver = fci.FCI(mf)
    ci_solver.verbose = 0
    energy = ci_solver.kernel()[0]
    
    return float(energy)


def compute_cisd_energy(
    atoms: list[tuple[str, tuple[float, float, float]]],
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
) -> float:
    """Compute Configuration Interaction Singles and Doubles energy.
    
    CISD includes all single and double excitations from the Hartree-Fock
    reference. It captures a significant portion of electron correlation
    at lower cost than FCI, but is not variational (can be below exact energy).
    
    Args:
        atoms: List of (element, (x, y, z)) tuples.
        basis: Basis set name.
        charge: Total molecular charge.
        multiplicity: Spin multiplicity.
        
    Returns:
        CISD energy in Hartree.
    """
    # Create PySCF molecule object
    mol = gto.Mole()
    mol.atom = atoms
    mol.basis = basis
    mol.charge = charge
    mol.spin = multiplicity - 1
    mol.build()
    
    # Run Hartree-Fock first
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    
    # Run CISD
    ci_solver = ci.CISD(mf)
    ci_solver.verbose = 0
    energy = ci_solver.kernel()[0]
    
    return float(energy)


def compute_reference_energy(
    atoms: list[tuple[str, tuple[float, float, float]]],
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "fci",
) -> dict[str, float]:
    """Compute reference energies using multiple methods.
    
    This function computes energies using Hartree-Fock and optionally
    more accurate correlated methods for comparison with VQE results.
    
    Args:
        atoms: List of (element, (x, y, z)) tuples.
        basis: Basis set name.
        charge: Total molecular charge.
        multiplicity: Spin multiplicity.
        method: Reference method ("hf", "cisd", "fci").
        
    Returns:
        Dictionary with keys:
            - 'hartree_fock': HF energy
            - 'reference': Best available reference energy
            - 'method': Method used for reference
            
    Raises:
        ValueError: If unknown method is specified.
    """
    results = {}
    
    # Always compute Hartree-Fock
    results['hartree_fock'] = compute_hartree_fock_energy(
        atoms, basis, charge, multiplicity
    )
    
    # Compute correlated energy based on method
    if method.lower() == "hf":
        results['reference'] = results['hartree_fock']
        results['method'] = 'hartree_fock'
    elif method.lower() == "cisd":
        results['cisd'] = compute_cisd_energy(atoms, basis, charge, multiplicity)
        results['reference'] = results['cisd']
        results['method'] = 'cisd'
    elif method.lower() == "fci":
        try:
            results['fci'] = compute_fci_energy(atoms, basis, charge, multiplicity)
            results['reference'] = results['fci']
            results['method'] = 'fci'
        except Exception as e:
            # FCI may fail for larger systems, fall back to CISD
            print(f"FCI failed ({e}), falling back to CISD")
            results['cisd'] = compute_cisd_energy(atoms, basis, charge, multiplicity)
            results['reference'] = results['cisd']
            results['method'] = 'cisd'
    else:
        raise ValueError(f"Unknown reference method: {method}")
    
    return results


if __name__ == "__main__":
    # Test with H2 molecule
    h2_atoms = [("H", (0.0, 0.0, -0.37)), ("H", (0.0, 0.0, 0.37))]
    
    print("H2 molecule at R = 0.74 Å (STO-3G basis)")
    print("=" * 50)
    
    energies = compute_reference_energy(h2_atoms, basis="sto-3g", method="fci")
    
    print(f"Hartree-Fock energy: {energies['hartree_fock']:.8f} Ha")
    print(f"Reference ({energies['method']}) energy: {energies['reference']:.8f} Ha")
    print(f"Correlation energy: {energies['reference'] - energies['hartree_fock']:.8f} Ha")
