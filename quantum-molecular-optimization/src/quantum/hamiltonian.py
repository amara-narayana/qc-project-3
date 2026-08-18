"""Molecular Hamiltonian construction using Qiskit Nature.

This module handles the construction of molecular Hamiltonians for quantum
chemistry calculations. It converts the electronic structure problem into
a qubit Hamiltonian suitable for VQE.

Key concepts:
-------------
1. Electronic Hamiltonian:
   H = Σ h_ij a†_i a_j + 1/2 Σ h_ijkl a†_i a†_j a_k a_l
   
   where a† and a are fermionic creation/annihilation operators.

2. Fermionic to qubit mapping:
   We use mappings like Jordan-Wigner or Parity to convert fermionic
   operators to Pauli operators acting on qubits.

3. Qubit Hamiltonian:
   H = Σ c_i P_i
   
   where P_i are tensor products of Pauli operators (I, X, Y, Z).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    ParityMapper,
    BravyiKitaevMapper,
)
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.operators import FermionicOp


@dataclass
class MolecularHamiltonian:
    """Represents a molecular Hamiltonian ready for VQE.
    
    Attributes:
        qubit_op: Qubit Hamiltonian as SparsePauliOp.
        num_qubits: Number of qubits required.
        num_particles: Number of electrons (alpha, beta).
        hf_energy: Hartree-Fock reference energy.
        nuclear_repulsion: Nuclear repulsion energy.
        mapper: Name of the fermion-to-qubit mapping used.
    """
    
    qubit_op: SparsePauliOp
    num_qubits: int
    num_particles: tuple[int, int]
    hf_energy: float
    nuclear_repulsion: float
    mapper: str
    
    @property
    def total_shift(self) -> float:
        """Return total energy shift (nuclear repulsion)."""
        return self.nuclear_repulsion
    
    def get_energy_shift(self) -> float:
        """Get the energy shift to add to measured expectation values."""
        return self.nuclear_repulsion


def create_fermionic_hamiltonian(
    atom_coords: list[tuple[str, tuple[float, float, float]]],
    basis: str = "sto3g",
    charge: int = 0,
    multiplicity: int = 1,
) -> tuple[FermionicOp, dict]:
    """Create fermionic Hamiltonian from molecular geometry.
    
    This function uses PySCF through Qiskit Nature's driver to compute
    the one- and two-body integrals needed for the second-quantized
    fermionic Hamiltonian.
    
    Args:
        atom_coords: List of (element, (x, y, z)) tuples.
        basis: Basis set name (e.g., "sto3g", "631g").
        charge: Total molecular charge.
        multiplicity: Spin multiplicity (2S+1).
        
    Returns:
        Tuple of (fermionic_operator, auxiliary data).
        
    The fermionic operator has the form:
        H = Σ h_pq a†_p a_q + 1/2 Σ h_pqrs a†_p a†_q a_r a_s
        
    where h_pq and h_pqrs are one- and two-body integrals.
    """
    # Format atom coordinates for Qiskit Nature
    # Qiskit Nature expects string format: "H 0.0 0.0 -0.37; H 0.0 0.0 0.37"
    atoms_formatted = []
    for element, coords in atom_coords:
        x, y, z = coords
        atoms_formatted.append(f"{element} {x} {y} {z}")
    atom_string = ";".join(atoms_formatted)
    
    # Create PySCF driver
    driver = PySCFDriver(
        atom=atom_string,
        unit=DistanceUnit.ANGSTROM,
        basis=basis,
        spin=multiplicity - 1,  # Qiskit uses 2S, not 2S+1
        charge=charge,
    )
    
    # Run driver to get problem
    problem = driver.run()
    
    # Get fermionic operator (second quantization)
    fermionic_op = problem.hamiltonian.second_q_op()
    
    # Auxiliary data for later use
    aux_data = {
        'num_particles': problem.num_particles,
        'hf_energy': problem.reference_energy,
        'nuclear_repulsion': problem.nuclear_repulsion_energy,
        'problem': problem,
    }
    
    return fermionic_op, aux_data


def map_to_qubits(
    fermionic_op: FermionicOp,
    mapper_name: str = "jordan_wigner",
) -> SparsePauliOp:
    """Map fermionic operator to qubit operator.
    
    Args:
        fermionic_op: Fermionic second-quantized operator.
        mapper_name: Name of mapping ("jordan_wigner", "parity", "bravyi_kitaev").
        
    Returns:
        Qubit operator as SparsePauliOp.
        
    Mappings:
    ---------
    - Jordan-Wigner: Simple but creates long Pauli strings (O(n) length).
    - Parity: Can reduce qubits by exploiting symmetries.
    - Bravyi-Kitaev: Better scaling (O(log n) Pauli string length).
    """
    mappers = {
        "jordan_wigner": JordanWignerMapper(),
        "parity": ParityMapper(),
        "bravyi_kitaev": BravyiKitaevMapper(),
    }
    
    if mapper_name.lower() not in mappers:
        raise ValueError(
            f"Unknown mapper: {mapper_name}. "
            f"Available: {list(mappers.keys())}"
        )
    
    mapper = mappers[mapper_name.lower()]
    qubit_op = mapper.map(fermionic_op)
    
    # Simplify (combine coefficients for same Pauli terms)
    qubit_op = qubit_op.simplify()
    
    return qubit_op


def build_molecular_hamiltonian(
    atom_coords: list[tuple[str, tuple[float, float, float]]],
    basis: str = "sto3g",
    charge: int = 0,
    multiplicity: int = 1,
    mapper: str = "jordan_wigner",
) -> MolecularHamiltonian:
    """Build complete molecular Hamiltonian for VQE.
    
    This is the main entry point for creating a Hamiltonian. It:
    1. Computes molecular integrals using PySCF
    2. Constructs fermionic Hamiltonian in second quantization
    3. Maps to qubit Hamiltonian using specified mapping
    
    Args:
        atom_coords: List of (element, (x, y, z)) tuples.
        basis: Basis set name.
        charge: Total molecular charge.
        multiplicity: Spin multiplicity.
        mapper: Fermion-to-qubit mapping method.
        
    Returns:
        MolecularHamiltonian object containing all necessary information.
        
    Example:
        >>> ham = build_molecular_hamiltonian(
        ...     [("H", (0, 0, -0.37)), ("H", (0, 0, 0.37))],
        ...     basis="sto3g"
        ... )
        >>> print(f"Number of qubits: {ham.num_qubits}")
        >>> print(f"HF energy: {ham.hf_energy:.6f} Ha")
    """
    # Create fermionic Hamiltonian
    fermionic_op, aux_data = create_fermionic_hamiltonian(
        atom_coords, basis, charge, multiplicity
    )
    
    # Map to qubits
    qubit_op = map_to_qubits(fermionic_op, mapper)
    
    return MolecularHamiltonian(
        qubit_op=qubit_op,
        num_qubits=qubit_op.num_qubits,
        num_particles=aux_data['num_particles'],
        hf_energy=aux_data['hf_energy'],
        nuclear_repulsion=aux_data['nuclear_repulsion'],
        mapper=mapper,
    )


def get_hamiltonian_info(ham: MolecularHamiltonian) -> dict:
    """Extract detailed information about the Hamiltonian.
    
    Args:
        ham: MolecularHamiltonian object.
        
    Returns:
        Dictionary with Hamiltonian statistics.
    """
    # Count non-zero terms (size of SparsePauliOp)
    num_terms = len(ham.qubit_op)
    
    # Get Pauli term types
    pauli_types = {}
    for i in range(len(ham.qubit_op)):
        pauli_str = ham.qubit_op.paulis[i].to_label()
        pauli_types[pauli_str] = pauli_types.get(pauli_str, 0) + 1
    
    return {
        'num_qubits': ham.num_qubits,
        'num_terms': num_terms,
        'num_particles': ham.num_particles,
        'hf_energy': ham.hf_energy,
        'nuclear_repulsion': ham.nuclear_repulsion,
        'mapper': ham.mapper,
        'pauli_term_distribution': pauli_types,
    }


if __name__ == "__main__":
    # Test with H2 molecule at equilibrium distance
    print("Building H2 Hamiltonian (R = 0.74 Å, STO-3G basis)")
    print("=" * 60)
    
    h2_coords = [
        ("H", (0.0, 0.0, -0.37)),
        ("H", (0.0, 0.0, 0.37)),
    ]
    
    ham = build_molecular_hamiltonian(h2_coords, basis="sto3g")
    info = get_hamiltonian_info(ham)
    
    print(f"Number of qubits: {info['num_qubits']}")
    print(f"Number of Pauli terms: {info['num_terms']}")
    print(f"Number of particles: {info['num_particles']}")
    print(f"Hartree-Fock energy: {info['hf_energy']:.8f} Ha")
    print(f"Nuclear repulsion: {info['nuclear_repulsion']:.8f} Ha")
    print(f"Mapping: {info['mapper']}")
    print("\nFirst few Pauli terms:")
    for i in range(min(5, len(ham.qubit_op))):
        label = ham.qubit_op.paulis[i].to_label()
        coeff = ham.qubit_op.coeffs[i]
        print(f"  {coeff.real:+.6f} * {label}")
