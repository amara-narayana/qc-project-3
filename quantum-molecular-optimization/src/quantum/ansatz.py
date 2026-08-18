"""Variational quantum ansatz circuits for VQE.

This module provides different types of ansatz circuits used in VQE:

1. EfficientSU2: Hardware-efficient ansatz with alternating rotation 
   and entanglement layers. Good for near-term devices.

2. UCCSD: Unitary Coupled Cluster Singles and Doubles. Physics-inspired
   ansatz that prepares states close to the exact ground state.

3. TwoLocal: General parameterized circuit with configurable structure.

Key concepts:
-------------
An ansatz is a parameterized quantum circuit: |ψ(θ)> = U(θ)|0>

The goal is to find parameters θ* that minimize:
    E(θ) = <ψ(θ)|H|ψ(θ)>

Different ansätze trade off:
- Expressibility: Can it represent the target state?
- Trainability: Is the optimization landscape smooth?
- Circuit depth: How many gates are needed?
- Hardware compatibility: Does it match device connectivity?
"""

from typing import Optional, Literal

from qiskit.circuit.library import EfficientSU2, TwoLocal
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem


def create_efficient_su2_ansatz(
    num_qubits: int,
    reps: int = 1,
    entanglement: str = "linear",
    initial_state: Optional[str] = None,
) -> EfficientSU2:
    """Create EfficientSU2 ansatz circuit.
    
    The EfficientSU2 ansatz consists of alternating layers of:
    1. Single-qubit Y and Z rotations on all qubits
    2. Entangling gates (CNOT or CZ) between qubits
    
    This ansatz is hardware-efficient because it uses native gates
    and respects device connectivity constraints.
    
    Args:
        num_qubits: Number of qubits in the circuit.
        reps: Number of repetitions of the rotation+entanglement block.
        entanglement: Entanglement pattern ("linear", "full", "circular").
        initial_state: Initial state preparation ("zero" or None).
        
    Returns:
        Parameterized EfficientSU2 quantum circuit.
        
    Example:
        >>> ansatz = create_efficient_su2_ansatz(4, reps=2)
        >>> print(f"Number of parameters: {ansatz.num_parameters}")
    """
    # Create EfficientSU2 ansatz
    ansatz = EfficientSU2(
        num_qubits=num_qubits,
        reps=reps,
        entanglement=entanglement,
        initial_state=None if initial_state == "zero" else None,
    )
    
    return ansatz


def create_uccsd_ansatz(
    problem: ElectronicStructureProblem,
    mapper: str = "jordan_wigner",
    reps: int = 1,
) -> UCCSD:
    """Create UCCSD (Unitary Coupled Cluster Singles and Doubles) ansatz.
    
    UCCSD is a physics-inspired ansatz based on the coupled cluster method:
        |ψ(θ)> = exp(T - T†)|Φ_HF>
    
    where T is the cluster operator containing single and double excitations:
        T = Σ t_i^a a†_a a_i + Σ t_ij^ab a†_a a†_b a_j a_i
    
    UCCSD typically requires fewer parameters than EfficientSU2 for 
    chemistry problems but creates deeper circuits.
    
    Args:
        problem: Electronic structure problem from Qiskit Nature.
        mapper: Fermion-to-qubit mapping method.
        reps: Number of repetitions (usually 1 for UCCSD).
        
    Returns:
        Parameterized UCCSD quantum circuit.
        
    Note:
        UCCSD is more chemically accurate but may be too deep for
        noisy quantum devices. Use EfficientSU2 for NISQ experiments.
    """
    # Get mapper object
    mappers = {
        "jordan_wigner": JordanWignerMapper(),
    }
    qubit_mapper = mappers.get(mapper.lower(), JordanWignerMapper())
    
    # Create UCCSD ansatz
    ansatz = UCCSD(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        qubit_mapper=qubit_mapper,
        reps=reps,
        generalized=True,  # Include all excitations
    )
    
    return ansatz


def create_two_local_ansatz(
    num_qubits: int,
    rotation_blocks: str = "ry",
    entanglement_blocks: str = "cx",
    entanglement: str = "linear",
    reps: int = 1,
) -> TwoLocal:
    """Create TwoLocal ansatz circuit.
    
    TwoLocal is a flexible parameterized circuit builder that alternates
    between layers of single-qubit rotations and two-qubit entangling gates.
    
    Args:
        num_qubits: Number of qubits.
        rotation_blocks: Single-qubit rotation gate(s) ("rx", "ry", "rz", "h").
        entanglement_blocks: Two-qubit gate ("cx", "cz", "swap").
        entanglement: Connectivity pattern.
        reps: Number of repetitions.
        
    Returns:
        Parameterized TwoLocal quantum circuit.
    """
    ansatz = TwoLocal(
        num_qubits=num_qubits,
        rotation_blocks=rotation_blocks,
        entanglement_blocks=entanglement_blocks,
        entanglement=entanglement,
        reps=reps,
        insert_barriers=True,
    )
    
    return ansatz


def create_ansatz(
    ansatz_type: str,
    num_qubits: int,
    problem: Optional[ElectronicStructureProblem] = None,
    reps: int = 1,
    entanglement: str = "linear",
    **kwargs,
) -> object:
    """Factory function to create ansatz circuits.
    
    Args:
        ansatz_type: Type of ansatz ("EfficientSU2", "UCCSD", "TwoLocal").
        num_qubits: Number of qubits required.
        problem: Electronic structure problem (needed for UCCSD).
        reps: Number of repetitions.
        entanglement: Entanglement pattern.
        **kwargs: Additional arguments passed to specific ansatz creators.
        
    Returns:
        Parameterized quantum circuit.
        
    Raises:
        ValueError: If unknown ansatz type is specified.
    """
    ansatz_creators = {
        "efficientsu2": lambda: create_efficient_su2_ansatz(
            num_qubits, reps, entanglement, kwargs.get("initial_state")
        ),
        "twolocal": lambda: create_two_local_ansatz(
            num_qubits,
            rotation_blocks=kwargs.get("rotation_blocks", "ry"),
            entanglement_blocks=kwargs.get("entanglement_blocks", "cx"),
            entanglement=entanglement,
            reps=reps,
        ),
    }
    
    # UCCSD requires problem instance
    if ansatz_type.lower() == "uccsd":
        if problem is None:
            raise ValueError("UCCSD ansatz requires an ElectronicStructureProblem")
        return create_uccsd_ansatz(
            problem,
            mapper=kwargs.get("mapper", "jordan_wigner"),
            reps=reps,
        )
    
    ansatz_lower = ansatz_type.lower().replace("_", "").replace("-", "")
    if ansatz_lower not in ansatz_creators:
        raise ValueError(
            f"Unknown ansatz type: {ansatz_type}. "
            f"Available: {list(ansatz_creators.keys()) + ['uccsd']}"
        )
    
    return ansatz_creators[ansatz_lower]()


def get_ansatz_info(ansatz) -> dict:
    """Extract information about an ansatz circuit.
    
    Args:
        ansatz: Parameterized quantum circuit.
        
    Returns:
        Dictionary with ansatz statistics.
    """
    return {
        'num_qubits': ansatz.num_qubits,
        'num_parameters': ansatz.num_parameters,
        'depth': ansatz.depth(),
        'num_gates': ansatz.size(),
        'type': type(ansatz).__name__,
    }


if __name__ == "__main__":
    # Test ansatz creation
    print("Testing Ansatz Circuits")
    print("=" * 60)
    
    # Test EfficientSU2
    print("\n1. EfficientSU2 Ansatz (4 qubits, 2 reps):")
    eff_su2 = create_efficient_su2_ansatz(4, reps=2, entanglement="linear")
    info = get_ansatz_info(eff_su2)
    print(f"   Qubits: {info['num_qubits']}")
    print(f"   Parameters: {info['num_parameters']}")
    print(f"   Depth: {info['depth']}")
    print(f"   Gates: {info['num_gates']}")
    
    # Test TwoLocal
    print("\n2. TwoLocal Ansatz (4 qubits, 2 reps):")
    two_local = create_two_local_ansatz(4, reps=2)
    info = get_ansatz_info(two_local)
    print(f"   Qubits: {info['num_qubits']}")
    print(f"   Parameters: {info['num_parameters']}")
    print(f"   Depth: {info['depth']}")
    print(f"   Gates: {info['num_gates']}")
    
    print("\n✓ Ansatz tests completed successfully!")
