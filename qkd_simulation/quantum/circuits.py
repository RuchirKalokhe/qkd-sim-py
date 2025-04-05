from qiskit import QuantumCircuit
import numpy as np
from typing import List, Optional, Tuple, Dict, Union

def create_bb84_circuits(alice_bits: np.ndarray,
                         alice_bases: np.ndarray,
                         bob_bases: np.ndarray) -> List[QuantumCircuit]:
    """
    Create quantum circuits for the BB84 protocol simulation.
    
    This function creates one quantum circuit for each qubit in the BB84 protocol:
    1. Alice prepares qubits according to her bits and bases
    2. Bob measures qubits according to his bases
    
    Args:
        alice_bits: Array of Alice's random bits (0 or 1)
        alice_bases: Array of Alice's random bases (0 for Z-basis, 1 for X-basis)
        bob_bases: Array of Bob's random bases (0 for Z-basis, 1 for X-basis)
        
    Returns:
        List of quantum circuits, one for each qubit
        
    Raises:
        ValueError: If input arrays have different lengths
    """
    n = len(alice_bits)
    if not (n == len(alice_bases) == len(bob_bases)):
        raise ValueError("Input arrays (bits, alice_bases, bob_bases) must have the same length.")

    circuits = []
    for i in range(n):
        qc = QuantumCircuit(1, 1, name=f"qbit_{i}")

        # Alice's preparation
        if alice_bits[i] == 1:
            qc.x(0)  # |1⟩ state

        if alice_bases[i] == 1:
            qc.h(0)  # Convert to X-basis if diagonal basis selected

        qc.barrier()  # Visual separation between Alice and Bob

        # Bob's measurement
        if bob_bases[i] == 1:
            qc.h(0)  # Convert from X-basis to Z-basis for measurement

        qc.measure(0, 0)

        circuits.append(qc)
    
    return circuits

def create_bb84_circuits_with_eve(alice_bits: np.ndarray,
                                 alice_bases: np.ndarray,
                                 eve_bases: np.ndarray,
                                 bob_bases: np.ndarray) -> List[QuantumCircuit]:
    """
    Create quantum circuits for the BB84 protocol with Eve's intercept-resend attack.
    
    This function creates one quantum circuit for each qubit in the BB84 protocol:
    1. Alice prepares qubits according to her bits and bases
    2. Eve measures qubits according to her bases and resends
    3. Bob measures qubits according to his bases
    
    Args:
        alice_bits: Array of Alice's random bits (0 or 1)
        alice_bases: Array of Alice's random bases (0 for Z-basis, 1 for X-basis)
        eve_bases: Array of Eve's random bases (0 for Z-basis, 1 for X-basis)
        bob_bases: Array of Bob's random bases (0 for Z-basis, 1 for X-basis)
        
    Returns:
        List of quantum circuits, one for each qubit
        
    Raises:
        ValueError: If input arrays have different lengths
    """
    n = len(alice_bits)
    if not (n == len(alice_bases) == len(eve_bases) == len(bob_bases)):
        raise ValueError("All input arrays must have the same length.")

    circuits = []
    for i in range(n):
        # Create circuit with 1 qubit and 2 classical bits (for Eve and Bob)
        qc = QuantumCircuit(1, 2, name=f"qbit_eve_{i}")

        # Alice's preparation
        if alice_bits[i] == 1:
            qc.x(0)  # |1⟩ state

        if alice_bases[i] == 1:
            qc.h(0)  # Convert to X-basis if diagonal basis selected

        qc.barrier()  # Visual separation between Alice and Eve

        # Eve's measurement and resend
        if eve_bases[i] == 1:
            qc.h(0)  # Convert from X-basis to Z-basis for measurement

        # Eve measures and stores result in classical bit 0
        qc.measure(0, 0)
        
        # Eve prepares new qubit based on her measurement
        # Reset qubit to |0⟩
        qc.reset(0)
        
        # Set to |1⟩ if Eve measured 1
        qc.x(0).c_if(0, 1)
        
        # Apply H if Eve used diagonal basis
        if eve_bases[i] == 1:
            qc.h(0)

        qc.barrier()  # Visual separation between Eve and Bob

        # Bob's measurement
        if bob_bases[i] == 1:
            qc.h(0)  # Convert from X-basis to Z-basis for measurement

        # Bob measures and stores result in classical bit 1
        qc.measure(0, 1)

        circuits.append(qc)
    
    return circuits

def create_e91_circuits(num_pairs: int) -> Tuple[List[QuantumCircuit], np.ndarray]:
    """
    Create quantum circuits for the E91 (Ekert91) protocol simulation.
    
    This function creates circuits for entangled pairs where:
    1. Entangled Bell pairs are created
    2. Alice and Bob each measure one qubit of each pair with random bases
    
    Args:
        num_pairs: Number of entangled pairs to create
        
    Returns:
        Tuple of (circuits, bases) where:
        - circuits is a list of quantum circuits
        - bases is an array of shape (num_pairs, 2) containing Alice and Bob's random bases
        
    Raises:
        ValueError: If num_pairs is not positive
    """
    if num_pairs <= 0:
        raise ValueError("Number of pairs must be positive.")
    
    # Generate random measurement bases for Alice and Bob
    # For E91, we use more than 2 bases per party
    # Alice uses 3 bases: 0°, 45°, 90° (0, 1, 2)
    # Bob uses 3 bases: 45°, 90°, 135° (1, 2, 3)
    alice_bases = np.random.randint(0, 3, size=num_pairs)
    bob_bases = np.random.randint(1, 4, size=num_pairs)
    bases = np.column_stack((alice_bases, bob_bases))
    
    circuits = []
    for i in range(num_pairs):
        # Create circuit with 2 qubits (one for Alice, one for Bob) and 2 classical bits
        qc = QuantumCircuit(2, 2, name=f"e91_pair_{i}")
        
        # Create Bell pair (|00⟩ + |11⟩)/√2
        qc.h(0)
        qc.cx(0, 1)
        
        qc.barrier()
        
        # Alice's measurement basis
        if alice_bases[i] == 1:  # 45°
            qc.ry(-np.pi/4, 0)
        elif alice_bases[i] == 2:  # 90°
            qc.ry(-np.pi/2, 0)
        
        # Bob's measurement basis
        if bob_bases[i] == 1:  # 45°
            qc.ry(-np.pi/4, 1)
        elif bob_bases[i] == 2:  # 90°
            qc.ry(-np.pi/2, 1)
        elif bob_bases[i] == 3:  # 135°
            qc.ry(-3*np.pi/4, 1)
        
        # Measure both qubits
        qc.measure(0, 0)
        qc.measure(1, 1)
        
        circuits.append(qc)
    
    return circuits, bases

def visualize_circuit(circuit: QuantumCircuit, filename: Optional[str] = None) -> Optional[str]:
    """
    Generate a visualization of a quantum circuit.
    
    Args:
        circuit: The quantum circuit to visualize
        filename: Optional filename to save the visualization (must end with .png)
        
    Returns:
        Path to the saved image file if filename is provided, None otherwise
        
    Raises:
        ImportError: If matplotlib is not installed
        ValueError: If filename doesn't end with .png
    """
    try:
        from qiskit.visualization import circuit_drawer
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Visualization requires matplotlib. Install with 'pip install matplotlib'")
    
    if filename and not filename.endswith('.png'):
        raise ValueError("Filename must end with .png")
    
    fig = plt.figure(figsize=(10, 6))
    circuit_drawer(circuit, output='mpl', style={'backgroundcolor': '#FFFFFF'})
    
    if filename:
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return filename
    else:
        plt.close(fig)
        return None

def optimize_circuits(circuits: List[QuantumCircuit], 
                     optimization_level: int = 1) -> List[QuantumCircuit]:
    """
    Optimize quantum circuits using Qiskit's transpiler.
    
    Args:
        circuits: List of quantum circuits to optimize
        optimization_level: Optimization level (0-3, higher means more optimization)
        
    Returns:
        List of optimized quantum circuits
        
    Raises:
        ValueError: If optimization_level is not in range 0-3
    """
    if not 0 <= optimization_level <= 3:
        raise ValueError("Optimization level must be between 0 and 3")
    
    try:
        from qiskit import transpile
    except ImportError:
        raise ImportError("Circuit optimization requires qiskit. Install with 'pip install qiskit'")
    
    return transpile(circuits, optimization_level=optimization_level)
