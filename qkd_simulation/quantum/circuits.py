# Functions to build quantum circuits (Alice, Bob, Eve)
# ./qkd_simulation/quantum/circuits.py

from qiskit import QuantumCircuit
import numpy as np
from typing import List

# Using the same convention as in bb84.py for clarity
# RECTILINEAR_BASIS = 0  # Z-basis (|0>, |1>)
# DIAGONAL_BASIS = 1     # X-basis (|+>, |->)

def create_bb84_circuits(alice_bits: np.ndarray,
                         alice_bases: np.ndarray,
                         bob_bases: np.ndarray) -> List[QuantumCircuit]:
    """
    Creates a list of QuantumCircuits for the BB84 protocol simulation.

    Each circuit represents one qubit transmission from Alice to Bob.
    - Alice prepares the qubit based on her bit and basis.
    - Bob measures the qubit based on his basis.

    Args:
        alice_bits: Numpy array of Alice's random bits (0 or 1).
        alice_bases: Numpy array of Alice's random bases (0 for Z, 1 for X).
        bob_bases: Numpy array of Bob's random measurement bases (0 for Z, 1 for X).

    Returns:
        A list of QuantumCircuit objects, one for each bit/qubit.

    Raises:
        ValueError: If the lengths of the input arrays do not match.
    """
    n = len(alice_bits)
    if not (n == len(alice_bases) == len(bob_bases)):
        raise ValueError("Input arrays (bits, alice_bases, bob_bases) must have the same length.")

    circuits = []
    for i in range(n):
        # Create a circuit with 1 qubit and 1 classical bit
        qc = QuantumCircuit(1, 1, name=f"qbit_{i}")

        # --- Step 1: Alice prepares the qubit ---
        # Apply X gate if bit is 1
        if alice_bits[i] == 1:
            qc.x(0) # Flip |0> to |1>

        # Apply H gate if basis is X (Diagonal)
        if alice_bases[i] == 1: # 1 corresponds to X basis
            qc.h(0) # Transform |0> to |+>, |1> to |->

        # Add a barrier for visual separation (optional)
        qc.barrier()

        # --- Step 2: Bob measures the qubit ---
        # Apply H gate if Bob's basis is X (Diagonal) before measurement
        if bob_bases[i] == 1: # 1 corresponds to X basis
            qc.h(0) # Transform |+>/|-> basis back to Z basis for measurement

        # Measure the qubit in the Z-basis (computational basis)
        qc.measure(0, 0) # Measure qubit 0 into classical bit 0

        circuits.append(qc)

    return circuits

# Example usage (for testing purposes)
if __name__ == '__main__':
    # Import functions from bb84 module for testing
    # Note: Adjust the path if running directly vs importing in main app
    try:
        from ..protocols.bb84 import generate_bits_and_bases, generate_bases
    except ImportError:
        # Fallback if running script directly (adjust path as needed)
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from qkd_simulation.protocols.bb84 import generate_bits_and_bases, generate_bases


    N = 5 # Number of qubits for the test
    print(f"Creating BB84 circuits for {N} qubits:")

    alice_bits_test, alice_bases_test = generate_bits_and_bases(N)
    bob_bases_test = generate_bases(N)

    print(f"Alice Bits:  {alice_bits_test}")
    print(f"Alice Bases: {alice_bases_test} (0=Z, 1=X)")
    print(f"Bob Bases:   {bob_bases_test} (0=Z, 1=X)")

    bb84_circs = create_bb84_circuits(alice_bits_test, alice_bases_test, bob_bases_test)

    print("\nExample Circuits:")
    if bb84_circs:
        print("\nCircuit 0:")
        print(bb84_circs[0].draw(output='text'))

        if len(bb84_circs) > 1:
            print("\nCircuit 1:")
            print(bb84_circs[1].draw(output='text'))
    else:
        print("No circuits generated.")

    # --- Explanation of a potential circuit ---
    # Example: Alice bit=1, Alice basis=X(1), Bob basis=X(1)
    # 1. Alice bit 1 -> qc.x(0) -> state is |1>
    # 2. Alice basis X -> qc.h(0) -> state is |->
    # 3. Bob basis X -> qc.h(0) -> state transforms back to |1>
    # 4. Bob measure -> qc.measure(0,0) -> result should be 1 with high probability
    #
    # Example: Alice bit=0, Alice basis=X(1), Bob basis=Z(0)
    # 1. Alice bit 0 -> (no X gate) -> state is |0>
    # 2. Alice basis X -> qc.h(0) -> state is |+>
    # 3. Bob basis Z -> (no H gate)
    # 4. Bob measure -> qc.measure(0,0) -> measuring |+> in Z basis gives 0 or 1 with 50% probability