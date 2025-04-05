from qiskit import QuantumCircuit
import numpy as np
from typing import List

def create_bb84_circuits(alice_bits: np.ndarray,
                         alice_bases: np.ndarray,
                         bob_bases: np.ndarray) -> List[QuantumCircuit]:
    n = len(alice_bits)
    if not (n == len(alice_bases) == len(bob_bases)):
        raise ValueError("Input arrays (bits, alice_bases, bob_bases) must have the same length.")

    circuits = []
    for i in range(n):
        qc = QuantumCircuit(1, 1, name=f"qbit_{i}")

        if alice_bits[i] == 1:
            qc.x(0)

        if alice_bases[i] == 1:
            qc.h(0)

        qc.barrier()

        if bob_bases[i] == 1:
            qc.h(0)

        qc.measure(0, 0)

        circuits.append(qc)

    return circuits