import numpy as np
from typing import Tuple, List

RECTILINEAR_BASIS = 0
DIAGONAL_BASIS = 1

def generate_bits_and_bases(num_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    if num_bits <= 0:
        raise ValueError("Number of bits must be positive.")

    bits = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)
    bases = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)
    return bits, bases

def generate_bases(num_bits: int) -> np.ndarray:
    if num_bits <= 0:
        raise ValueError("Number of bits must be positive.")

    bases = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)
    return bases

def compare_bases(alice_bases: np.ndarray, bob_bases: np.ndarray) -> Tuple[List[int], List[int]]:
    if len(alice_bases) != len(bob_bases):
        raise ValueError("Alice's and Bob's base arrays must have the same length.")

    matches = (alice_bases == bob_bases)
    matching_indices = np.where(matches)[0].tolist()
    mismatching_indices = np.where(~matches)[0].tolist()

    return matching_indices, mismatching_indices