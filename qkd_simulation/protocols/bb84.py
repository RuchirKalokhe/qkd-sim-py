# ./qkd_simulation/protocols/bb84.py

import numpy as np
from typing import Tuple, List

# Define basis constants for clarity (optional but good practice)
RECTILINEAR_BASIS = 0  # Corresponds to Z-basis measurement (|0>, |1>)
DIAGONAL_BASIS = 1     # Corresponds to X-basis measurement (|+>, |->)

def generate_bits_and_bases(num_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates random bits and bases for Alice.

    Args:
        num_bits: The number of bits (and corresponding bases) to generate.

    Returns:
        A tuple containing:
            - bits (np.ndarray): An array of random bits (0 or 1).
            - bases (np.ndarray): An array of random bases (0 for Rectilinear/Z, 1 for Diagonal/X).
    """
    if num_bits <= 0:
        raise ValueError("Number of bits must be positive.")

    bits = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)
    bases = np.random.randint(0, 2, size=num_bits, dtype=np.uint8) # 0 for Z, 1 for X
    return bits, bases

def generate_bases(num_bits: int) -> np.ndarray:
    """
    Generates random measurement bases for Bob.

    Args:
        num_bits: The number of bases to generate.

    Returns:
        bases (np.ndarray): An array of random bases (0 for Rectilinear/Z, 1 for Diagonal/X).
    """
    if num_bits <= 0:
        raise ValueError("Number of bits must be positive.")

    bases = np.random.randint(0, 2, size=num_bits, dtype=np.uint8) # 0 for Z, 1 for X
    return bases

def compare_bases(alice_bases: np.ndarray, bob_bases: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Compares Alice's and Bob's bases to find where they match.
    This simulates the public discussion phase of BB84.

    Args:
        alice_bases: Numpy array of Alice's bases (0 or 1).
        bob_bases: Numpy array of Bob's bases (0 or 1).

    Returns:
        A tuple containing:
            - matching_indices (List[int]): A list of indices where the bases matched.
            - mismatching_indices (List[int]): A list of indices where the bases did not match.

    Raises:
        ValueError: If the lengths of the base arrays do not match.
    """
    if len(alice_bases) != len(bob_bases):
        raise ValueError("Alice's and Bob's base arrays must have the same length.")

    matches = (alice_bases == bob_bases)
    matching_indices = np.where(matches)[0].tolist()
    mismatching_indices = np.where(~matches)[0].tolist() # ~ is logical NOT

    return matching_indices, mismatching_indices

# Example usage (for testing purposes, not typically run from here)
if __name__ == '__main__':
    N = 10
    print(f"Simulating BB84 basis comparison for {N} bits:")

    alice_bits, alice_bases = generate_bits_and_bases(N)
    print(f"Alice's bits:  {alice_bits}")
    print(f"Alice's bases: {alice_bases} (0=Z, 1=X)")

    bob_bases = generate_bases(N)
    print(f"Bob's bases:   {bob_bases} (0=Z, 1=X)")

    matching_idx, mismatching_idx = compare_bases(alice_bases, bob_bases)
    print(f"\nIndices where bases matched:   {matching_idx}")
    print(f"Indices where bases mismatched: {mismatching_idx}")

    # Note: The actual sifted key would be constructed later using Alice's bits
    # and Bob's measurement results ONLY at the matching_indices.
    # QBER would be estimated by comparing a subset of Alice's bits and Bob's results
    # at the matching_indices.