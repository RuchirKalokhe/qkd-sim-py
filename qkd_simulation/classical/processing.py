import numpy as np
from typing import Tuple, List, Optional
import random

def sift_key(alice_bits: np.ndarray,
             bob_measured_bits: np.ndarray,
             matching_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    if len(alice_bits) != len(bob_measured_bits):
        raise ValueError("Alice's bits and Bob's measured bits must have the same length.")
    if not matching_indices:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.uint8)

    max_index = len(alice_bits) - 1
    if any(idx < 0 or idx > max_index for idx in matching_indices):
        raise ValueError("matching_indices contains out-of-bounds values.")

    alice_sifted_key = alice_bits[matching_indices]
    bob_sifted_key = bob_measured_bits[matching_indices]

    return alice_sifted_key, bob_sifted_key

def estimate_qber(alice_sifted_key: np.ndarray,
                  bob_sifted_key: np.ndarray,
                  sample_fraction: float = 0.5) -> Tuple[float, List[int], List[int]]:
    if len(alice_sifted_key) != len(bob_sifted_key):
        raise ValueError("Alice's and Bob's sifted keys must have the same length.")
    if not (0 < sample_fraction <= 1.0):
        raise ValueError("sample_fraction must be between 0 (exclusive) and 1 (inclusive).")

    key_len = len(alice_sifted_key)
    if key_len == 0:
        return 0.0, [], []

    sample_size = int(np.ceil(key_len * sample_fraction))
    if sample_size > key_len:
        sample_size = key_len

    if sample_size == 0:
         return 0.0, [], list(range(key_len))

    all_indices = list(range(key_len))
    random.shuffle(all_indices)

    qber_indices = sorted(all_indices[:sample_size])
    remaining_indices = sorted(all_indices[sample_size:])

    alice_sample = alice_sifted_key[qber_indices]
    bob_sample = bob_sifted_key[qber_indices]

    mismatches = np.sum(alice_sample != bob_sample)

    qber = mismatches / sample_size

    return qber, qber_indices, remaining_indices

def extract_final_key(sifted_key: np.ndarray, remaining_indices: List[int]) -> np.ndarray:
    if not remaining_indices:
        return np.array([], dtype=np.uint8)

    return sifted_key[remaining_indices]