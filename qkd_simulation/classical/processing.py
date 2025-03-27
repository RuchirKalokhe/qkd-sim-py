# Functions for sifting, QBER calculation, etc.
# ./qkd_simulation/classical/processing.py

import numpy as np
from typing import Tuple, List, Optional
import random # For selecting QBER sample

def sift_key(alice_bits: np.ndarray,
             bob_measured_bits: np.ndarray,
             matching_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sifts the keys based on the indices where bases matched.

    Args:
        alice_bits: Numpy array of Alice's original random bits.
        bob_measured_bits: Numpy array of Bob's measured bits (results from quantum run).
                            Should be the same length as alice_bits.
        matching_indices: A list of integer indices where Alice's and Bob's bases matched.

    Returns:
        A tuple containing:
            - alice_sifted_key (np.ndarray): Alice's bits corresponding to matching bases.
            - bob_sifted_key (np.ndarray): Bob's measured bits corresponding to matching bases.

    Raises:
        ValueError: If input array lengths don't match expected relationships.
    """
    if len(alice_bits) != len(bob_measured_bits):
        raise ValueError("Alice's bits and Bob's measured bits must have the same length.")
    if not matching_indices:
        # Return empty arrays if no bases matched
        return np.array([], dtype=np.uint8), np.array([], dtype=np.uint8)

    # Ensure indices are within bounds (optional sanity check)
    max_index = len(alice_bits) - 1
    if any(idx < 0 or idx > max_index for idx in matching_indices):
        raise ValueError("matching_indices contains out-of-bounds values.")

    # Use numpy fancy indexing to select bits at matching indices
    alice_sifted_key = alice_bits[matching_indices]
    bob_sifted_key = bob_measured_bits[matching_indices]

    return alice_sifted_key, bob_sifted_key


def estimate_qber(alice_sifted_key: np.ndarray,
                  bob_sifted_key: np.ndarray,
                  sample_fraction: float = 0.5) -> Tuple[float, List[int], List[int]]:
    """
    Estimates the Quantum Bit Error Rate (QBER) by comparing a sample of the sifted keys.

    Args:
        alice_sifted_key: Alice's sifted key bits.
        bob_sifted_key: Bob's sifted key bits.
        sample_fraction: The fraction of the sifted key to use for QBER estimation (default 0.5).
                         The indices used for estimation are revealed and discarded from the final key.

    Returns:
        A tuple containing:
            - qber (float): The estimated Quantum Bit Error Rate (mismatches / sample size).
                            Returns -1.0 if the sample size is zero.
            - qber_indices (List[int]): The indices within the *sifted* key used for estimation.
            - remaining_indices (List[int]): The indices within the *sifted* key *not* used for estimation.

    Raises:
        ValueError: If sifted key lengths don't match or sample_fraction is invalid.
    """
    if len(alice_sifted_key) != len(bob_sifted_key):
        raise ValueError("Alice's and Bob's sifted keys must have the same length.")
    if not (0 < sample_fraction <= 1.0):
        raise ValueError("sample_fraction must be between 0 (exclusive) and 1 (inclusive).")

    key_len = len(alice_sifted_key)
    if key_len == 0:
        return 0.0, [], [] # No errors if key length is zero

    sample_size = int(np.ceil(key_len * sample_fraction)) # Use ceil to ensure at least 1 if fraction > 0
    if sample_size > key_len:
        sample_size = key_len # Cannot sample more than available bits

    if sample_size == 0:
         # This case should ideally not be hit if key_len > 0 and sample_fraction > 0,
         # but handle defensively. No bits compared, so QBER is undefined/zero.
         return 0.0, [], list(range(key_len))


    all_indices = list(range(key_len))
    random.shuffle(all_indices) # Shuffle indices to pick a random sample

    qber_indices = sorted(all_indices[:sample_size])
    remaining_indices = sorted(all_indices[sample_size:])

    # Select the bits at the chosen indices for comparison
    alice_sample = alice_sifted_key[qber_indices]
    bob_sample = bob_sifted_key[qber_indices]

    # Count mismatches
    mismatches = np.sum(alice_sample != bob_sample)

    # Calculate QBER
    qber = mismatches / sample_size

    return qber, qber_indices, remaining_indices


def extract_final_key(sifted_key: np.ndarray, remaining_indices: List[int]) -> np.ndarray:
    """
    Extracts the final key bits after QBER estimation bits are removed.
    (Assumes no further error correction or privacy amplification).

    Args:
        sifted_key: The sifted key (either Alice's or Bob's).
        remaining_indices: Indices within the sifted key NOT used for QBER estimation.

    Returns:
        final_key (np.ndarray): The bits forming the final shared secret key.
    """
    if not remaining_indices:
        return np.array([], dtype=np.uint8)

    return sifted_key[remaining_indices]


# Example usage (for testing purposes)
if __name__ == '__main__':
    print("--- Testing Classical Processing ---")

    # Simulate input data
    N_total = 20
    alice_bits_test = np.random.randint(0, 2, N_total, dtype=np.uint8)
    # Simulate Bob's measurements with some errors (~10%)
    bob_measured_test = alice_bits_test.copy()
    error_indices = np.random.choice(N_total, size=int(N_total * 0.1), replace=False)
    for idx in error_indices:
        bob_measured_test[idx] = 1 - bob_measured_test[idx] # Flip the bit

    # Simulate matching bases (around 50%)
    matching_indices_test = sorted(np.random.choice(N_total, size=int(N_total * 0.55), replace=False).tolist())

    print(f"Original Alice Bits:    {alice_bits_test}")
    print(f"Simulated Bob Measured: {bob_measured_test}")
    print(f"Indices Bases Matched:  {matching_indices_test}")

    # --- Test Sifting ---
    print("\n--- Sifting ---")
    try:
        alice_sifted, bob_sifted = sift_key(alice_bits_test, bob_measured_test, matching_indices_test)
        print(f"Alice Sifted Key: {alice_sifted} (Length: {len(alice_sifted)})")
        print(f"Bob Sifted Key:   {bob_sifted} (Length: {len(bob_sifted)})")
        if len(alice_sifted) > 0:
             print(f"Errors in Sifted Key: {np.sum(alice_sifted != bob_sifted)}")
        else:
             print("Sifted keys are empty.")
    except ValueError as e:
        print(f"Sifting Error: {e}")

    # --- Test QBER Estimation ---
    print("\n--- QBER Estimation ---")
    if len(alice_sifted) > 0:
        try:
            qber_estimate, qber_idx, remain_idx = estimate_qber(alice_sifted, bob_sifted, sample_fraction=0.4)
            print(f"Estimated QBER: {qber_estimate:.4f}")
            print(f"Indices used for QBER (in sifted key): {qber_idx}")
            print(f"Indices remaining (in sifted key):     {remain_idx}")

            # --- Test Final Key Extraction ---
            print("\n--- Final Key Extraction ---")
            final_key_alice = extract_final_key(alice_sifted, remain_idx)
            final_key_bob = extract_final_key(bob_sifted, remain_idx)
            print(f"Final Alice Key: {final_key_alice} (Length: {len(final_key_alice)})")
            print(f"Final Bob Key:   {final_key_bob} (Length: {len(final_key_bob)})")
            if len(final_key_alice) > 0:
                 print(f"Errors in Final Key: {np.sum(final_key_alice != final_key_bob)}")
            else:
                 print("Final keys are empty.")


        except ValueError as e:
            print(f"QBER Estimation Error: {e}")
    else:
        print("Skipping QBER estimation as sifted keys are empty.")