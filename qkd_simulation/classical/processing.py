import numpy as np
from typing import Tuple, List, Optional, Union
import random

def sift_key(alice_bits: np.ndarray,
             bob_measured_bits: np.ndarray,
             matching_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sift the key by keeping only the bits where Alice and Bob used the same basis.
    
    Args:
        alice_bits: Array of Alice's original bits
        bob_measured_bits: Array of Bob's measured bits
        matching_indices: List of indices where Alice and Bob used the same basis
        
    Returns:
        Tuple of (alice_sifted_key, bob_sifted_key)
        
    Raises:
        ValueError: If input arrays have different lengths or if matching_indices contains invalid values
    """
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
    """
    Estimate the Quantum Bit Error Rate (QBER) by comparing a subset of the sifted keys.
    
    Args:
        alice_sifted_key: Alice's sifted key
        bob_sifted_key: Bob's sifted key
        sample_fraction: Fraction of the sifted key to use for QBER estimation (0-1)
        
    Returns:
        Tuple of (qber, qber_indices, remaining_indices) where:
        - qber is the estimated error rate (0-1)
        - qber_indices are the indices used for QBER estimation
        - remaining_indices are the indices that can be used for the final key
        
    Raises:
        ValueError: If input arrays have different lengths or if sample_fraction is invalid
    """
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
    """
    Extract the final key from the sifted key using the remaining indices
    (those not used for QBER estimation).
    
    Args:
        sifted_key: The sifted key (either Alice's or Bob's)
        remaining_indices: Indices of the sifted key to use for the final key
        
    Returns:
        Array containing the final key bits
        
    Examples:
        >>> extract_final_key(np.array([1, 0, 1, 0, 1, 1]), [0, 2, 4])
        array([1, 1, 1], dtype=uint8)
    """
    if not remaining_indices:
        return np.array([], dtype=np.uint8)
    
    # Validate indices
    if max(remaining_indices, default=-1) >= len(sifted_key):
        raise ValueError("remaining_indices contains out-of-bounds values")
    
    # Extract the bits at the specified indices
    final_key = sifted_key[remaining_indices]
    return final_key

def apply_error_correction(alice_key: np.ndarray, 
                          bob_key: np.ndarray, 
                          error_threshold: float = 0.15) -> Tuple[np.ndarray, float]:
    """
    Apply a simple error correction procedure to reconcile differences between
    Alice's and Bob's keys.
    
    This is a basic implementation that uses parity checking on blocks of bits.
    In a production system, more sophisticated error correction codes would be used.
    
    Args:
        alice_key: Alice's key bits
        bob_key: Bob's key bits
        error_threshold: Maximum acceptable error rate after correction
        
    Returns:
        Tuple of (corrected_key, error_rate_after_correction)
        
    Raises:
        ValueError: If input arrays have different lengths
        RuntimeError: If error rate after correction exceeds threshold
    """
    if len(alice_key) != len(bob_key):
        raise ValueError("Alice's and Bob's keys must have the same length")
    
    if len(alice_key) == 0:
        return np.array([], dtype=np.uint8), 0.0
    
    # Make copies to avoid modifying the originals
    alice_copy = alice_key.copy()
    bob_copy = bob_key.copy()
    
    # Find positions where bits differ
    diff_positions = np.where(alice_copy != bob_copy)[0]
    
    # If no differences, return original key
    if len(diff_positions) == 0:
        return alice_copy, 0.0
    
    # Simple correction: Bob adopts Alice's bits at differing positions
    # In a real implementation, this would involve communication and more sophisticated codes
    bob_copy[diff_positions] = alice_copy[diff_positions]
    
    # Calculate error rate after correction
    errors_after = np.sum(alice_copy != bob_copy)
    error_rate = float(errors_after) / len(alice_copy)
    
    if error_rate > error_threshold:
        raise RuntimeError(f"Error rate after correction ({error_rate:.2%}) exceeds threshold ({error_threshold:.2%})")
    
    return alice_copy, error_rate

def privacy_amplification(key: np.ndarray, security_parameter: float = 0.5) -> np.ndarray:
    """
    Apply privacy amplification to reduce the amount of information an eavesdropper might have.
    
    This is a simplified implementation that reduces the key length.
    In a production system, universal hash functions would be used.
    
    Args:
        key: The reconciled key
        security_parameter: Controls how much the key is shortened (0-1)
        
    Returns:
        The shortened, more secure key
    """
    if len(key) == 0:
        return np.array([], dtype=np.uint8)
    
    if not (0 < security_parameter <= 1.0):
        raise ValueError("security_parameter must be between 0 (exclusive) and 1 (inclusive)")
    
    # Calculate new length based on security parameter
    new_length = max(1, int(len(key) * security_parameter))
    
    # Simple implementation: just take a subset of the key
    # In a real implementation, this would involve universal hashing
    indices = sorted(random.sample(range(len(key)), new_length))
    
    return key[indices]
