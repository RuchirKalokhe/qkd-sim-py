import numpy as np
from typing import Tuple, List, Optional, Dict
import random

# Constants for basis representation
RECTILINEAR_BASIS = 0  # Z basis (|0⟩, |1⟩)
DIAGONAL_BASIS = 1     # X basis (|+⟩, |-⟩)

def generate_bits_and_bases(num_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random bits and bases for Alice.
    
    Args:
        num_bits: Number of bits to generate
        
    Returns:
        Tuple of (bits, bases) where:
        - bits is an array of random bits (0 or 1)
        - bases is an array of random bases (0 for rectilinear, 1 for diagonal)
        
    Raises:
        ValueError: If num_bits is not positive
        
    Examples:
        >>> bits, bases = generate_bits_and_bases(10)
        >>> len(bits), len(bases)
        (10, 10)
    """
    if num_bits <= 0:
        raise ValueError("Number of bits must be positive.")

    bits = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)
    bases = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)
    return bits, bases

def generate_bases(num_bits: int) -> np.ndarray:
    """
    Generate random bases for Bob or Eve.
    
    Args:
        num_bits: Number of bases to generate
        
    Returns:
        Array of random bases (0 for rectilinear, 1 for diagonal)
        
    Raises:
        ValueError: If num_bits is not positive
        
    Examples:
        >>> bases = generate_bases(10)
        >>> len(bases)
        10
    """
    if num_bits <= 0:
        raise ValueError("Number of bits must be positive.")

    bases = np.random.randint(0, 2, size=num_bits, dtype=np.uint8)
    return bases

def compare_bases(alice_bases: np.ndarray, bob_bases: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Compare Alice's and Bob's bases and identify matching and mismatching positions.
    
    Args:
        alice_bases: Array of Alice's bases
        bob_bases: Array of Bob's bases
        
    Returns:
        Tuple of (matching_indices, mismatching_indices) where:
        - matching_indices is a list of indices where bases match
        - mismatching_indices is a list of indices where bases differ
        
    Raises:
        ValueError: If input arrays have different lengths
        
    Examples:
        >>> alice_bases = np.array([0, 1, 0, 1, 0])
        >>> bob_bases = np.array([0, 0, 1, 1, 0])
        >>> matching, mismatching = compare_bases(alice_bases, bob_bases)
        >>> matching
        [0, 3, 4]
        >>> mismatching
        [1, 2]
    """
    if len(alice_bases) != len(bob_bases):
        raise ValueError("Alice's and Bob's base arrays must have the same length.")

    matches = (alice_bases == bob_bases)
    matching_indices = np.where(matches)[0].tolist()
    mismatching_indices = np.where(~matches)[0].tolist()
    
    return matching_indices, mismatching_indices

def simulate_eve_interception(alice_bits: np.ndarray, 
                             alice_bases: np.ndarray, 
                             eve_bases: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Simulate Eve's intercept-resend attack on the quantum channel.
    
    Args:
        alice_bits: Alice's original bits
        alice_bases: Alice's chosen bases
        eve_bases: Eve's randomly chosen measurement bases
        
    Returns:
        Tuple of (intercepted_bits, statistics) where:
        - intercepted_bits is an array of bits as measured by Eve
        - statistics is a dictionary with counts of correct/incorrect measurements
        
    Raises:
        ValueError: If input arrays have different lengths
        
    Examples:
        >>> alice_bits = np.array([0, 1, 0, 1, 0])
        >>> alice_bases = np.array([0, 0, 1, 1, 0])
        >>> eve_bases = np.array([0, 1, 0, 1, 1])
        >>> eve_bits, stats = simulate_eve_interception(alice_bits, alice_bases, eve_bases)
    """
    if not (len(alice_bits) == len(alice_bases) == len(eve_bases)):
        raise ValueError("All input arrays must have the same length")
    
    # Initialize Eve's measured bits
    eve_bits = np.zeros_like(alice_bits)
    
    # Statistics
    stats = {
        "total_bits": len(alice_bits),
        "correct_basis": 0,
        "incorrect_basis": 0,
        "correct_measurements": 0,
        "incorrect_measurements": 0
    }
    
    for i in range(len(alice_bits)):
        if alice_bases[i] == eve_bases[i]:
            # Eve used the correct basis, will get the correct bit
            eve_bits[i] = alice_bits[i]
            stats["correct_basis"] += 1
            stats["correct_measurements"] += 1
        else:
            # Eve used the wrong basis, 50% chance of getting the correct bit
            if random.random() < 0.5:
                eve_bits[i] = alice_bits[i]
                stats["correct_measurements"] += 1
            else:
                eve_bits[i] = 1 - alice_bits[i]  # Flip the bit
                stats["incorrect_measurements"] += 1
            stats["incorrect_basis"] += 1
    
    return eve_bits, stats

def calculate_theoretical_qber(alice_bases: np.ndarray, 
                              bob_bases: np.ndarray, 
                              eve_present: bool = False) -> float:
    """
    Calculate the theoretical Quantum Bit Error Rate (QBER) based on basis choices.
    
    Args:
        alice_bases: Alice's chosen bases
        bob_bases: Bob's chosen bases
        eve_present: Whether Eve is intercepting the communication
        
    Returns:
        Theoretical QBER as a float between 0.0 and 1.0
        
    Examples:
        >>> alice_bases = np.array([0, 1, 0, 1, 0])
        >>> bob_bases = np.array([0, 0, 1, 1, 0])
        >>> calculate_theoretical_qber(alice_bases, bob_bases, eve_present=True)
        0.125
    """
    if len(alice_bases) != len(bob_bases):
        raise ValueError("Alice's and Bob's base arrays must have the same length")
    
    if not eve_present:
        return 0.0  # No errors in ideal case without Eve
    
    # With Eve present, calculate theoretical QBER
    matching_indices, _ = compare_bases(alice_bases, bob_bases)
    
    if not matching_indices:
        return 0.0
    
    # For each matching basis between Alice and Bob, there's a 25% chance of error
    # when Eve is present (50% chance Eve uses wrong basis * 50% chance of error when using wrong basis)
    return 0.25
