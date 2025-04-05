import numpy as np
from typing import Union, List

def bits_to_string(bits: Union[np.ndarray, List[int]]) -> str:
    """
    Convert a bit array to a string representation.
    
    Args:
        bits: Array or list of bits (0s and 1s)
        
    Returns:
        String representation of the bits
        
    Examples:
        >>> bits_to_string(np.array([0, 1, 0, 1]))
        '0101'
    """
    if bits is None or len(bits) == 0:
        return ""
    
    # Convert numpy array to list if needed
    if isinstance(bits, np.ndarray):
        bits_list = bits.tolist()
    else:
        bits_list = bits
    
    # Join the bits into a string
    return ''.join(str(bit) for bit in bits_list)

def string_to_bits(bit_string: str) -> np.ndarray:
    """
    Convert a string of bits to a numpy array.
    
    Args:
        bit_string: String containing only '0' and '1' characters
        
    Returns:
        Numpy array of bits
        
    Raises:
        ValueError: If the string contains characters other than '0' and '1'
        
    Examples:
        >>> string_to_bits('0101')
        array([0, 1, 0, 1], dtype=uint8)
    """
    if not bit_string:
        return np.array([], dtype=np.uint8)
    
    # Validate input
    if not all(c in '01' for c in bit_string):
        raise ValueError("Input string must contain only '0' and '1' characters")
    
    # Convert string to numpy array
    return np.array([int(bit) for bit in bit_string], dtype=np.uint8)

def calculate_bit_error_rate(bits1: np.ndarray, bits2: np.ndarray) -> float:
    """
    Calculate the bit error rate between two bit arrays.
    
    Args:
        bits1: First bit array
        bits2: Second bit array of the same length
        
    Returns:
        Bit error rate as a float between 0.0 and 1.0
        
    Raises:
        ValueError: If the arrays have different lengths
        
    Examples:
        >>> calculate_bit_error_rate(np.array([0, 1, 0, 1]), np.array([0, 0, 0, 1]))
        0.25
    """
    if len(bits1) != len(bits2):
        raise ValueError("Bit arrays must have the same length")
    
    if len(bits1) == 0:
        return 0.0
    
    # Count the number of differing bits
    mismatches = np.sum(bits1 != bits2)
    
    # Calculate the error rate
    return float(mismatches) / len(bits1)

def format_bit_array(bits: np.ndarray, group_size: int = 4, separator: str = ' ') -> str:
    """
    Format a bit array with separators for better readability.
    
    Args:
        bits: Array of bits to format
        group_size: Number of bits per group
        separator: Separator string between groups
        
    Returns:
        Formatted string with separators
        
    Examples:
        >>> format_bit_array(np.array([1, 0, 1, 0, 1, 1, 0, 1]), 4, ' ')
        '1010 1101'
    """
    if bits is None or len(bits) == 0:
        return ""
    
    bit_string = bits_to_string(bits)
    
    # Group the bits
    groups = [bit_string[i:i+group_size] for i in range(0, len(bit_string), group_size)]
    
    # Join the groups with the separator
    return separator.join(groups)
