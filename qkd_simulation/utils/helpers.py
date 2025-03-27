# Random bit generation, maybe plotting helpers, etc.
# ./qkd_simulation/utils/helpers.py

import numpy as np

# This file is for general utility functions used across the qkd_simulation package.
# Avoid putting UI-specific code (like plotting) here. Plotting should be handled
# in the main Streamlit script (app.py) where st.pyplot() etc. are called.

def bits_to_string(bits: np.ndarray) -> str:
    """
    Converts a numpy array of bits (0s and 1s) into a string representation.
    Useful for displaying keys or sequences.

    Args:
        bits: A numpy array containing 0s and 1s.

    Returns:
        A string representation of the bits (e.g., "01101").
    """
    if bits is None or len(bits) == 0:
        return ""
    return "".join(map(str, bits.astype(int))) # Ensure elements are int before mapping to str


# Example usage (for testing purposes)
if __name__ == '__main__':
    test_bits = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    bit_string = bits_to_string(test_bits)
    print(f"Input bits: {test_bits}")
    print(f"Output string: '{bit_string}'")

    empty_bits = np.array([], dtype=np.uint8)
    empty_string = bits_to_string(empty_bits)
    print(f"Input empty bits: {empty_bits}")
    print(f"Output empty string: '{empty_string}'")