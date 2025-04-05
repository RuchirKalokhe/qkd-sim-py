import numpy as np

def bits_to_string(bits: np.ndarray) -> str:
    if bits is None or len(bits) == 0:
        return ""
    return "".join(map(str, bits.astype(int)))