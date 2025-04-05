import unittest
import numpy as np
from qkd_simulation.protocols import bb84
from qkd_simulation.classical import processing
from qkd_simulation.utils import helpers

class TestBB84Protocol(unittest.TestCase):
    """Test cases for the BB84 protocol implementation."""
    
    def test_generate_bits_and_bases(self):
        """Test that bits and bases are generated correctly."""
        num_bits = 100
        bits, bases = bb84.generate_bits_and_bases(num_bits)
        
        # Check shapes
        self.assertEqual(len(bits), num_bits)
        self.assertEqual(len(bases), num_bits)
        
        # Check types
        self.assertEqual(bits.dtype, np.uint8)
        self.assertEqual(bases.dtype, np.uint8)
        
        # Check values are valid (0 or 1)
        self.assertTrue(np.all((bits == 0) | (bits == 1)))
        self.assertTrue(np.all((bases == 0) | (bases == 1)))
        
        # Check randomness (statistical test)
        # For a large enough sample, bits should be roughly 50% 0s and 50% 1s
        zeros_bits = np.sum(bits == 0)
        ones_bits = np.sum(bits == 1)
        self.assertGreater(zeros_bits, num_bits * 0.3)
        self.assertGreater(ones_bits, num_bits * 0.3)
        
        zeros_bases = np.sum(bases == 0)
        ones_bases = np.sum(bases == 1)
        self.assertGreater(zeros_bases, num_bits * 0.3)
        self.assertGreater(ones_bases, num_bits * 0.3)
    
    def test_generate_bases(self):
        """Test that bases are generated correctly."""
        num_bits = 100
        bases = bb84.generate_bases(num_bits)
        
        # Check shape
        self.assertEqual(len(bases), num_bits)
        
        # Check type
        self.assertEqual(bases.dtype, np.uint8)
        
        # Check values are valid (0 or 1)
        self.assertTrue(np.all((bases == 0) | (bases == 1)))
        
        # Check randomness (statistical test)
        zeros = np.sum(bases == 0)
        ones = np.sum(bases == 1)
        self.assertGreater(zeros, num_bits * 0.3)
        self.assertGreater(ones, num_bits * 0.3)
    
    def test_compare_bases(self):
        """Test that basis comparison works correctly."""
        # Test with all matching bases
        alice_bases = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        bob_bases = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        matching, mismatching = bb84.compare_bases(alice_bases, bob_bases)
        self.assertEqual(matching, [0, 1, 2, 3, 4])
        self.assertEqual(mismatching, [])
        
        # Test with no matching bases
        alice_bases = np.array([0, 0, 0, 0, 0], dtype=np.uint8)
        bob_bases = np.array([1, 1, 1, 1, 1], dtype=np.uint8)
        matching, mismatching = bb84.compare_bases(alice_bases, bob_bases)
        self.assertEqual(matching, [])
        self.assertEqual(mismatching, [0, 1, 2, 3, 4])
        
        # Test with some matching bases
        alice_bases = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        bob_bases = np.array([0, 0, 1, 1, 1], dtype=np.uint8)
        matching, mismatching = bb84.compare_bases(alice_bases, bob_bases)
        self.assertEqual(matching, [0, 3])
        self.assertEqual(mismatching, [1, 2, 4])
    
    def test_simulate_eve_interception(self):
        """Test Eve's intercept-resend attack simulation."""
        alice_bits = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        alice_bases = np.array([0, 0, 1, 1, 0], dtype=np.uint8)
        eve_bases = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
        
        eve_bits, stats = bb84.simulate_eve_interception(alice_bits, alice_bases, eve_bases)
        
        # Check shape and type
        self.assertEqual(len(eve_bits), len(alice_bits))
        self.assertEqual(eve_bits.dtype, np.uint8)
        
        # Check statistics
        self.assertEqual(stats['total_bits'], 5)
        self.assertEqual(stats['correct_basis'] + stats['incorrect_basis'], 5)
        self.assertEqual(stats['correct_measurements'] + stats['incorrect_measurements'], 5)
        
        # Check that Eve gets correct bits when using correct basis
        for i in range(len(alice_bits)):
            if alice_bases[i] == eve_bases[i]:
                self.assertEqual(eve_bits[i], alice_bits[i])
    
    def test_calculate_theoretical_qber(self):
        """Test theoretical QBER calculation."""
        alice_bases = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        bob_bases = np.array([0, 0, 1, 1, 0], dtype=np.uint8)
        
        # Without Eve
        qber = bb84.calculate_theoretical_qber(alice_bases, bob_bases, eve_present=False)
        self.assertEqual(qber, 0.0)
        
        # With Eve
        qber = bb84.calculate_theoretical_qber(alice_bases, bob_bases, eve_present=True)
        self.assertEqual(qber, 0.25)

class TestClassicalProcessing(unittest.TestCase):
    """Test cases for the classical processing module."""
    
    def test_sift_key(self):
        """Test key sifting."""
        alice_bits = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        bob_bits = np.array([0, 0, 0, 1, 1], dtype=np.uint8)
        matching_indices = [0, 2, 4]
        
        alice_sifted, bob_sifted = processing.sift_key(alice_bits, bob_bits, matching_indices)
        
        # Check shapes
        self.assertEqual(len(alice_sifted), len(matching_indices))
        self.assertEqual(len(bob_sifted), len(matching_indices))
        
        # Check values
        np.testing.assert_array_equal(alice_sifted, np.array([0, 0, 0], dtype=np.uint8))
        np.testing.assert_array_equal(bob_sifted, np.array([0, 0, 1], dtype=np.uint8))
        
        # Test with empty matching indices
        alice_sifted, bob_sifted = processing.sift_key(alice_bits, bob_bits, [])
        self.assertEqual(len(alice_sifted), 0)
        self.assertEqual(len(bob_sifted), 0)
        
        # Test with invalid indices
        with self.assertRaises(ValueError):
            processing.sift_key(alice_bits, bob_bits, [10])
    
    def test_estimate_qber(self):
        """Test QBER estimation."""
        # Test with no errors
        alice_sifted = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        bob_sifted = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        qber, qber_indices, remaining_indices = processing.estimate_qber(alice_sifted, bob_sifted, 0.4)
        self.assertEqual(qber, 0.0)
        self.assertEqual(len(qber_indices) + len(remaining_indices), len(alice_sifted))
        
        # Test with some errors
        alice_sifted = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        bob_sifted = np.array([0, 0, 0, 1, 1], dtype=np.uint8)
        qber, qber_indices, remaining_indices = processing.estimate_qber(alice_sifted, bob_sifted, 0.4)
        self.assertGreaterEqual(qber, 0.0)
        self.assertLessEqual(qber, 1.0)
        self.assertEqual(len(qber_indices) + len(remaining_indices), len(alice_sifted))
        
        # Test with empty arrays
        alice_sifted = np.array([], dtype=np.uint8)
        bob_sifted = np.array([], dtype=np.uint8)
        qber, qber_indices, remaining_indices = processing.estimate_qber(alice_sifted, bob_sifted, 0.4)
        self.assertEqual(qber, 0.0)
        self.assertEqual(qber_indices, [])
        self.assertEqual(remaining_indices, [])
    
    def test_extract_final_key(self):
        """Test final key extraction."""
        sifted_key = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        remaining_indices = [0, 2, 4]
        
        final_key = processing.extract_final_key(sifted_key, remaining_indices)
        
        # Check shape
        self.assertEqual(len(final_key), len(remaining_indices))
        
        # Check values
        np.testing.assert_array_equal(final_key, np.array([0, 0, 0], dtype=np.uint8))
        
        # Test with empty remaining indices
        final_key = processing.extract_final_key(sifted_key, [])
        self.assertEqual(len(final_key), 0)
    
    def test_apply_error_correction(self):
        """Test error correction."""
        alice_key = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        bob_key = np.array([0, 0, 0, 1, 1], dtype=np.uint8)
        
        corrected_key, error_rate = processing.apply_error_correction(alice_key, bob_key)
        
        # Check shape
        self.assertEqual(len(corrected_key), len(alice_key))
        
        # Check error rate
        self.assertGreaterEqual(error_rate, 0.0)
        self.assertLessEqual(error_rate, 1.0)
        
        # Test with no errors
        alice_key = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        bob_key = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        corrected_key, error_rate = processing.apply_error_correction(alice_key, bob_key)
        self.assertEqual(error_rate, 0.0)
        np.testing.assert_array_equal(corrected_key, alice_key)
        
        # Test with empty arrays
        alice_key = np.array([], dtype=np.uint8)
        bob_key = np.array([], dtype=np.uint8)
        corrected_key, error_rate = processing.apply_error_correction(alice_key, bob_key)
        self.assertEqual(error_rate, 0.0)
        self.assertEqual(len(corrected_key), 0)
    
    def test_privacy_amplification(self):
        """Test privacy amplification."""
        key = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
        security_parameter = 0.5
        
        final_key = processing.privacy_amplification(key, security_parameter)
        
        # Check shape
        expected_length = int(len(key) * security_parameter)
        self.assertEqual(len(final_key), expected_length)
        
        # Check values are valid
        self.assertTrue(np.all((final_key == 0) | (final_key == 1)))
        
        # Test with empty key
        key = np.array([], dtype=np.uint8)
        final_key = processing.privacy_amplification(key, security_parameter)
        self.assertEqual(len(final_key), 0)

class TestHelpers(unittest.TestCase):
    """Test cases for the helper functions."""
    
    def test_bits_to_string(self):
        """Test bits to string conversion."""
        bits = np.array([0, 1, 0, 1], dtype=np.uint8)
        bit_string = helpers.bits_to_string(bits)
        self.assertEqual(bit_string, "0101")
        
        # Test with empty array
        bits = np.array([], dtype=np.uint8)
        bit_string = helpers.bits_to_string(bits)
        self.assertEqual(bit_string, "")
        
        # Test with None
        bit_string = helpers.bits_to_string(None)
        self.assertEqual(bit_string, "")
    
    def test_string_to_bits(self):
        """Test string to bits conversion."""
        bit_string = "0101"
        bits = helpers.string_to_bits(bit_string)
        np.testing.assert_array_equal(bits, np.array([0, 1, 0, 1], dtype=np.uint8))
        
        # Test with empty string
        bit_string = ""
        bits = helpers.string_to_bits(bit_string)
        self.assertEqual(len(bits), 0)
        
        # Test with invalid string
        with self.assertRaises(ValueError):
            helpers.string_to_bits("01012")
    
    def test_calculate_bit_error_rate(self):
        """Test bit error rate calculation."""
        bits1 = np.array([0, 1, 0, 1], dtype=np.uint8)
        bits2 = np.array([0, 0, 0, 1], dtype=np.uint8)
        error_rate = helpers.calculate_bit_error_rate(bits1, bits2)
        self.assertEqual(error_rate, 0.25)
        
        # Test with no errors
        bits1 = np.array([0, 1, 0, 1], dtype=np.uint8)
        bits2 = np.array([0, 1, 0, 1], dtype=np.uint8)
        error_rate = helpers.calculate_bit_error_rate(bits1, bits2)
        self.assertEqual(error_rate, 0.0)
        
        # Test with all errors
        bits1 = np.array([0, 0, 0, 0], dtype=np.uint8)
        bits2 = np.array([1, 1, 1, 1], dtype=np.uint8)
        error_rate = helpers.calculate_bit_error_rate(bits1, bits2)
        self.assertEqual(error_rate, 1.0)
        
        # Test with empty arrays
        bits1 = np.array([], dtype=np.uint8)
        bits2 = np.array([], dtype=np.uint8)
        error_rate = helpers.calculate_bit_error_rate(bits1, bits2)
        self.assertEqual(error_rate, 0.0)
        
        # Test with different lengths
        bits1 = np.array([0, 1, 0, 1], dtype=np.uint8)
        bits2 = np.array([0, 1], dtype=np.uint8)
        with self.assertRaises(ValueError):
            helpers.calculate_bit_error_rate(bits1, bits2)
    
    def test_format_bit_array(self):
        """Test bit array formatting."""
        bits = np.array([1, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
        formatted = helpers.format_bit_array(bits, group_size=4, separator=" ")
        self.assertEqual(formatted, "1010 1101")
        
        # Test with different group size
        formatted = helpers.format_bit_array(bits, group_size=2, separator="-")
        self.assertEqual(formatted, "10-10-11-01")
        
        # Test with empty array
        bits = np.array([], dtype=np.uint8)
        formatted = helpers.format_bit_array(bits)
        self.assertEqual(formatted, "")
        
        # Test with None
        formatted = helpers.format_bit_array(None)
        self.assertEqual(formatted, "")

if __name__ == "__main__":
    unittest.main()
