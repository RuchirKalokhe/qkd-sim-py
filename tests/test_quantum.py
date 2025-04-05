import unittest
import numpy as np
from qiskit import QuantumCircuit
from qkd_simulation.quantum import circuits
from qkd_simulation.quantum import runner
from qkd_simulation.quantum import optimized_runner
from qkd_simulation.utils import performance

class TestQuantumCircuits(unittest.TestCase):
    """Test cases for the quantum circuits module."""
    
    def test_create_bb84_circuits(self):
        """Test BB84 circuit creation."""
        alice_bits = np.array([0, 1, 0, 1], dtype=np.uint8)
        alice_bases = np.array([0, 0, 1, 1], dtype=np.uint8)
        bob_bases = np.array([0, 1, 1, 0], dtype=np.uint8)
        
        qkd_circs = circuits.create_bb84_circuits(alice_bits, alice_bases, bob_bases)
        
        # Check number of circuits
        self.assertEqual(len(qkd_circs), len(alice_bits))
        
        # Check circuit properties
        for i, circ in enumerate(qkd_circs):
            self.assertEqual(circ.num_qubits, 1)
            self.assertEqual(circ.num_clbits, 1)
            self.assertEqual(circ.name, f"qbit_{i}")
            
            # Check that the circuit contains the expected gates
            # For X gate when alice_bits[i] is 1
            has_x = any(inst[0].name == 'x' for inst in circ.data)
            self.assertEqual(has_x, alice_bits[i] == 1)
            
            # For H gate when alice_bases[i] is 1
            alice_h_count = sum(1 for inst in circ.data if inst[0].name == 'h')
            if alice_bases[i] == 1:
                self.assertGreaterEqual(alice_h_count, 1)
            
            # For H gate when bob_bases[i] is 1
            total_h_count = sum(1 for inst in circ.data if inst[0].name == 'h')
            if bob_bases[i] == 1:
                self.assertGreaterEqual(total_h_count, 1)
            
            # Check that the circuit contains a measurement
            has_measure = any(inst[0].name == 'measure' for inst in circ.data)
            self.assertTrue(has_measure)
        
        # Test with invalid inputs
        with self.assertRaises(ValueError):
            circuits.create_bb84_circuits(
                np.array([0, 1], dtype=np.uint8),
                np.array([0, 0, 1], dtype=np.uint8),
                np.array([0, 1, 1, 0], dtype=np.uint8)
            )
    
    def test_create_bb84_circuits_with_eve(self):
        """Test BB84 circuit creation with Eve's interception."""
        alice_bits = np.array([0, 1, 0, 1], dtype=np.uint8)
        alice_bases = np.array([0, 0, 1, 1], dtype=np.uint8)
        eve_bases = np.array([0, 1, 0, 1], dtype=np.uint8)
        bob_bases = np.array([0, 1, 1, 0], dtype=np.uint8)
        
        eve_circs = circuits.create_bb84_circuits_with_eve(alice_bits, alice_bases, eve_bases, bob_bases)
        
        # Check number of circuits
        self.assertEqual(len(eve_circs), len(alice_bits))
        
        # Check circuit properties
        for i, circ in enumerate(eve_circs):
            self.assertEqual(circ.num_qubits, 1)
            self.assertEqual(circ.num_clbits, 2)  # One for Eve, one for Bob
            self.assertEqual(circ.name, f"qbit_eve_{i}")
            
            # Check that the circuit contains the expected gates
            # For X gate when alice_bits[i] is 1
            has_x = any(inst[0].name == 'x' for inst in circ.data)
            self.assertEqual(has_x, alice_bits[i] == 1)
            
            # Check that the circuit contains measurements
            measure_count = sum(1 for inst in circ.data if inst[0].name == 'measure')
            self.assertEqual(measure_count, 2)  # One for Eve, one for Bob
            
            # Check that the circuit contains a reset operation
            has_reset = any(inst[0].name == 'reset' for inst in circ.data)
            self.assertTrue(has_reset)
        
        # Test with invalid inputs
        with self.assertRaises(ValueError):
            circuits.create_bb84_circuits_with_eve(
                np.array([0, 1], dtype=np.uint8),
                np.array([0, 0], dtype=np.uint8),
                np.array([0, 1, 0], dtype=np.uint8),
                np.array([0, 1], dtype=np.uint8)
            )
    
    def test_create_e91_circuits(self):
        """Test E91 circuit creation."""
        num_pairs = 5
        e91_circs, bases = circuits.create_e91_circuits(num_pairs)
        
        # Check number of circuits and bases
        self.assertEqual(len(e91_circs), num_pairs)
        self.assertEqual(bases.shape, (num_pairs, 2))
        
        # Check circuit properties
        for i, circ in enumerate(e91_circs):
            self.assertEqual(circ.num_qubits, 2)
            self.assertEqual(circ.num_clbits, 2)
            self.assertEqual(circ.name, f"e91_pair_{i}")
            
            # Check that the circuit contains the expected gates
            # H gate for creating entanglement
            has_h = any(inst[0].name == 'h' for inst in circ.data)
            self.assertTrue(has_h)
            
            # CX gate for creating entanglement
            has_cx = any(inst[0].name == 'cx' for inst in circ.data)
            self.assertTrue(has_cx)
            
            # Check that the circuit contains measurements
            measure_count = sum(1 for inst in circ.data if inst[0].name == 'measure')
            self.assertEqual(measure_count, 2)  # One for Alice, one for Bob
        
        # Test with invalid input
        with self.assertRaises(ValueError):
            circuits.create_e91_circuits(0)
    
    def test_visualize_circuit(self):
        """Test circuit visualization."""
        # Create a simple circuit
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        
        # Test without saving to file
        result = circuits.visualize_circuit(qc)
        self.assertIsNone(result)
        
        # We can't easily test the file saving functionality in a unit test
        # without actually creating files, so we'll just test the validation
        with self.assertRaises(ValueError):
            circuits.visualize_circuit(qc, filename="invalid_extension.txt")
    
    def test_optimize_circuits(self):
        """Test circuit optimization."""
        # Create some test circuits
        test_circs = []
        for i in range(3):
            qc = QuantumCircuit(1, 1, name=f"test_{i}")
            qc.h(0)
            if i % 2 == 1:
                qc.x(0)
            qc.h(0)  # This could be optimized away when followed by measurement
            qc.measure(0, 0)
            test_circs.append(qc)
        
        # Test optimization
        opt_level = 3  # Highest optimization level
        opt_circs = circuits.optimize_circuits(test_circs, optimization_level=opt_level)
        
        # Check that we got the right number of circuits back
        self.assertEqual(len(opt_circs), len(test_circs))
        
        # Check that each circuit still has the same measurement
        for circ in opt_circs:
            has_measure = any(inst[0].name == 'measure' for inst in circ.data)
            self.assertTrue(has_measure)
        
        # Test with invalid optimization level
        with self.assertRaises(ValueError):
            circuits.optimize_circuits(test_circs, optimization_level=4)

class TestOptimizedProcessing(unittest.TestCase):
    """Test cases for the optimized classical processing module."""
    
    def test_parallel_transpile(self):
        """Test parallel transpilation."""
        # Create some test circuits
        test_circs = []
        for i in range(10):
            qc = QuantumCircuit(1, 1, name=f"test_{i}")
            qc.h(0)
            if i % 2 == 1:
                qc.x(0)
            qc.measure(0, 0)
            test_circs.append(qc)
        
        # Test parallel transpilation
        from qiskit_aer import AerSimulator
        simulator = AerSimulator()
        
        transpiled_circs = performance.parallel_transpile(
            test_circs, 
            simulator, 
            optimization_level=1,
            max_workers=2
        )
        
        # Check that we got the right number of circuits back
        self.assertEqual(len(transpiled_circs), len(test_circs))
        
        # Check that each circuit still has the same measurement
        for circ in transpiled_circs:
            has_measure = any(inst[0].name == 'measure' for inst in circ.data)
            self.assertTrue(has_measure)
    
    def test_cached_basis_comparison(self):
        """Test cached basis comparison."""
        alice_bases = (0, 1, 0, 1, 0)
        bob_bases = (0, 0, 1, 1, 0)
        
        # First call should compute the result
        matching1, mismatching1 = performance.cached_basis_comparison(alice_bases, bob_bases)
        
        # Second call should use the cached result
        matching2, mismatching2 = performance.cached_basis_comparison(alice_bases, bob_bases)
        
        # Results should be the same
        self.assertEqual(matching1, matching2)
        self.assertEqual(mismatching1, mismatching2)
        
        # Check the actual results
        self.assertEqual(matching1, (0, 3, 4))
        self.assertEqual(mismatching1, (1, 2))
    
    def test_vectorized_bit_operations(self):
        """Test vectorized bit operations."""
        bits1 = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        bits2 = np.array([0, 0, 0, 1, 1], dtype=np.uint8)
        
        matching, total, percentage = performance.vectorized_bit_operations(bits1, bits2)
        
        # Check results
        self.assertEqual(matching, 3)  # 3 matching bits
        self.assertEqual(total, 5)     # 5 total bits
        self.assertEqual(percentage, 0.6)  # 60% match
        
        # Test with different lengths
        bits1 = np.array([0, 1, 0], dtype=np.uint8)
        bits2 = np.array([0, 0], dtype=np.uint8)
        with self.assertRaises(ValueError):
            performance.vectorized_bit_operations(bits1, bits2)

class TestIntegration(unittest.TestCase):
    """Integration tests for the QKD simulation."""
    
    def test_full_bb84_protocol(self):
        """Test the full BB84 protocol flow."""
        from qkd_simulation.protocols import bb84
        from qkd_simulation.quantum import circuits
        from qkd_simulation.quantum import runner
        from qkd_simulation.classical import processing
        
        # 1. Generate random bits and bases for Alice
        num_bits = 20
        alice_bits, alice_bases = bb84.generate_bits_and_bases(num_bits)
        
        # 2. Generate random bases for Bob
        bob_bases = bb84.generate_bases(num_bits)
        
        # 3. Create quantum circuits
        qkd_circs = circuits.create_bb84_circuits(alice_bits, alice_bases, bob_bases)
        
        # 4. Run circuits on local simulator
        measured_bits = runner.run_circuits_local_simulator(qkd_circs, shots=1)
        self.assertIsNotNone(measured_bits)
        self.assertEqual(len(measured_bits), num_bits)
        
        # 5. Compare bases
        matching_indices, _ = bb84.compare_bases(alice_bases, bob_bases)
        
        # 6. Sift key
        bob_measured_bits = np.array(measured_bits, dtype=np.uint8)
        alice_sifted, bob_sifted = processing.sift_key(alice_bits, bob_measured_bits, matching_indices)
        
        # 7. Estimate QBER
        if len(alice_sifted) > 0:
            qber, qber_indices, remaining_indices = processing.estimate_qber(
                alice_sifted, bob_sifted, sample_fraction=0.3
            )
            self.assertGreaterEqual(qber, 0.0)
            self.assertLessEqual(qber, 1.0)
            
            # 8. Extract final key
            final_key = processing.extract_final_key(alice_sifted, remaining_indices)
            self.assertEqual(len(final_key), len(remaining_indices))
    
    def test_optimized_processing(self):
        """Test the optimized classical processing."""
        from qkd_simulation.classical import optimized_processing
        
        # Generate test data
        alice_bits = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
        bob_bits = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.uint8)
        alice_bases = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.uint8)
        bob_bases = np.array([0, 1, 1, 0, 0, 0, 1, 0], dtype=np.uint8)
        
        # Test batch processing
        results = optimized_processing.batch_process_keys(
            alice_bits,
            bob_bits,
            alice_bases,
            bob_bases,
            qber_sample_fraction=0.3,
            error_threshold=0.2,
            security_parameter=0.8
        )
        
        # Check that we got results
        self.assertIn('matching_indices', results)
        self.assertIn('alice_sifted', results)
        self.assertIn('bob_sifted', results)
        self.assertIn('qber', results)
        
        # Check that the sifted keys have the expected length
        matching_indices = results['matching_indices']
        self.assertEqual(len(results['alice_sifted']), len(matching_indices))
        self.assertEqual(len(results['bob_sifted']), len(matching_indices))

if __name__ == "__main__":
    unittest.main()
