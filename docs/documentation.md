# QKD Simulation Documentation

## Introduction

This document provides detailed information about the Quantum Key Distribution (QKD) simulation application, including its architecture, components, and usage.

## Quantum Key Distribution Overview

Quantum Key Distribution is a secure communication method that uses principles of quantum mechanics to establish a shared secret key between two parties (traditionally called Alice and Bob). The security of QKD is based on the fundamental properties of quantum mechanics, particularly the no-cloning theorem and the observer effect.

### BB84 Protocol

The BB84 protocol, developed by Charles Bennett and Gilles Brassard in 1984, was the first QKD protocol. It works as follows:

1. **Qubit Preparation**: Alice generates random bits and random bases (rectilinear or diagonal), then prepares qubits accordingly.
2. **Qubit Transmission**: Alice sends the qubits to Bob through a quantum channel.
3. **Qubit Measurement**: Bob randomly chooses measurement bases and measures the received qubits.
4. **Basis Reconciliation**: Alice and Bob publicly compare their bases (but not the bit values).
5. **Key Sifting**: They keep only the bits where they used the same basis.
6. **Error Estimation**: They sacrifice a portion of their sifted key to estimate the error rate.
7. **Error Correction**: They correct errors in the remaining bits.
8. **Privacy Amplification**: They reduce the amount of information an eavesdropper might have about the key.

### E91 Protocol

The E91 protocol, proposed by Artur Ekert in 1991, uses entangled pairs of qubits. The protocol works as follows:

1. **Entangled Pair Generation**: A source generates entangled qubit pairs and sends one qubit to Alice and one to Bob.
2. **Measurement**: Alice and Bob independently choose random measurement bases and measure their qubits.
3. **Basis Comparison**: They publicly compare their measurement bases.
4. **Key Sifting**: They keep only the results where they used compatible bases.
5. **Bell Inequality Test**: They use some of their measurements to test Bell's inequality, which can detect eavesdropping.
6. **Key Extraction**: The remaining correlated measurements form their shared key.

## Application Architecture

The QKD simulation application is structured as follows:

### Core Components

1. **Protocols Module**: Implements the quantum key distribution protocols.
   - `bb84.py`: Implementation of the BB84 protocol.

2. **Quantum Module**: Handles quantum circuit creation and execution.
   - `circuits.py`: Creates quantum circuits for the protocols.
   - `runner.py`: Executes circuits on simulators or real quantum hardware.

3. **Classical Module**: Implements classical post-processing steps.
   - `processing.py`: Handles key sifting, error estimation, error correction, and privacy amplification.

4. **Utils Module**: Provides utility functions.
   - `helpers.py`: Contains helper functions for bit manipulation and formatting.

5. **Main Application**: The Streamlit web interface.
   - `app.py`: Implements the user interface and orchestrates the simulation.

### Data Flow

1. User configures simulation parameters in the web interface.
2. The application generates random bits and bases for Alice and Bob.
3. Quantum circuits are created based on these bits and bases.
4. Circuits are executed on the selected backend (simulator or IBM Quantum).
5. Measurement results are processed to extract the sifted key.
6. Error rate is estimated and error correction is applied if needed.
7. Final key is extracted and displayed to the user.

## API Reference

### Protocols Module

#### BB84 Protocol

```python
from qkd_simulation.protocols import bb84

# Generate random bits and bases for Alice
alice_bits, alice_bases = bb84.generate_bits_and_bases(num_bits=20)

# Generate random bases for Bob
bob_bases = bb84.generate_bases(num_bits=20)

# Compare bases to find matching positions
matching_indices, mismatching_indices = bb84.compare_bases(alice_bases, bob_bases)

# Simulate Eve's interception
eve_bases = bb84.generate_bases(num_bits=20)
eve_bits, stats = bb84.simulate_eve_interception(alice_bits, alice_bases, eve_bases)

# Calculate theoretical QBER with Eve present
qber = bb84.calculate_theoretical_qber(alice_bases, bob_bases, eve_present=True)
```

### Quantum Module

#### Circuit Creation

```python
from qkd_simulation.quantum import circuits

# Create BB84 circuits
qkd_circs = circuits.create_bb84_circuits(alice_bits, alice_bases, bob_bases)

# Create BB84 circuits with Eve's interception
eve_circs = circuits.create_bb84_circuits_with_eve(alice_bits, alice_bases, eve_bases, bob_bases)

# Create E91 circuits
e91_circs, bases = circuits.create_e91_circuits(num_pairs=10)

# Visualize a circuit
circuits.visualize_circuit(qkd_circs[0], filename="circuit.png")

# Optimize circuits
optimized_circs = circuits.optimize_circuits(qkd_circs, optimization_level=2)
```

#### Circuit Execution

```python
from qkd_simulation.quantum import runner

# Run on local simulator
results = runner.run_circuits_local_simulator(qkd_circs, shots=1)

# Run with Eve simulation
eve_bits, bob_bits = runner.run_circuits_with_eve_local_simulator(eve_circs, shots=1)

# Run with custom noise model
noise_model = runner.create_noise_model({'readout': 0.01, 'gate1': 0.005})
noisy_results = runner.run_circuits_local_simulator(qkd_circs, noise_model=noise_model)

# Run in batches
batch_results = runner.run_circuits_batch(qkd_circs, batch_size=20)

# Run on IBM Quantum hardware
from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_TOKEN")
ibm_results = runner.run_circuits_ibm_runtime(service, "ibm_brisbane", qkd_circs)
```

### Classical Module

#### Post-processing

```python
from qkd_simulation.classical import processing

# Sift the key
alice_sifted, bob_sifted = processing.sift_key(alice_bits, bob_measured_bits, matching_indices)

# Estimate QBER
qber, qber_indices, remaining_indices = processing.estimate_qber(alice_sifted, bob_sifted, sample_fraction=0.3)

# Extract final key
final_key = processing.extract_final_key(alice_sifted, remaining_indices)

# Apply error correction
corrected_key, error_rate = processing.apply_error_correction(alice_key, bob_key, error_threshold=0.15)

# Apply privacy amplification
secure_key = processing.privacy_amplification(corrected_key, security_parameter=0.8)
```

### Utils Module

#### Helper Functions

```python
from qkd_simulation.utils import helpers

# Convert bits to string
bit_string = helpers.bits_to_string(bits)

# Convert string to bits
bits = helpers.string_to_bits("0101")

# Calculate bit error rate
error_rate = helpers.calculate_bit_error_rate(bits1, bits2)

# Format bit array for display
formatted = helpers.format_bit_array(bits, group_size=4, separator=" ")
```

## Advanced Usage

### Custom Noise Models

The application supports custom noise models to simulate realistic quantum environments:

```python
from qkd_simulation.quantum.runner import create_noise_model

# Create a custom noise model
noise_model = create_noise_model({
    'readout': 0.02,  # 2% readout error
    'gate1': 0.005,   # 0.5% single-qubit gate error
    'gate2': 0.02     # 2% two-qubit gate error
})
```

### Batch Processing

For large simulations, use batch processing to improve performance:

```python
from qkd_simulation.quantum.runner import run_circuits_batch

# Run 1000 circuits in batches of 50
results = run_circuits_batch(circuits, batch_size=50)
```

### Error Correction and Privacy Amplification

Apply post-processing techniques to the sifted key:

```python
from qkd_simulation.classical.processing import apply_error_correction, privacy_amplification

# Apply error correction
corrected_key, error_rate = apply_error_correction(alice_key, bob_key)

# Apply privacy amplification
final_key = privacy_amplification(corrected_key, security_parameter=0.8)
```

## Troubleshooting

### Common Issues

1. **IBM Quantum Authentication Errors**:
   - Ensure your IBM Quantum token is correctly set in `.streamlit/secrets.toml`
   - Check that your token has not expired
   - Verify your internet connection

2. **Simulation Errors**:
   - For memory errors, reduce the number of qubits
   - For timeout errors, use batch processing
   - For transpilation errors, try a lower optimization level

3. **Installation Issues**:
   - Ensure you have Python 3.8 or higher
   - Try installing dependencies one by one to identify problematic packages
   - For Qiskit errors, check compatibility with your Python version

### Getting Help

If you encounter issues not covered in this documentation, please:
1. Check the GitHub repository issues section
2. Create a new issue with detailed information about your problem
3. Include error messages and steps to reproduce the issue
