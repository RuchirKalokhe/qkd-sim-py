import time
import logging
from typing import List, Optional, Dict, Union, Tuple
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.providers import Backend
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
import streamlit as st
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

def run_circuits_local_simulator(circuits: List[QuantumCircuit], 
                                shots: int = 1,
                                noise_model = None,
                                optimization_level: int = 1) -> Optional[List[int]]:
    """
    Run quantum circuits on the local Qiskit Aer simulator.
    
    Args:
        circuits: List of quantum circuits to run
        shots: Number of shots per circuit
        noise_model: Optional noise model to apply to the simulation
        optimization_level: Transpiler optimization level (0-3)
        
    Returns:
        List of measured bits (one per circuit), or None if simulation fails
        
    Examples:
        >>> from qiskit import QuantumCircuit
        >>> qc = QuantumCircuit(1, 1)
        >>> qc.h(0)
        >>> qc.measure(0, 0)
        >>> results = run_circuits_local_simulator([qc], shots=100)
    """
    if not circuits:
        st.warning("No circuits provided to simulate.")
        return []

    st.info(f"Starting local simulation of {len(circuits)} circuits with {shots} shots each...")
    try:
        # Create simulator with noise model if provided
        simulator = AerSimulator(noise_model=noise_model) if noise_model else AerSimulator()
        
        # Transpile circuits for the simulator
        compiled_circuits = transpile(circuits, simulator, optimization_level=optimization_level)

        # Run the simulation
        job = simulator.run(compiled_circuits, shots=shots)
        result = job.result()

        # Process results
        measured_bits = []
        progress_bar = st.progress(0, text="Processing simulation results...")
        total_circuits = len(circuits)

        for i, circ in enumerate(circuits):
            counts = result.get_counts(i)
            if not counts:
                st.warning(f"Circuit {i} produced no measurement counts.")
                measured_bits.append(-1)
            else:
                measured_bit_str = max(counts, key=counts.get)
                measured_bits.append(int(measured_bit_str))

            progress_bar.progress((i + 1) / total_circuits, 
                                 text=f"Processing simulation results... ({i+1}/{total_circuits})")

        st.success(f"Local simulation completed successfully.")
        return measured_bits

    except Exception as e:
        st.error(f"Local simulation failed: {e}")
        logger.exception("Error in local simulation")
        return None

def run_circuits_with_eve_local_simulator(circuits: List[QuantumCircuit], 
                                         shots: int = 1) -> Optional[Tuple[List[int], List[int]]]:
    """
    Run quantum circuits with Eve's measurements on the local simulator.
    
    This function is specifically for circuits created with create_bb84_circuits_with_eve()
    that have two classical bits (one for Eve, one for Bob).
    
    Args:
        circuits: List of quantum circuits with Eve's measurements
        shots: Number of shots per circuit
        
    Returns:
        Tuple of (eve_bits, bob_bits) or None if simulation fails
        
    Examples:
        >>> from qkd_simulation.quantum.circuits import create_bb84_circuits_with_eve
        >>> circuits = create_bb84_circuits_with_eve(alice_bits, alice_bases, eve_bases, bob_bases)
        >>> eve_bits, bob_bits = run_circuits_with_eve_local_simulator(circuits)
    """
    if not circuits:
        st.warning("No circuits provided to simulate.")
        return None

    st.info(f"Starting local simulation with Eve of {len(circuits)} circuits with {shots} shots each...")
    try:
        simulator = AerSimulator()
        compiled_circuits = transpile(circuits, simulator)

        job = simulator.run(compiled_circuits, shots=shots)
        result = job.result()

        eve_bits = []
        bob_bits = []
        progress_bar = st.progress(0, text="Processing simulation results...")
        total_circuits = len(circuits)

        for i, circ in enumerate(circuits):
            counts = result.get_counts(i)
            if not counts:
                st.warning(f"Circuit {i} produced no measurement counts.")
                eve_bits.append(-1)
                bob_bits.append(-1)
            else:
                # Get the most frequent result (format: "eve_bit bob_bit")
                most_frequent = max(counts, key=counts.get)
                # Parse the result - in reverse order due to qiskit's bit ordering
                bits = most_frequent.split()
                if len(bits) >= 2:
                    eve_bits.append(int(bits[0]))
                    bob_bits.append(int(bits[1]))
                else:
                    st.warning(f"Unexpected result format for circuit {i}: {most_frequent}")
                    eve_bits.append(-1)
                    bob_bits.append(-1)

            progress_bar.progress((i + 1) / total_circuits, 
                                 text=f"Processing simulation results... ({i+1}/{total_circuits})")

        st.success(f"Local simulation with Eve completed successfully.")
        return eve_bits, bob_bits

    except Exception as e:
        st.error(f"Local simulation with Eve failed: {e}")
        logger.exception("Error in local simulation with Eve")
        return None

def run_circuits_ibm_runtime(service: QiskitRuntimeService,
                           backend_name: str,
                           circuits: List[QuantumCircuit],
                           shots: int = 1,
                           optimization_level: int = 1) -> Optional[List[int]]:
    """
    Run quantum circuits on an IBM Quantum backend using Qiskit Runtime.
    
    Args:
        service: Initialized QiskitRuntimeService object
        backend_name: Name of the IBM Quantum backend to use
        circuits: List of quantum circuits to run
        shots: Number of shots per circuit
        optimization_level: Transpiler optimization level (0-3)
        
    Returns:
        List of measured bits (one per circuit), or None if execution fails
        
    Examples:
        >>> from qiskit_ibm_runtime import QiskitRuntimeService
        >>> service = QiskitRuntimeService(channel="ibm_quantum", token="YOUR_TOKEN")
        >>> results = run_circuits_ibm_runtime(service, "ibm_brisbane", circuits, shots=1)
    """
    st.info(f"Preparing to run {len(circuits)} circuits on IBM backend '{backend_name}' ({shots} shots each)...")

    try:
        # Get the backend
        backend = service.backend(backend_name)
        st.write(f"Backend '{backend_name}' status: {backend.status().status_msg}")
        if not backend.status().operational:
             st.error(f"Backend '{backend_name}' is not operational. Status: {backend.status().status_msg}")
             return None

        # Transpile circuits for the backend
        st.write(f"Transpiling {len(circuits)} circuits for backend '{backend_name}'...")
        transpiled_circuits = transpile(circuits, backend=backend, optimization_level=optimization_level)
        st.write("Transpilation complete.")

        # Create sampler
        sampler = Sampler(mode=backend)

        # Submit job
        st.info("Submitting job to SamplerV2...")
        job = sampler.run(transpiled_circuits, shots=shots)
        job_id = job.job_id()
        st.info(f"Job submitted successfully. Job ID: {job_id}")
        st.write("Waiting for job to complete...")

        # Monitor job status
        start_time = time.time()
        last_status = None
        status_placeholder = st.empty()
        progress_placeholder = st.empty()

        while job.status() not in ["DONE", "ERROR", "CANCELLED"]:
            current_status = job.status()
            current_status_upper = current_status.upper()
            if current_status_upper != last_status:
                status_placeholder.info(f"Job Status: {current_status_upper} (Elapsed: {time.time() - start_time:.1f}s)")
                last_status = current_status_upper

            time.sleep(5)

        # Process final status
        final_status = job.status()
        final_status_upper = final_status.upper()
        status_placeholder.info(f"Job final status: {final_status_upper} (Total time: {time.time() - start_time:.1f}s)")

        if final_status_upper == "DONE":
            st.success("Job completed successfully.")
            result = job.result()

            # Process results
            measured_bits = []
            progress_bar = st.progress(0, text="Processing IBM results...")
            total_results = len(result)

            for i, pub_result in enumerate(result):
                try:
                    counts = pub_result.data.c.get_counts()
                    if not counts:
                        st.warning(f"Circuit {i} (Pub {i}) produced no measurement counts.")
                        measured_bits.append(-1)
                    else:
                        measured_bit_str = max(counts, key=counts.get)
                        measured_bits.append(int(measured_bit_str))
                except AttributeError:
                     st.warning(f"Could not extract counts for classical register 'c' in Pub {i}.")
                     measured_bits.append(-1)
                except Exception as pub_e:
                    st.warning(f"Error processing results for Pub {i}: {pub_e}")
                    measured_bits.append(-1)

                progress_bar.progress((i + 1) / total_results, 
                                     text=f"Processing IBM results... ({i+1}/{total_results})")

            return measured_bits

        else:
            st.error(f"Job {job_id} failed or was cancelled. Status: {final_status_upper}")
            try:
                 st.error(f"Error message: {job.error_message()}")
            except:
                 st.error("Could not retrieve specific error message.")
            return None

    except Exception as e:
        st.error(f"Failed to run job on IBM Runtime: {e}")
        import traceback
        st.error(traceback.format_exc())
        logger.exception("Error in IBM Runtime execution")
        return None

def run_circuits_batch(circuits: List[QuantumCircuit], 
                      batch_size: int = 20,
                      backend: Optional[Union[str, Backend]] = None,
                      service: Optional[QiskitRuntimeService] = None,
                      shots: int = 1) -> Optional[List[int]]:
    """
    Run quantum circuits in batches to improve performance with many circuits.
    
    Args:
        circuits: List of quantum circuits to run
        batch_size: Number of circuits per batch
        backend: Backend to use (string name or Backend object)
                 If None, uses local simulator
        service: QiskitRuntimeService object (required if backend is an IBM backend)
        shots: Number of shots per circuit
        
    Returns:
        List of measured bits (one per circuit), or None if execution fails
        
    Examples:
        >>> results = run_circuits_batch(circuits, batch_size=50)  # Local simulator
        >>> results = run_circuits_batch(circuits, batch_size=20, backend="ibm_brisbane", 
        ...                             service=service)  # IBM backend
    """
    if not circuits:
        st.warning("No circuits provided to run.")
        return []
    
    total_circuits = len(circuits)
    st.info(f"Running {total_circuits} circuits in batches of {batch_size}...")
    
    # Create batches
    batches = [circuits[i:i+batch_size] for i in range(0, total_circuits, batch_size)]
    st.info(f"Created {len(batches)} batches")
    
    all_results = []
    progress_bar = st.progress(0, text="Processing batches...")
    
    for i, batch in enumerate(batches):
        st.write(f"Running batch {i+1}/{len(batches)} ({len(batch)} circuits)...")
        
        if backend is None:
            # Use local simulator
            batch_results = run_circuits_local_simulator(batch, shots=shots)
        elif isinstance(backend, str) and service is not None:
            # Use IBM backend
            batch_results = run_circuits_ibm_runtime(service, backend, batch, shots=shots)
        else:
            st.error("Invalid backend configuration")
            return None
        
        if batch_results is None:
            st.error(f"Batch {i+1} failed. Aborting.")
            return None
        
        all_results.extend(batch_results)
        progress_bar.progress((i + 1) / len(batches), 
                             text=f"Processing batches... ({i+1}/{len(batches)})")
    
    st.success(f"All {len(batches)} batches completed successfully.")
    return all_results

def create_noise_model(error_probabilities: Dict[str, float] = None) -> object:
    """
    Create a simple noise model for simulation.
    
    Args:
        error_probabilities: Dictionary with error probabilities for different operations
                            Keys can include 'readout', 'gate1', 'gate2', 'thermal'
        
    Returns:
        Noise model object for use with AerSimulator
        
    Examples:
        >>> noise_model = create_noise_model({'readout': 0.01, 'gate1': 0.005})
        >>> results = run_circuits_local_simulator(circuits, noise_model=noise_model)
    """
    try:
        from qiskit_aer.noise import NoiseModel
        from qiskit_aer.noise.errors import readout_error, depolarizing_error
        from qiskit.providers.fake_provider import FakeProvider
    except ImportError:
        st.error("Noise modeling requires qiskit-aer. Install with 'pip install qiskit-aer'")
        return None
    
    # Default error probabilities
    default_probs = {
        'readout': 0.01,    # 1% readout error
        'gate1': 0.001,     # 0.1% single-qubit gate error
        'gate2': 0.01,      # 1% two-qubit gate error
        'thermal': 0.005    # 0.5% thermal relaxation
    }
    
    # Use provided probabilities or defaults
    probs = default_probs.copy()
    if error_probabilities:
        probs.update(error_probabilities)
    
    # Create noise model
    noise_model = NoiseModel()
    
    # Add readout error
    if probs['readout'] > 0:
        # Simple symmetric readout error
        error = readout_error([probs['readout'], 1-probs['readout']], 
                             [1-probs['readout'], probs['readout']])
        noise_model.add_all_qubit_readout_error(error)
    
    # Add gate errors
    if probs['gate1'] > 0:
        # Depolarizing error for single-qubit gates
        error = depolarizing_error(probs['gate1'], 1)
        noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3', 'x', 'y', 'z', 'h'])
    
    if probs['gate2'] > 0:
        # Depolarizing error for two-qubit gates
        error = depolarizing_error(probs['gate2'], 2)
        noise_model.add_all_qubit_quantum_error(error, ['cx', 'cz', 'swap'])
    
    return noise_model
