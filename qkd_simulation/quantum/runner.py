# Functions to run circuits (simulators, IBM Runtime)
# ./qkd_simulation/quantum/runner.py

import time
from typing import List, Optional
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator # For local simulation
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler # Use SamplerV2
import streamlit as st # Import streamlit for progress updates

# --- Local Simulator Execution ---

def run_circuits_local_simulator(circuits: List[QuantumCircuit], shots: int) -> Optional[List[int]]:
    """
    Runs a list of quantum circuits on a local Aer simulator.

    Args:
        circuits: A list of QuantumCircuit objects.
        shots: The number of times to run each circuit.

    Returns:
        A list of measured bits (0 or 1) corresponding to each circuit,
        or None if simulation fails. The bit is determined by the most
        frequent outcome for the single classical bit in each circuit.
    """
    if not circuits:
        st.warning("No circuits provided to simulate.")
        return []

    st.info(f"Starting local simulation of {len(circuits)} circuits with {shots} shots each...")
    try:
        simulator = AerSimulator()
        # Transpile circuits for optimization (good practice)
        compiled_circuits = transpile(circuits, simulator)

        # Run the job
        job = simulator.run(compiled_circuits, shots=shots)
        result = job.result() # This blocks until the job is done

        measured_bits = []
        progress_bar = st.progress(0, text="Processing simulation results...")
        total_circuits = len(circuits)

        for i, circ in enumerate(circuits):
            counts = result.get_counts(i) # Get counts for the i-th circuit
            # Determine the measured bit - assumes one classical bit output '0' or '1'
            # If counts are {'0': X, '1': Y}, finds the key ('0' or '1') with max value
            if not counts:
                 # Handle rare case of no counts (e.g., 0 shots, though unlikely here)
                 st.warning(f"Circuit {i} produced no measurement counts.")
                 measured_bits.append(-1) # Indicate an error or undefined state
            else:
                 # Find the outcome ('0' or '1') with the highest count
                 measured_bit_str = max(counts, key=counts.get)
                 measured_bits.append(int(measured_bit_str))

            # Update progress bar
            progress_bar.progress((i + 1) / total_circuits, text=f"Processing simulation results... ({i+1}/{total_circuits})")

        st.success(f"Local simulation completed successfully.")
        return measured_bits

    except Exception as e:
        st.error(f"Local simulation failed: {e}")
        return None

# --- IBM Quantum Runtime Execution ---

def run_circuits_ibm_runtime(service: QiskitRuntimeService,
                             backend_name: str,
                             circuits: List[QuantumCircuit],
                             shots: int) -> Optional[List[int]]:
    # ... (docstring and initial checks remain the same) ...
    """
    Runs a list of quantum circuits using IBM Quantum Runtime Service (SamplerV2).

    Args:
        service: Initialized QiskitRuntimeService object.
        backend_name: The name of the IBM backend (e.g., "simulator_stabilizer", "ibm_brisbane").
        circuits: A list of QuantumCircuit objects.
        shots: The number of times to run each circuit.

    Returns:
        A list of measured bits (0 or 1) corresponding to each circuit,
        or None if the job fails or is cancelled. The bit is determined
        by the most frequent outcome.
    """

    st.info(f"Preparing to run {len(circuits)} circuits on IBM backend '{backend_name}' ({shots} shots each)...")

    try:
        # --- Get Backend ---
        backend = service.backend(backend_name)
        st.write(f"Backend '{backend_name}' status: {backend.status().status_msg}")
        if not backend.status().operational:
             st.error(f"Backend '{backend_name}' is not operational. Status: {backend.status().status_msg}")
             return None

        # --- !!! ADD TRANSPILATION STEP HERE !!! ---
        st.write(f"Transpiling {len(circuits)} circuits for backend '{backend_name}'...")
        # Transpile the circuits specifically for the target backend's ISA and coupling map.
        # Optimization level 1 is generally recommended for runtime primitives.
        # This converts gates like 'h' into the backend's basis gates (e.g., sx, rz).
        transpiled_circuits = transpile(circuits, backend=backend, optimization_level=1)
        st.write("Transpilation complete.")
        # ------------------------------------------

        # --- Initialize Sampler ---
        # Use 'mode' for SamplerV2 initialization
        sampler = Sampler(mode=backend)

        # --- Submit Job ---
        st.info("Submitting job to SamplerV2...")
        # Pass the *transpiled* circuits to the sampler
        job = sampler.run(transpiled_circuits, shots=shots) # <<< USE transpiled_circuits
        job_id = job.job_id()
        st.info(f"Job submitted successfully. Job ID: {job_id}")
        st.write("Waiting for job to complete...")

        # --- Monitor Job Status ---
        start_time = time.time()
        last_status = None
        status_placeholder = st.empty() # Placeholder to update status message
        progress_placeholder = st.empty() # Placeholder for potential progress bar (if available)

        while job.status() not in ["DONE", "ERROR", "CANCELLED"]:
            current_status = job.status()
            current_status_upper = current_status.upper()
            if current_status_upper != last_status:
                # CORRECTED LINE: Use the status string directly, no .name needed
                status_placeholder.info(f"Job Status: {current_status_upper} (Elapsed: {time.time() - start_time:.1f}s)")
                last_status = current_status_upper # Store the uppercase status

            time.sleep(5) # Poll every 5 seconds

        final_status = job.status()
        final_status_upper = final_status.upper() # Use uppercase for consistency
        status_placeholder.info(f"Job final status: {final_status_upper} (Total time: {time.time() - start_time:.1f}s)")

        if final_status_upper == "DONE":
            st.success("Job completed successfully.")
            result = job.result() # Get the results object

            measured_bits = []
            progress_bar = st.progress(0, text="Processing IBM results...")
            total_results = len(result)

            # Process results using SamplerV2 structure
            for i, pub_result in enumerate(result):
                # pub_result is a PubResult object
                # pub_result.data contains DataBin objects (e.g., .c for classical register 'c')
                try:
                    # Assumes classical register is named 'c' (default for measure(q,c))
                    counts = pub_result.data.c.get_counts()
                    if not counts:
                        st.warning(f"Circuit {i} (Pub {i}) produced no measurement counts.")
                        measured_bits.append(-1) # Indicate error
                    else:
                        measured_bit_str = max(counts, key=counts.get)
                        measured_bits.append(int(measured_bit_str))
                except AttributeError:
                     # Handle cases where the 'c' register might not exist or data is missing
                     st.warning(f"Could not extract counts for classical register 'c' in Pub {i}.")
                     measured_bits.append(-1) # Indicate error

                except Exception as pub_e:
                    st.warning(f"Error processing results for Pub {i}: {pub_e}")
                    measured_bits.append(-1)

                # Update progress bar
                progress_bar.progress((i + 1) / total_results, text=f"Processing IBM results... ({i+1}/{total_results})")

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
        st.error(traceback.format_exc()) # Print detailed traceback for debugging
        return None

# Example usage (for testing purposes)
if __name__ == '__main__':
    # --- Test Local Simulation ---
    print("--- Testing Local Simulator ---")
    qc1 = QuantumCircuit(1, 1, name="test1_Z0")
    qc1.measure(0, 0) # Expect 0

    qc2 = QuantumCircuit(1, 1, name="test2_Z1")
    qc2.x(0)
    qc2.measure(0, 0) # Expect 1

    qc3 = QuantumCircuit(1, 1, name="test3_XPlus")
    qc3.h(0) # Prepare |+>
    qc3.measure(0, 0) # Expect 0 or 1 with ~50% prob

    test_circuits_local = [qc1, qc2, qc3]
    local_results = run_circuits_local_simulator(test_circuits_local, shots=100)

    if local_results is not None:
        print(f"Local Simulation Measured Bits: {local_results}")
        # Expected: [0, 1, (0 or 1)]

    # --- Test IBM Runtime (Conceptual - Requires Setup) ---
    print("\n--- Testing IBM Runtime (Conceptual) ---")
    # This part requires you to have a valid IBM Quantum token configured
    # and uncomment the necessary lines in a real execution environment (like app.py)
    IBM_QUANTUM_TOKEN_AVAILABLE = True # Set to True if you have token configured

    if IBM_QUANTUM_TOKEN_AVAILABLE:
        try:
            # Replace with your actual token loading mechanism (e.g., st.secrets or env var)
            # For testing here, you might hardcode it temporarily or load from a file
            # WARNING: Do not commit tokens to Git!
            my_token = "b6e78b7dea2ce8fd7759724ed681bf5afb6f5399254926463b379af04a707976d8cae86fe135e4d6d50c9293605a7e970f66880d1fc9a3614d1cb4aba597542a" # Replace or load securely
            service = QiskitRuntimeService(channel="ibm_quantum", token=my_token)

            # Choose a backend (simulator recommended for testing)
            # backend_name_ibm = "simulator_stabilizer"
            backend_name_ibm = "ibm_brisbane" # A generic simulator alias

            qc_ibm_1 = QuantumCircuit(1, 1, name="ibm_test1")
            qc_ibm_1.h(0)
            qc_ibm_1.measure(0,0)
            qc_ibm_2 = QuantumCircuit(1, 1, name="ibm_test2")
            qc_ibm_2.x(0)
            qc_ibm_2.h(0)
            qc_ibm_2.measure(0,0)

            test_circuits_ibm = [qc_ibm_1, qc_ibm_2]

            ibm_results = run_circuits_ibm_runtime(service, backend_name_ibm, test_circuits_ibm, shots=100)

            if ibm_results is not None:
                print(f"IBM Runtime Measured Bits: {ibm_results}")

        except Exception as e:
            print(f"Could not run IBM Runtime test: {e}")
    else:
        print("Skipping IBM Runtime test (IBM_QUANTUM_TOKEN_AVAILABLE = False).")
        print("Set IBM_QUANTUM_TOKEN_AVAILABLE to True and configure token if needed.")