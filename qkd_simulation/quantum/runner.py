import time
from typing import List, Optional
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
import streamlit as st

def run_circuits_local_simulator(circuits: List[QuantumCircuit], shots: int) -> Optional[List[int]]:
    if not circuits:
        st.warning("No circuits provided to simulate.")
        return []

    st.info(f"Starting local simulation of {len(circuits)} circuits with {shots} shots each...")
    try:
        simulator = AerSimulator()
        compiled_circuits = transpile(circuits, simulator)

        job = simulator.run(compiled_circuits, shots=shots)
        result = job.result()

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

            progress_bar.progress((i + 1) / total_circuits, text=f"Processing simulation results... ({i+1}/{total_circuits})")

        st.success(f"Local simulation completed successfully.")
        return measured_bits

    except Exception as e:
        st.error(f"Local simulation failed: {e}")
        return None

def run_circuits_ibm_runtime(service: QiskitRuntimeService,
                             backend_name: str,
                             circuits: List[QuantumCircuit],
                             shots: int) -> Optional[List[int]]:
    st.info(f"Preparing to run {len(circuits)} circuits on IBM backend '{backend_name}' ({shots} shots each)...")

    try:
        backend = service.backend(backend_name)
        st.write(f"Backend '{backend_name}' status: {backend.status().status_msg}")
        if not backend.status().operational:
             st.error(f"Backend '{backend_name}' is not operational. Status: {backend.status().status_msg}")
             return None

        st.write(f"Transpiling {len(circuits)} circuits for backend '{backend_name}'...")
        transpiled_circuits = transpile(circuits, backend=backend, optimization_level=1)
        st.write("Transpilation complete.")

        sampler = Sampler(mode=backend)

        st.info("Submitting job to SamplerV2...")
        job = sampler.run(transpiled_circuits, shots=shots)
        job_id = job.job_id()
        st.info(f"Job submitted successfully. Job ID: {job_id}")
        st.write("Waiting for job to complete...")

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

        final_status = job.status()
        final_status_upper = final_status.upper()
        status_placeholder.info(f"Job final status: {final_status_upper} (Total time: {time.time() - start_time:.1f}s)")

        if final_status_upper == "DONE":
            st.success("Job completed successfully.")
            result = job.result()

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
        st.error(traceback.format_exc())
        return None