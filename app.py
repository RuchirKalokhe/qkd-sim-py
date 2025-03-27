# ./app.py

import streamlit as st
import numpy as np
import pandas as pd
import time
import random # For QBER sampling choice

# Import Qiskit and Runtime Service
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit # For potential circuit display if needed

# Import our simulation modules
from qkd_simulation.protocols import bb84
from qkd_simulation.quantum import circuits as qkd_circuits
from qkd_simulation.quantum import runner as qkd_runner
from qkd_simulation.classical import processing as qkd_processing
from qkd_simulation.utils import helpers

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="BB84 QKD Simulator")

# --- Constants for Display ---
Z_BASIS_SYM = "▬" # Rectilinear (Z)
X_BASIS_SYM = "✚" # Diagonal (X)
BASIS_MAP = {0: Z_BASIS_SYM, 1: X_BASIS_SYM}
QUBIT_MAP = {
    (0, 0): "|0⟩", (1, 0): "|1⟩", # Bit, Basis (Z)
    (0, 1): "|+⟩", (1, 1): "|−⟩"  # Bit, Basis (X)
}
QBER_THRESHOLD = 0.15 # Example threshold for aborting protocol
QBER_SAMPLE_FRACTION = 0.30 # Fixed 30% for QBER check

# --- Initialize Session State ---
# Store results here to prevent loss on reruns from widget changes (except button)
if 'simulation_run_completed' not in st.session_state:
    st.session_state.simulation_run_completed = False
if 'run_params' not in st.session_state:
    st.session_state.run_params = {}
if 'results' not in st.session_state:
    st.session_state.results = {}


# --- IBM Quantum Service Initialization (runs only once per session unless error) ---
# Store service in session state to avoid re-initializing unnecessarily
if 'ibm_service' not in st.session_state:
    st.session_state.ibm_service = None
    st.session_state.ibm_token_present = False
    st.session_state.available_ibm_backends = []
    st.session_state.ibm_init_error = None

    try:
        if 'IBM_QUANTUM_TOKEN' in st.secrets:
            api_token = st.secrets["IBM_QUANTUM_TOKEN"]
            if api_token and api_token != "YOUR_API_TOKEN_HERE":
                try:
                    service_instance = QiskitRuntimeService(channel="ibm_quantum", token=api_token)
                    # Attempt to list backends as verification
                    all_backends = service_instance.backends()

                    st.session_state.ibm_service = service_instance
                    st.session_state.ibm_token_present = True
                    st.session_state.available_ibm_backends = [
                        b.name for b in all_backends
                        if b.status().operational and ("simulator" in b.name or b.num_qubits >= 1)
                    ]
                    simulators = [b for b in st.session_state.available_ibm_backends if "simulator" in b]
                    real_devices = sorted([b for b in st.session_state.available_ibm_backends if "simulator" not in b])
                    st.session_state.available_ibm_backends = sorted(simulators) + real_devices
                    st.sidebar.success("IBM Quantum Service Initialized.")

                except Exception as e:
                    st.session_state.ibm_init_error = f"IBM Service Error: {e}"
                    st.sidebar.error(st.session_state.ibm_init_error)
            else:
                st.sidebar.warning("IBM_QUANTUM_TOKEN found but seems invalid/placeholder.")
        else:
            st.sidebar.warning("IBM_QUANTUM_TOKEN not found in secrets. IBM execution disabled.")
    except KeyError:
        st.sidebar.warning("Streamlit secrets not configured locally or `IBM_QUANTUM_TOKEN` missing. IBM execution disabled.")
    except Exception as e:
        st.session_state.ibm_init_error = f"Error initializing IBM Service: {e}"
        st.sidebar.error(st.session_state.ibm_init_error)

# Retrieve from session state for use in UI
service = st.session_state.ibm_service
available_ibm_backends = st.session_state.available_ibm_backends

# Check if target backend is available
TARGET_BACKEND = "ibm_brisbane"
target_backend_available = TARGET_BACKEND in available_ibm_backends
if service and not target_backend_available:
    st.sidebar.warning(f"Target backend '{TARGET_BACKEND}' is not currently available or operational.")


# --- Streamlit UI ---
st.title("BB84 Quantum Key Distribution (QKD) Simulator")
st.markdown("""
Simulate the BB84 protocol between Alice and Bob, optionally with an eavesdropper (Eve).
Choose parameters, run the simulation locally or on IBM Quantum, and observe the results.
**Note:** Changing parameters requires pressing "Run Simulation" again.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Simulation Parameters")
num_bits = st.sidebar.number_input("Number of Qubits (Bits for Key):", min_value=4, max_value=100, value=20, step=4, key="num_bits_input")

include_eve = st.sidebar.checkbox("Include Eavesdropper (Eve)?", key="include_eve_input")
eve_strategy = "Intercept-Resend (Random Basis) - Conceptual" # Updated description

backend_options = ["Local Simulator (qiskit-aer)"]
if service and available_ibm_backends:
    # Ensure target backend is listed if available
    if target_backend_available and TARGET_BACKEND not in backend_options:
         backend_options.append(TARGET_BACKEND)
    # Add other available backends, avoiding duplicates
    other_backends = [b for b in available_ibm_backends if b != TARGET_BACKEND]
    backend_options.extend(other_backends)

# Set default index for backend selection
default_backend_index = 0
if target_backend_available:
    try:
        default_backend_index = backend_options.index(TARGET_BACKEND)
    except ValueError:
        default_backend_index = 0 # Fallback to local if target not found unexpectedly

selected_backend = st.sidebar.selectbox(
    "Select Execution Backend:",
    backend_options,
    index=default_backend_index, # Default to target or local
    key="selected_backend_input"
)

shots = 1 # For QKD, each qubit transmission is unique, so 1 shot per circuit.

run_simulation = st.button("Run QKD Simulation")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Basis Legend:** {Z_BASIS_SYM} = Z (Rectilinear), {X_BASIS_SYM} = X (Diagonal)")
st.sidebar.markdown(f"**Qubit Legend:** |0⟩, |1⟩ (Z-basis); |+⟩, |−⟩ (X-basis)")
st.sidebar.markdown(f"**QBER Check:** Fixed at {QBER_SAMPLE_FRACTION*100:.0f}% of sifted key.")


# --- Simulation Logic (Runs ONLY when button is pressed) ---
if run_simulation:
    st.session_state.simulation_run_completed = False # Reset flag
    st.session_state.results = {} # Clear previous results
    st.session_state.run_params = { # Store params used for this run
        'num_bits': num_bits,
        'include_eve': include_eve,
        'selected_backend': selected_backend,
        'target_backend': TARGET_BACKEND if selected_backend == TARGET_BACKEND else None
    }

    run_container = st.container() # Use a container for output of this run
    run_container.header("Simulation Run")

    # --- Step 1: Alice ---
    run_container.subheader("1. Alice Prepares Qubits")
    alice_bits, alice_bases = bb84.generate_bits_and_bases(num_bits)
    alice_states = [QUBIT_MAP[(bit, basis)] for bit, basis in zip(alice_bits, alice_bases)]
    alice_basis_syms = [BASIS_MAP[b] for b in alice_bases]
    alice_df = pd.DataFrame({
        'Index': range(num_bits), 'Alice Bit': alice_bits,
        'Alice Basis': alice_basis_syms, 'Sent State': alice_states
    })
    st.session_state.results['alice_df'] = alice_df
    st.session_state.results['alice_bits'] = alice_bits # Need for later
    st.session_state.results['alice_bases'] = alice_bases # Need for later
    run_container.success(f"Alice generated {num_bits} random bits and bases.")

    # --- Step 1.5: Eve (Conceptual) ---
    eve_bases = None
    if include_eve:
        run_container.subheader("1.5. Eve Intercepts & Measures (!)")
        run_container.warning(f"Eve intercepts using strategy: **{eve_strategy}**")
        eve_bases = bb84.generate_bases(num_bits)
        eve_basis_syms = [BASIS_MAP[b] for b in eve_bases]
        eve_df = pd.DataFrame({
            'Index': range(num_bits), 'Alice Sent State': alice_states,
            'Eve Measure Basis': eve_basis_syms
        })
        st.session_state.results['eve_df'] = eve_df
        run_container.markdown("_**Conceptual:** Eve measures, potentially disturbing the state. This simulation **does not** modify the circuits passed to the backend based on Eve's actions. Her impact is primarily seen in increased QBER on noisy backends._")

    # --- Step 2: Bob Bases ---
    run_container.subheader("2. Bob Chooses Measurement Bases")
    bob_bases = bb84.generate_bases(num_bits)
    bob_basis_syms = [BASIS_MAP[b] for b in bob_bases]
    bob_bases_df = pd.DataFrame({'Index': range(num_bits), 'Bob Measure Basis': bob_basis_syms})
    st.session_state.results['bob_bases_df'] = bob_bases_df
    st.session_state.results['bob_bases'] = bob_bases # Need for later
    run_container.success(f"Bob randomly chose {num_bits} measurement bases.")

    # --- Step 3: Quantum Transmission ---
    run_container.subheader("3. Quantum Transmission & Measurement")
    bob_measured_bits_np = None
    sim_success = False
    # CORRECTED LINE: Use st.spinner directly
    with st.spinner("Preparing and running quantum circuits..."):
        try:
            qkd_circs = qkd_circuits.create_bb84_circuits(alice_bits, alice_bases, bob_bases)
            st.session_state.results['sample_circuit_0'] = qkd_circs[0].draw(output='text') if qkd_circs else None

            # --- Select and Run Backend ---
            # (Code for selecting local vs IBM remains the same)
            if selected_backend == "Local Simulator (qiskit-aer)":
                # Use run_container inside the runner if you want messages there,
                # OR keep using st.info/error/etc. - they will appear in run_container anyway.
                measured_bits_list = qkd_runner.run_circuits_local_simulator(qkd_circs, shots=shots)
            elif service and selected_backend in available_ibm_backends:
                if service:
                     measured_bits_list = qkd_runner.run_circuits_ibm_runtime(service, selected_backend, qkd_circs, shots=shots)
                else:
                     # Use run_container.error if you want error specifically inside container
                     # Or just st.error - it will appear in the container context
                     st.error("IBM Service not initialized. Cannot run on IBM backend.")
                     measured_bits_list = None
            else:
                st.error(f"Selected backend '{selected_backend}' is not available or configured.")
                measured_bits_list = None

            # --- Process Results ---
            # (Code for processing results remains the same)
            # ... [rest of the try block] ...

        except Exception as e:
            # Use st.error or run_container.error
            st.error(f"An error occurred during quantum step: {e}")
            import traceback
            st.error(traceback.format_exc()) # Use st.error for tracebacks usually

    # --- Steps 4-7 (Run only if Quantum Step Succeeded) ---
    if sim_success:
        try:
            # --- Step 4: Basis Comparison ---
            run_container.subheader("4. Public Discussion: Basis Comparison")
            res = st.session_state.results # Shortcut
            matching_indices, mismatching_indices = bb84.compare_bases(res['alice_bases'], res['bob_bases'])
            res['matching_indices'] = matching_indices
            res['mismatching_indices'] = mismatching_indices
            num_matches = len(matching_indices)
            num_mismatches = len(mismatching_indices)
            run_container.write(f"- Bases matched at **{num_matches}** positions.")
            run_container.write(f"- Bases mismatched at **{num_mismatches}** positions.")
            compare_df = pd.DataFrame({
                'Index': range(num_bits),
                'Alice Basis': [BASIS_MAP[b] for b in res['alice_bases']],
                'Bob Basis': [BASIS_MAP[b] for b in res['bob_bases']],
                'Match?': ['✅ Yes' if i in matching_indices else '❌ No' for i in range(num_bits)]
            })
            res['compare_df'] = compare_df
            run_container.success("Basis comparison complete.")

            # --- Step 5: Sifting ---
            run_container.subheader("5. Sifting the Key")
            if num_matches > 0:
                alice_sifted_key, bob_sifted_key = qkd_processing.sift_key(res['alice_bits'], res['bob_measured_bits_np'], matching_indices)
                sifted_len = len(alice_sifted_key)
                res['alice_sifted_key'] = alice_sifted_key
                res['bob_sifted_key'] = bob_sifted_key
                res['sifted_len'] = sifted_len
                run_container.write(f"Length of sifted key: **{sifted_len} bits**")
                sifted_df = pd.DataFrame({
                    'Sifted Index': range(sifted_len), 'Original Index': matching_indices,
                    'Alice Bit': alice_sifted_key, 'Bob Bit': bob_sifted_key,
                    'Agree?': ['✅ Yes' if a == b else '❌ NO!' for a,b in zip(alice_sifted_key, bob_sifted_key)]
                })
                res['sifted_df'] = sifted_df
                run_container.success("Sifting complete.")
            else:
                run_container.warning("No matching bases. Cannot create sifted key.")
                sifted_len = 0 # Ensure sifted_len is 0 if no matches
                res['sifted_len'] = 0

            # --- Step 6: QBER Estimation ---
            run_container.subheader("6. Estimate QBER")
            if sifted_len > 0:
                qber, qber_indices, remaining_indices = qkd_processing.estimate_qber(
                    res['alice_sifted_key'], res['bob_sifted_key'], sample_fraction=QBER_SAMPLE_FRACTION
                )
                num_qber_bits = len(qber_indices)
                num_final_bits = len(remaining_indices)
                res['qber'] = qber
                res['qber_indices'] = qber_indices
                res['remaining_indices'] = remaining_indices
                res['num_qber_bits'] = num_qber_bits
                res['num_final_bits'] = num_final_bits

                run_container.metric(label="Estimated QBER", value=f"{qber:.2%}")
                qber_details_df = pd.DataFrame({
                     'Sifted Index (for QBER)': qber_indices,
                     'Alice Bit': res['alice_sifted_key'][qber_indices],
                     'Bob Bit': res['bob_sifted_key'][qber_indices],
                     'Match?': ['✅ Yes' if a == b else '❌ ERROR' for a,b in zip(res['alice_sifted_key'][qber_indices], res['bob_sifted_key'][qber_indices])]
                })
                res['qber_details_df'] = qber_details_df

                # Explain potential 0% QBER
                if qber == 0.0 and not include_eve and "simulator" in selected_backend.lower():
                     run_container.info("QBER is 0%. This is expected for a noiseless simulation without an active eavesdropper modifying the qubits.")
                elif qber == 0.0 and include_eve and "simulator" in selected_backend.lower():
                     run_container.warning("QBER is 0% despite Eve being 'included'. This is because this simulation **does not modify the quantum circuits** based on Eve's actions. Run on real hardware or a noisy simulator to see Eve's impact via increased QBER.")
                elif qber > 0.0:
                     run_container.info(f"QBER is {qber:.2%}. This indicates noise in the channel or potential eavesdropping.")


                if qber > QBER_THRESHOLD:
                    run_container.error(f"QBER ({qber:.2%}) > Threshold ({QBER_THRESHOLD:.2%}). **Protocol Aborted.**")
                    res['protocol_aborted'] = True
                else:
                    run_container.success(f"QBER ({qber:.2%}) is acceptable.")
                    res['protocol_aborted'] = False
            else:
                run_container.warning("Sifted key empty. Cannot estimate QBER.")
                res['qber'] = None
                res['protocol_aborted'] = True # Treat as aborted if no key

            # --- Step 7: Final Key ---
            run_container.subheader("7. Final Shared Secret Key")
            if not res.get('protocol_aborted', True) and res.get('num_final_bits', 0) > 0:
                final_key_alice = qkd_processing.extract_final_key(res['alice_sifted_key'], res['remaining_indices'])
                final_key_bob = qkd_processing.extract_final_key(res['bob_sifted_key'], res['remaining_indices'])
                res['final_key_alice_str'] = helpers.bits_to_string(final_key_alice)
                res['final_key_bob_str'] = helpers.bits_to_string(final_key_bob)
                res['final_keys_match'] = np.array_equal(final_key_alice, final_key_bob)
                run_container.write(f"Final Key Length: **{res['num_final_bits']} bits**")
            else:
                 run_container.warning("Protocol aborted or no bits remaining. No final key generated.")

            # --- Add Debugging Info ---
            st.write("DEBUG: Classical processing finished without error.")
            # --------------------------

            st.session_state.simulation_run_completed = True # Mark run as complete
            run_container.success("Full simulation cycle finished.")
            st.balloons()

        except Exception as e:
            # --- Improve Error Reporting ---
            st.error(f"ERROR during classical post-processing (Steps 4-7): {e}")
            import traceback
            st.error(traceback.format_exc()) # Ensure full traceback is shown
            # -----------------------------
            st.session_state.simulation_run_completed = False # Mark as incomplete on error

# --- Display Area (Shows results from session_state if available) ---
st.write(f"DEBUG: Checking display. Completion flag is: {st.session_state.get('simulation_run_completed', 'Not Set')}") # Add this line
if st.session_state.simulation_run_completed:
    st.write("DEBUG: Entering display section.") # Add this line
    st.header("Last Simulation Results")
    res = st.session_state.results
    params = st.session_state.run_params
    st.info(f"Showing results for run with: {params['num_bits']} bits, Eve included: {params['include_eve']}, Backend: '{params['selected_backend']}'")

    # Display Step 1
    st.subheader("1. Alice's Preparation")
    if 'alice_df' in res:
        with st.expander("Details", expanded=False):
            st.dataframe(res['alice_df'], use_container_width=True)

    # Display Step 1.5
    if 'eve_df' in res:
        st.subheader("1.5. Eve's Actions (Conceptual)")
        with st.expander("Details", expanded=False):
            st.dataframe(res['eve_df'], use_container_width=True)
            st.markdown("_**Conceptual:** Eve measures, potentially disturbing the state. This simulation **does not** modify the circuits passed to the backend based on Eve's actions. Her impact is primarily seen in increased QBER on noisy backends._")

    # Display Step 2
    st.subheader("2. Bob's Basis Choices")
    if 'bob_bases_df' in res:
         with st.expander("Details", expanded=False):
            st.dataframe(res['bob_bases_df'], use_container_width=True)

    # Display Step 3
    st.subheader("3. Quantum Transmission & Measurement")
    if 'bob_measured_bits_np' in res:
         bob_res_df = pd.DataFrame({
             'Index': range(params['num_bits']),
             'Bob Measure Basis': [BASIS_MAP[b] for b in res['bob_bases']],
             'Bob Measured Bit': res['bob_measured_bits_np']
         })
         with st.expander("Bob's Measurement Results", expanded=False):
             st.dataframe(bob_res_df, use_container_width=True)
    if 'sample_circuit_0' in res and res['sample_circuit_0']:
         with st.expander("Sample Quantum Circuit (Qubit 0)", expanded=False):
            st.code(res['sample_circuit_0'], language='text')

    # Display Step 4
    st.subheader("4. Basis Comparison")
    if 'compare_df' in res:
        st.write(f"- Bases matched at **{len(res.get('matching_indices',[]))}** positions.")
        st.write(f"- Bases mismatched at **{len(res.get('mismatching_indices',[]))}** positions.")
        with st.expander("Details", expanded=False):
            st.dataframe(res['compare_df'], use_container_width=True)

    # Display Step 5
    st.subheader("5. Sifted Key")
    if 'sifted_df' in res:
        st.write(f"Length of sifted key: **{res.get('sifted_len', 0)} bits**")
        with st.expander("Details", expanded=False):
            st.dataframe(res['sifted_df'], use_container_width=True)
            st.markdown("_Disagreements here indicate errors (noise or Eve!)._")
    elif 'sifted_len' in res and res['sifted_len'] == 0:
         st.warning("No matching bases found. Sifted key is empty.")


    # Display Step 6
    st.subheader("6. QBER Estimation")
    if 'qber' in res and res['qber'] is not None:
        st.metric(label="Estimated QBER", value=f"{res['qber']:.2%}")
        if 'qber_details_df' in res:
            with st.expander(f"QBER Sample Details ({res.get('num_qber_bits', 0)} bits)", expanded=False):
                 st.dataframe(res['qber_details_df'], use_container_width=True)

        # Repeat QBER explanation
        if res['qber'] == 0.0 and not params['include_eve'] and "simulator" in params['selected_backend'].lower():
             st.info("QBER is 0%. Expected for noiseless simulation without active eavesdropping.")
        elif res['qber'] == 0.0 and params['include_eve'] and "simulator" in params['selected_backend'].lower():
             st.warning("QBER is 0% despite Eve being 'included', as this simulation doesn't modify quantum circuits for Eve's actions.")
        elif res['qber'] > 0.0:
             st.info(f"QBER is {res['qber']:.2%}. Indicates noise or potential eavesdropping.")

        if res.get('protocol_aborted', False):
            st.error(f"QBER ({res['qber']:.2%}) likely exceeded threshold ({QBER_THRESHOLD:.2%}). **Protocol Aborted.**")
        else:
            st.success(f"QBER ({res['qber']:.2%}) is acceptable.")

    elif 'sifted_len' in res and res['sifted_len'] == 0:
         st.warning("Sifted key empty. Cannot estimate QBER.")

    # Display Step 7
    st.subheader("7. Final Shared Secret Key")
    if not res.get('protocol_aborted', True) and 'final_key_alice_str' in res:
        st.write(f"Final Key Length: **{res.get('num_final_bits', 0)} bits**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Alice's Final Key:**")
            st.code(res['final_key_alice_str'], language='text')
        with col2:
            st.markdown("**Bob's Final Key:**")
            st.code(res['final_key_bob_str'], language='text')

        if res.get('final_keys_match', False):
            st.success("✅ Alice's and Bob's final keys match!")
        else:
            st.error("❌ Discrepancy! Final keys DO NOT match. Errors remain.")
            st.markdown("_Note: Error Correction would typically fix this._")
    else:
        st.warning("Protocol aborted or no bits remaining. No final key generated.")

else:
    st.info("Configure parameters in the sidebar and click 'Run QKD Simulation' to start.") # This might be showing instead