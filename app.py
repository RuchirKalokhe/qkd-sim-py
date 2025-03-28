# ./app.py (Restructured for New Layout - WITH IBM Integration)

import streamlit as st
import numpy as np
import pandas as pd
import time
import random
import traceback # Added for detailed error logging

# --- Qiskit/Runtime Imports ---
# Ensure qiskit-ibm-runtime is installed: pip install qiskit-ibm-runtime
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    QISKIT_RUNTIME_AVAILABLE = True
except ImportError:
    QISKIT_RUNTIME_AVAILABLE = False
    st.error("`qiskit-ibm-runtime` package not found. Please install it (`pip install qiskit-ibm-runtime`) to use IBM Quantum backends.")

from qiskit import QuantumCircuit

# --- Simulation Module Imports ---
# Assuming these modules exist in the specified structure
try:
    from qkd_simulation.protocols import bb84
    from qkd_simulation.quantum import circuits as qkd_circuits
    from qkd_simulation.quantum import runner as qkd_runner
    from qkd_simulation.classical import processing as qkd_processing
    from qkd_simulation.utils import helpers
    SIMULATION_MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import simulation modules: {e}. Make sure `qkd_simulation` package is in your Python path.")
    SIMULATION_MODULES_AVAILABLE = False

# --- Page Config ---
st.set_page_config(layout="wide", page_title="BB84 QKD Simulator")

# --- Constants ---
Z_BASIS_SYM = "▬" # Rectilinear (Z)
X_BASIS_SYM = "✚" # Diagonal (X)
BASIS_MAP = {0: Z_BASIS_SYM, 1: X_BASIS_SYM}
QUBIT_MAP_Z = {0: "→ (0)", 1: "↑ (1)"} # Using arrows as described
QUBIT_MAP_X = {0: "↘ (0)", 1: "↗ (1)"} # Using arrows as described
QBER_THRESHOLD = 0.15
QBER_SAMPLE_FRACTION = 0.30
TARGET_BACKEND = "ibm_brisbane" # Default preferred IBM backend

# --- Initialize Session State (Essential for complex layouts/interactions) ---
if 'simulation_started' not in st.session_state:
    st.session_state.simulation_started = False
if 'simulation_run_completed' not in st.session_state:
    st.session_state.simulation_run_completed = False
if 'run_params' not in st.session_state:
    st.session_state.run_params = {}
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'include_eve' not in st.session_state: # Default for toggle
    st.session_state.include_eve = False
# Add state for IBM service and backends
if 'ibm_service' not in st.session_state:
    st.session_state.ibm_service = None
if 'available_ibm_backends' not in st.session_state:
    st.session_state.available_ibm_backends = []
if 'ibm_init_error' not in st.session_state:
    st.session_state.ibm_init_error = None


# --- IBM Service Initialization (Attempt only once per session) ---
# Moved this section higher up to ensure it runs before UI elements needing it
if QISKIT_RUNTIME_AVAILABLE and st.session_state.ibm_service is None and st.session_state.ibm_init_error is None:
    try:
        # Check if the secret exists and is not empty
        if "IBM_QUANTUM_TOKEN" in st.secrets and st.secrets["IBM_QUANTUM_TOKEN"]:
            token = st.secrets["IBM_QUANTUM_TOKEN"]
            # Attempt to initialize the service
            # Use channel='ibm_quantum', instance might be needed for specific plans
            st.session_state.ibm_service = QiskitRuntimeService(channel="ibm_quantum", token=token)
            # Fetch backends ONLY if service initialized successfully
            # Filter for operational, non-simulator quantum devices
            backends = st.session_state.ibm_service.backends(simulator=False, operational=True)
            st.session_state.available_ibm_backends = sorted([b.name for b in backends])
            st.sidebar.success("✅ IBM Quantum Service Initialized.") # Feedback in sidebar
            st.session_state.ibm_init_error = False # Mark successful init
        elif "IBM_QUANTUM_TOKEN" in st.secrets:
             st.session_state.ibm_init_error = "IBM_QUANTUM_TOKEN found in secrets but is empty."
        else:
            st.session_state.ibm_init_error = "IBM_QUANTUM_TOKEN not found in Streamlit secrets (.streamlit/secrets.toml)."

    except Exception as e:
        st.session_state.ibm_init_error = f"Failed to initialize IBM Quantum Service: {e}"
        st.session_state.ibm_service = None # Ensure service is None on error
        st.session_state.available_ibm_backends = []

# Display error message prominently if initialization failed
if st.session_state.ibm_init_error and isinstance(st.session_state.ibm_init_error, str):
    st.sidebar.warning(f"⚠️ IBM Quantum Integration Issue: {st.session_state.ibm_init_error}")

# Retrieve from session state for use later in the script
service = st.session_state.get('ibm_service', None)
available_ibm_backends = st.session_state.get('available_ibm_backends', [])
target_backend_available = TARGET_BACKEND in available_ibm_backends

# --- Custom CSS Injection ---
# (CSS remains the same as provided)
css = """
<style>
    /* General Styling */
    .stApp {
        /* background-color: #f0f2f6; /* Light background */
    }

    /* Header Area */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center; /* Vertically align items */
        padding-bottom: 1rem; /* Add some space below header */
        border-bottom: 1px solid #ddd; /* Separator line */
        margin-bottom: 1rem;
    }
    .header-container .title-subtitle {
        flex-grow: 1; /* Allow title to take available space */
    }
    .header-container .controls {
        display: flex;
        align-items: center;
        gap: 1rem; /* Space between controls */
    }

    /* Custom Button Styling */
    .stButton>button {
        border-radius: 20px; /* More rounded */
        padding: 0.5rem 1rem;
        /* Add hover effects etc. */
    }

    /* Custom Toggle Styling (May need adjustment based on Streamlit version) */
    .stToggle label {
        /* Style label if needed */
    }

    /* Basis Representation */
    .basis-section {
        border: 1px solid #eee;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        background-color: white;
    }
    .basis-section h5 { /* Target subheader within section */
        margin-bottom: 1rem;
    }
    .basis-section p {
        font-size: 1.2rem; /* Larger symbols */
        margin: 0.5rem 0;
    }

    /* Table Styling */
    .stDataFrame { /* Target the container */
       /* width: 100%; */
    }
     .stDataFrame table { /* Target the actual table */
        /* width: 100%; */
        /* border-collapse: collapse; */ /* Ensure borders connect */
    }
    .stDataFrame th {
        background-color: #f8f9fa; /* Light header background */
        border-bottom: 2px solid #dee2e6;
        text-align: left;
        padding: 8px;
    }
    .stDataFrame td {
        border: 1px solid #dee2e6;
        padding: 8px;
        text-align: left;
    }
    /* Optional: Alternating row colors */
    /* .stDataFrame tbody tr:nth-child(even) {
        background-color: #f2f2f2;
    } */

</style>
"""
st.markdown(css, unsafe_allow_html=True)


# --- Header Section ---
# Use columns for layout, but apply flexbox via CSS container for better control
st.markdown('<div class="header-container">', unsafe_allow_html=True) # Start CSS container

# Column 1: Title and Subtitle
st.markdown('<div class="title-subtitle">', unsafe_allow_html=True)
st.title("BB84 Quantum Key Distribution")
st.subheader("Quantum State Transmission")
st.markdown('</div>', unsafe_allow_html=True)

# Column 2: Controls (Toggle and Button)
st.markdown('<div class="controls">', unsafe_allow_html=True)
# Use st.session_state for toggle value persistence
eve_toggle = st.toggle("Include Eve (Intercept-Resend)", value=st.session_state.include_eve, key="eve_toggle_widget", help="Simulates Eve's intercept-resend attack by introducing errors when using the **Local Simulator**. Does not affect real hardware runs.")
st.session_state.include_eve = eve_toggle # Update state when toggle changes

# Start Simulation Button
start_simulation_pressed = st.button("Start Simulation", disabled=not SIMULATION_MODULES_AVAILABLE) # Disable if modules failed
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # End CSS container


# --- Main Content Area with Tabs ---
tab1, tab2 = st.tabs(["**Visualization**", "**Statistics**"])

with tab1:
    if not SIMULATION_MODULES_AVAILABLE:
        st.error("Cannot proceed with simulation visualization as core modules failed to load.")
    else:
        st.header("Simulation Steps & Results")

        # --- Basis Representation (Revised) ---
        st.subheader("Photon Basis Representation")
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                 st.subheader("Rectilinear Basis (Z)")
                 st.markdown(f"<p style='font-size: 1.2rem; text-align: center; margin: 0.5rem 0;'>{QUBIT_MAP_Z[0]} (Bit 0)</p>", unsafe_allow_html=True)
                 st.markdown(f"<p style='font-size: 1.2rem; text-align: center; margin: 0.5rem 0;'>{QUBIT_MAP_Z[1]} (Bit 1)</p>", unsafe_allow_html=True)
                 st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True) # Spacer

        with col2:
             with st.container(border=True):
                 st.subheader("Diagonal Basis (X)")
                 st.markdown(f"<p style='font-size: 1.2rem; text-align: center; margin: 0.5rem 0;'>{QUBIT_MAP_X[0]} (Bit 0)</p>", unsafe_allow_html=True)
                 st.markdown(f"<p style='font-size: 1.2rem; text-align: center; margin: 0.5rem 0;'>{QUBIT_MAP_X[1]} (Bit 1)</p>", unsafe_allow_html=True)
                 st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True) # Spacer

        st.markdown("---") # Separator

        # --- Simulation Execution and Display Area ---
        st.markdown("#### Configuration")
        num_bits = st.number_input("Number of Qubits:", min_value=4, max_value=100, value=20, step=4, key="num_bits_main")

        # --- Backend Selection Dropdown ---
        # Start with local simulator
        backend_options_main = ["Local Simulator (qiskit-aer)"]
        # Add IBM backends if service is initialized and backends are available
        if service and available_ibm_backends:
            # Add the target backend first if available
            if target_backend_available and TARGET_BACKEND not in backend_options_main:
                 backend_options_main.append(TARGET_BACKEND)
            # Add other available IBM backends, excluding the target if already added
            other_ibm_backends = [b for b in available_ibm_backends if b != TARGET_BACKEND]
            backend_options_main.extend(other_ibm_backends)

        # Determine default index (prefer target IBM backend if available, else local)
        default_backend_index_main = 0
        if target_backend_available:
            try:
                # Find the index of the target backend in the combined list
                default_backend_index_main = backend_options_main.index(TARGET_BACKEND)
            except ValueError:
                pass # Should not happen if target_backend_available is True, but safety first

        selected_backend_main = st.selectbox(
            "Execution Backend:",
            backend_options_main,
            index=default_backend_index_main,
            key="selected_backend_main",
            help="Select 'Local Simulator' for fast, noiseless runs or an IBM Quantum backend (requires API token in secrets) for real hardware execution."
        )

        # --- Run Simulation Logic (Triggered by Button) ---
        if start_simulation_pressed:
            st.session_state.simulation_started = True
            st.session_state.simulation_run_completed = False
            st.session_state.results = {} # Clear previous results
            st.session_state.run_params = {
                'num_bits': num_bits,
                'include_eve': st.session_state.include_eve, # Get from state
                'selected_backend': selected_backend_main,
            }

            st.markdown("---")
            st.subheader("Running Simulation...")
            progress_bar = st.progress(0, text="Starting...")
            eve_simulation_active = False # Flag to track if Eve simulation was performed

            try:
                    # Step 1: Alice
                    progress_bar.progress(5, text="Step 1/7: Alice generating bits and bases...")
                    alice_bits, alice_bases = bb84.generate_bits_and_bases(num_bits)
                    st.session_state.results['alice_bits'] = alice_bits
                    st.session_state.results['alice_bases'] = alice_bases

                    # Step 2: Bob Bases
                    progress_bar.progress(10, text="Step 2/7: Bob choosing bases...")
                    bob_bases = bb84.generate_bases(num_bits)
                    st.session_state.results['bob_bases'] = bob_bases

                    # Step 3: Quantum Run
                    progress_bar.progress(20, text="Step 3/7: Preparing quantum circuits...")
                    qkd_circs = qkd_circuits.create_bb84_circuits(alice_bits, alice_bases, bob_bases)

                    measured_bits_list = None
                    run_start_time = time.time()
                    is_local_simulator = selected_backend_main == "Local Simulator (qiskit-aer)"
                    spinner_text = f"Step 3/7: Running {len(qkd_circs)} circuits on {selected_backend_main}..."
                    if is_local_simulator and st.session_state.include_eve:
                        spinner_text += " (Simulating Eve's Interception)"

                    with st.spinner(spinner_text):
                        if is_local_simulator:
                            measured_bits_list = qkd_runner.run_circuits_local_simulator(qkd_circs, shots=1)

                            # *** EVE INTERCEPTION SIMULATION (Local Simulator Only) ***
                            if st.session_state.include_eve and measured_bits_list is not None:
                                eve_simulation_active = True
                                progress_bar.progress(50, text="Step 3/7: Simulating Eve's measurements...")
                                # Eve randomly chooses bases
                                eve_bases = bb84.generate_bases(num_bits)
                                st.session_state.results['eve_bases'] = eve_bases # Store Eve's bases

                                bob_measured_bits_potentially_altered = np.array(measured_bits_list, dtype=np.uint8)

                                bits_flipped_by_eve = 0
                                for i in range(num_bits):
                                    # If Eve's basis differs from Alice's preparation basis...
                                    if alice_bases[i] != eve_bases[i]:
                                        # ...there's a 50% chance the state Bob receives is effectively
                                        # orthogonal to Alice's original state in Alice's basis,
                                        # leading to a 50% chance Bob measures the wrong bit
                                        # *if* Bob uses Alice's original basis.
                                        # We simulate this potential error directly on Bob's measurement result.
                                        if random.random() < 0.5:
                                            # Flip Bob's measured bit for this position
                                            original_bit = bob_measured_bits_potentially_altered[i]
                                            bob_measured_bits_potentially_altered[i] = 1 - original_bit
                                            bits_flipped_by_eve += 1

                                # Use the potentially altered bits as Bob's results
                                measured_bits_list = bob_measured_bits_potentially_altered.tolist()
                                st.info(f"🕵️ Eve simulation active: Intercepted {num_bits} qubits. Introduced approx. {bits_flipped_by_eve} errors due to basis mismatches.")
                            # *** END EVE SIMULATION ***

                        elif service and selected_backend_main in available_ibm_backends:
                            # Ensure the service object is valid before using
                            if st.session_state.ibm_service:
                                measured_bits_list = qkd_runner.run_circuits_ibm_runtime(st.session_state.ibm_service, selected_backend_main, qkd_circs, shots=1)
                                if st.session_state.include_eve:
                                    st.warning(f"⚠️ 'Include Eve' is checked, but backend is '{selected_backend_main}'. Active Eve interception is only simulated on the 'Local Simulator'. Results reflect hardware noise, not simulated eavesdropping.")
                            else:
                                st.error("IBM Service object is not available. Cannot run on IBM backend.")
                        else:
                            st.error(f"Selected backend '{selected_backend_main}' is not configured or available.")

                    run_duration = time.time() - run_start_time
                    progress_text = f"Step 3/7: Quantum run finished ({run_duration:.2f}s)."
                    if eve_simulation_active:
                        progress_text += " Eve simulation applied."
                    progress_bar.progress(60, text=progress_text)


                    if measured_bits_list is not None:
                        # Use the final list (potentially altered by Eve sim)
                        bob_measured_bits_np = np.array(measured_bits_list, dtype=np.uint8)
                        st.session_state.results['bob_measured_bits_np'] = bob_measured_bits_np
                    else:
                        raise ValueError("Quantum simulation or execution failed to return results.")

                    # --- Steps 4-7: Classical Processing (using potentially Eve-altered Bob bits) ---
                    progress_bar.progress(70, text="Step 4/7: Comparing bases (Alice/Bob)...")
                    matching_indices, _ = bb84.compare_bases(alice_bases, bob_bases)
                    st.session_state.results['matching_indices'] = matching_indices

                    progress_bar.progress(80, text="Step 5/7: Sifting keys...")
                    # Sifting uses Alice's original bits and Bob's (potentially altered) measured bits
                    alice_sifted, bob_sifted = qkd_processing.sift_key(alice_bits, bob_measured_bits_np, matching_indices)
                    st.session_state.results['alice_sifted'] = alice_sifted
                    st.session_state.results['bob_sifted'] = bob_sifted # This now contains Eve's potential errors

                    progress_bar.progress(90, text="Step 6/7: Estimating QBER...")
                    if len(alice_sifted) > 1: # Need at least 2 bits to estimate QBER
                        # QBER calculation will now compare Alice's sifted key with Bob's potentially error-containing sifted key
                        qber, qber_idx, remain_idx = qkd_processing.estimate_qber(alice_sifted, bob_sifted, QBER_SAMPLE_FRACTION)
                        st.session_state.results['qber'] = qber
                        st.session_state.results['qber_indices_in_sifted'] = qber_idx
                        st.session_state.results['remaining_indices_in_sifted'] = remain_idx
                        st.info(f"Step 6: QBER Estimated: {qber:.2%}") # Show QBER info message
                    else:
                        st.warning("Not enough sifted bits to estimate QBER. Skipping QBER estimation and final key extraction.")
                        qber = None
                        remain_idx = []
                        st.session_state.results['qber'] = None
                        st.session_state.results['qber_indices_in_sifted'] = []
                        st.session_state.results['remaining_indices_in_sifted'] = []


                    if qber is not None and qber > QBER_THRESHOLD:
                        progress_bar.progress(100, text="Finished - Protocol Aborted (High QBER).")
                        st.error(f"QBER ({qber:.2%}) exceeded threshold ({QBER_THRESHOLD:.2%}). Protocol Aborted.")
                        st.session_state.results['protocol_aborted'] = True
                        st.session_state.results['final_key_str'] = ""
                        st.session_state.results['final_key_len'] = 0
                    elif qber is not None:
                        progress_bar.progress(95, text="Step 7/7: Extracting final key...")
                        # Final key uses Alice's bits (assuming no error correction implemented)
                        final_key = qkd_processing.extract_final_key(alice_sifted, remain_idx)
                        st.session_state.results['final_key_str'] = helpers.bits_to_string(final_key)
                        st.session_state.results['final_key_len'] = len(final_key)
                        st.session_state.results['protocol_aborted'] = False
                        progress_bar.progress(100, text="Finished - Final Key Extracted.")
                    else: # Case where QBER wasn't calculated
                        st.session_state.results['protocol_aborted'] = False
                        st.session_state.results['final_key_str'] = ""
                        st.session_state.results['final_key_len'] = 0
                        progress_bar.progress(100, text="Finished - Insufficient bits for QBER/Key.")


                    st.session_state.simulation_run_completed = True
                    st.success(f"Simulation finished successfully on {selected_backend_main}.")
                    if eve_simulation_active:
                        st.success("🕵️ Eve's intercept-resend attack was simulated.")


            except Exception as e:
                    st.error(f"Simulation Error: {e}")
                    st.error(traceback.format_exc()) # Print detailed traceback
                    st.session_state.simulation_run_completed = False
                    progress_bar.progress(100, text="Finished - Error Occurred.")


        # --- Display Results Table (Conditional) ---
        st.markdown("---")
        st.subheader("Results Table")
        if st.session_state.simulation_run_completed:
            res = st.session_state.results
            params = st.session_state.run_params
            num_display_bits = params['num_bits'] # Length of original run

            # --- Construct DataFrame ---
            table_data = {
                'Bit #': list(range(num_display_bits)),
                'Alice Photon': [""] * num_display_bits,
                'Alice Basis': [""] * num_display_bits,
                'Bob Basis': [""] * num_display_bits,
                'Bob Measurement': [""] * num_display_bits,
                'Bases Match': [""] * num_display_bits,
                'Sifted Bit (Alice)': ["-"] * num_display_bits,
                'Sifted Bit (Bob)': ["-"] * num_display_bits,
                'Used for QBER': ["-"] * num_display_bits,
                'Final Key Bit': ["-"] * num_display_bits
            }

            # Safe access to results with defaults
            alice_bits = res.get('alice_bits', [])
            alice_bases = res.get('alice_bases', [])
            bob_bases = res.get('bob_bases', [])
            bob_measured_bits = res.get('bob_measured_bits_np', np.array([]))
            matching_indices = res.get('matching_indices', [])
            alice_sifted = res.get('alice_sifted', [])
            bob_sifted = res.get('bob_sifted', [])
            qber_indices_sifted = res.get('qber_indices_in_sifted', []) # Indices within the *sifted* key
            remaining_indices_sifted = res.get('remaining_indices_in_sifted', []) # Indices within the *sifted* key
            protocol_aborted = res.get('protocol_aborted', False)

            sifted_idx_map = {original_idx: sifted_pos for sifted_pos, original_idx in enumerate(matching_indices)}

            for i in range(num_display_bits):
                # Alice Info
                if i < len(alice_bits) and i < len(alice_bases):
                    bit = alice_bits[i]
                    basis = alice_bases[i]
                    table_data['Alice Basis'][i] = BASIS_MAP.get(basis, '?')
                    if basis == 0: table_data['Alice Photon'][i] = QUBIT_MAP_Z.get(bit, '?')
                    else: table_data['Alice Photon'][i] = QUBIT_MAP_X.get(bit, '?')
                else:
                    table_data['Alice Photon'][i] = 'ERR'
                    table_data['Alice Basis'][i] = 'ERR'

                # Bob Info
                if i < len(bob_bases):
                    table_data['Bob Basis'][i] = BASIS_MAP.get(bob_bases[i], '?')
                else: table_data['Bob Basis'][i] = 'ERR'

                if i < len(bob_measured_bits):
                    table_data['Bob Measurement'][i] = str(bob_measured_bits[i])
                else: table_data['Bob Measurement'][i] = 'ERR'

                # Matching and Sifted/Key Info
                is_match = i in matching_indices
                table_data['Bases Match'][i] = '✅' if is_match else '❌'

                if is_match:
                    sifted_pos = sifted_idx_map.get(i)
                    if sifted_pos is not None and sifted_pos < len(alice_sifted):
                        table_data['Sifted Bit (Alice)'][i] = str(alice_sifted[sifted_pos])
                    else: table_data['Sifted Bit (Alice)'][i] = 'ERR'

                    if sifted_pos is not None and sifted_pos < len(bob_sifted):
                         table_data['Sifted Bit (Bob)'][i] = str(bob_sifted[sifted_pos])
                    else: table_data['Sifted Bit (Bob)'][i] = 'ERR'


                    if res['qber'] is not None: # Only if QBER was calculated
                        is_qber_bit = sifted_pos in qber_indices_sifted
                        is_final_key_bit = sifted_pos in remaining_indices_sifted

                        table_data['Used for QBER'][i] = 'Yes' if is_qber_bit else 'No'

                        if not protocol_aborted and is_final_key_bit:
                             # Find the position within the final key
                             try:
                                 final_key_pos = remaining_indices_sifted.index(sifted_pos)
                                 if final_key_pos < len(res.get('final_key_str', '')):
                                      table_data['Final Key Bit'][i] = res['final_key_str'][final_key_pos]
                                 else: table_data['Final Key Bit'][i] = 'ERR'
                             except ValueError:
                                 table_data['Final Key Bit'][i] = 'ERR' # Should not happen if logic is correct
                        elif protocol_aborted:
                             table_data['Final Key Bit'][i] = '(Aborted)'
                        elif is_qber_bit:
                             table_data['Final Key Bit'][i] = '(QBER Sample)'
                        else:
                             table_data['Final Key Bit'][i] = '-' # Not used for QBER, not final key
                    else: # QBER not calculated
                         table_data['Used for QBER'][i] = 'N/A'
                         table_data['Final Key Bit'][i] = '(No QBER)'


            results_df = pd.DataFrame(table_data)
            # Display the dataframe, allowing horizontal scrolling if needed
            st.dataframe(results_df, use_container_width=False) # Set to False to enable scrolling if wide

        elif st.session_state.simulation_started:
            st.warning("Simulation started but did not complete successfully. Check logs above.")
        else:
            st.info("Click 'Start Simulation' to configure and run the BB84 protocol.")

with tab2:
    st.header("Simulation Statistics")
    if not SIMULATION_MODULES_AVAILABLE:
        st.error("Cannot display statistics as core simulation modules failed to load.")
    elif st.session_state.simulation_run_completed:
        res = st.session_state.results
        params = st.session_state.run_params

        st.markdown(f"**Run Parameters:**")
        # Use st.expander for potentially long JSON
        with st.expander("View Parameters"):
            st.json(params)

        st.markdown("---")
        st.subheader("Key Lengths & Efficiency")
        initial_bits = params.get('num_bits', 0)
        sifted_len = len(res.get('alice_sifted', []))
        final_key_len = res.get('final_key_len', 0) if not res.get('protocol_aborted', True) else 0
        qber_sample_len = len(res.get('qber_indices_in_sifted', []))

        col1, col2, col3 = st.columns(3)
        col1.metric("Initial Qubits", initial_bits)
        col2.metric("Sifted Key Length", sifted_len)
        col3.metric("Final Key Length", final_key_len)

        st.metric("QBER Sample Size", qber_sample_len)

        efficiency = (final_key_len / initial_bits) * 100 if initial_bits > 0 else 0
        st.metric("Overall Key Efficiency", f"{efficiency:.2f}%")
        st.progress(efficiency / 100)

        st.markdown("---")
        st.subheader("Error Rate Analysis")
        if 'qber' in res and res['qber'] is not None:
            qber_val = res['qber']
            st.metric(label="Estimated QBER", value=f"{qber_val:.2%}")

            is_simulator = "simulator" in params.get('selected_backend', '').lower()
            eve_included = params.get('include_eve', False)

            if qber_val == 0.0:
                if is_simulator and not eve_included:
                    st.info("💡 QBER is 0%. Expected for a **noiseless simulator** run without Eve.")
                elif is_simulator and eve_included:
                    # This case should be less likely now with active simulation
                    st.warning("💡 QBER is 0% despite simulating Eve's interception. This could happen by chance if Eve guessed all bases correctly, or if the QBER sample happened to miss all errors. Try increasing the number of qubits or the QBER sample fraction.")
                elif not is_simulator:
                    st.info("💡 QBER is 0%. Indicates very low noise on the quantum hardware for this run/sample, or the sample chosen for QBER estimation happened to have no errors.")
            elif qber_val > 0.0:
                 if is_simulator and eve_included:
                     st.success(f"💡 QBER is {qber_val:.2%}. This is the **expected outcome** when simulating Eve's intercept-resend attack on the local simulator, as her measurements introduce errors.")
                 elif is_simulator and not eve_included:
                     st.warning(f"💡 QBER is {qber_val:.2%}. This is **unexpected** on the ideal local simulator *without* Eve. Check if the simulation code (e.g., `circuits.py`, `runner.py`) intentionally introduces noise or errors.")
                 elif not is_simulator:
                     st.info(f"💡 QBER is {qber_val:.2%}. On real hardware, this reflects noise (environmental, gate/measurement errors) and potentially eavesdropping (indistinguishable from noise here).")
            # --- End Enhanced Explanation ---


            if res.get('protocol_aborted', False):
                st.error(f"Protocol Aborted: QBER ({qber_val:.2%}) likely exceeded threshold ({QBER_THRESHOLD:.2%}). No final key generated.")
            else:
                st.success(f"Protocol Successful: QBER ({qber_val:.2%}) is within the acceptable threshold ({QBER_THRESHOLD:.2%}).")
        else:
            st.info("QBER was not calculated (e.g., insufficient sifted bits).")

        # Optionally display the final key if generated
        if not res.get('protocol_aborted', True) and res.get('final_key_str'):
            st.markdown("---")
            st.subheader("Generated Final Key")
            # Use an expander for potentially long keys
            with st.expander("Show Final Key String"):
                st.text(res.get('final_key_str', ''))
        elif res.get('protocol_aborted', False):
             st.warning("Final key was not generated because the protocol was aborted.")
        else:
             st.info("Final key was not generated (likely due to insufficient bits after sifting/QBER).")


    else:
        st.info("Run a simulation on the 'Visualization' tab to see statistics.")

# Add a footer or credits if desired
st.markdown("---")
st.caption("BB84 Simulator using Streamlit and Qiskit")