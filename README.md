# Quantum Key Distribution (QKD) Simulator

An interactive web application that simulates the BB84 Quantum Key Distribution protocol using Qiskit and Streamlit. This tool allows users to visualize and understand the fundamental principles of quantum cryptography.

![QKD Simulation](images/qkd-screenshot.png) <!-- Replace with an actual screenshot of the application -->

## Overview

This simulator demonstrates the BB84 protocol, the first quantum cryptographic protocol developed by Bennett and Brassard in 1984. It allows users to:

- Generate and exchange quantum keys between two parties (Alice and Bob)
- Optionally simulate an eavesdropper (Eve) using intercept-resend attacks
- Run simulations on a local Qiskit simulator or real IBM Quantum hardware
- Visualize each step of the protocol with interactive tables and statistics
- Calculate and analyze Quantum Bit Error Rates (QBER)

## Features

- **Protocol Simulation**: Complete BB84 protocol implementation
- **Interactive UI**: User-friendly Streamlit interface with tabbed visualization and statistics views
- **Quantum Backend Options**: 
  - Local simulator using Qiskit Aer
  - IBM Quantum real hardware integration
- **Eavesdropping Simulation**: Optional Eve intercept-resend attack modeling
- **Performance Metrics**: QBER calculation and protocol efficiency statistics
- **Educational Value**: Step-by-step visualization of quantum states and measurements


## Setup

1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Configure secrets for local development in `.streamlit/secrets.toml` (DO NOT COMMIT THIS FILE).

## Running Locally

`streamlit run app.py`

## Deployment

To deploy the application, follow these steps:

1. Ensure all dependencies are installed and the virtual environment is activated.
2. Configure the necessary secrets in `.streamlit/secrets.toml`.
3. Use Streamlit sharing or any other hosting service that supports Streamlit applications.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

