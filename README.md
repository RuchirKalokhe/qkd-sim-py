# QKD Streamlit Simulation App

This application simulates Quantum Key Distribution using Qiskit and Streamlit.

## Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Configure secrets for local development in `.streamlit/secrets.toml` (DO NOT COMMIT THIS FILE).
6. For deployment (e.g., Streamlit Community Cloud), configure secrets using their secrets management.

## Running Locally
`streamlit run app.py`
