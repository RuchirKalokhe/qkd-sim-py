from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qkd-simulation",
    version="1.0.0",
    author="QKD Simulation Team",
    author_email="example@example.com",
    description="Quantum Key Distribution (QKD) Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/qkd_sim_py",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Security :: Cryptography",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "qiskit>=1.0.0",
        "qiskit_aer>=0.12.0",
        "qiskit-ibm-runtime>=0.18.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "typing-extensions>=4.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qkd-simulator=qkd_simulation.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "qkd_simulation": ["images/*"],
    },
)
