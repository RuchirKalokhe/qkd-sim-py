#!/usr/bin/env python
"""
Command-line interface for QKD Simulation.

This module provides a command-line interface to run QKD simulations
without the Streamlit web interface.
"""

import argparse
import sys
import numpy as np
import logging
import json
from pathlib import Path

from qkd_simulation.protocols import bb84
from qkd_simulation.quantum import circuits
from qkd_simulation.quantum import optimized_runner as runner
from qkd_simulation.classical import optimized_processing as processing
from qkd_simulation.utils import helpers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='QKD Simulation CLI')
    
    # Main command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # BB84 protocol command
    bb84_parser = subparsers.add_parser('bb84', help='Run BB84 protocol simulation')
    bb84_parser.add_argument('--num-bits', type=int, default=20, 
                           help='Number of qubits to simulate')
    bb84_parser.add_argument('--include-eve', action='store_true',
                           help='Include Eve (eavesdropper) in the simulation')
    bb84_parser.add_argument('--qber-sample', type=float, default=0.3,
                           help='Fraction of sifted key to use for QBER estimation')
    bb84_parser.add_argument('--error-threshold', type=float, default=0.15,
                           help='Maximum acceptable error rate')
    bb84_parser.add_argument('--security-parameter', type=float, default=0.8,
                           help='Security parameter for privacy amplification')
    bb84_parser.add_argument('--output', type=str, default='results.json',
                           help='Output file for results (JSON format)')
    bb84_parser.add_argument('--parallel', action='store_true',
                           help='Use parallel processing for better performance')
    bb84_parser.add_argument('--batch-size', type=int, default=20,
                           help='Batch size for circuit execution')
    
    # E91 protocol command
    e91_parser = subparsers.add_parser('e91', help='Run E91 protocol simulation')
    e91_parser.add_argument('--num-pairs', type=int, default=20,
                          help='Number of entangled pairs to simulate')
    e91_parser.add_argument('--output', type=str, default='results.json',
                          help='Output file for results (JSON format)')
    
    # Visualization command
    viz_parser = subparsers.add_parser('visualize', help='Visualize quantum circuits')
    viz_parser.add_argument('--protocol', type=str, choices=['bb84', 'e91'], default='bb84',
                          help='Protocol to visualize')
    viz_parser.add_argument('--num-circuits', type=int, default=1,
                          help='Number of circuits to visualize')
    viz_parser.add_argument('--output-dir', type=str, default='circuits',
                          help='Directory to save circuit visualizations')
    
    return parser.parse_args()

def run_bb84_simulation(args):
    """Run BB84 protocol simulation."""
    logger.info(f"Starting BB84 simulation with {args.num_bits} qubits")
    
    # Generate random bits and bases for Alice
    alice_bits, alice_bases = bb84.generate_bits_and_bases(args.num_bits)
    logger.info("Generated Alice's bits and bases")
    
    # Generate random bases for Bob
    bob_bases = bb84.generate_bases(args.num_bits)
    logger.info("Generated Bob's bases")
    
    # Generate random bases for Eve if included
    eve_bases = None
    if args.include_eve:
        eve_bases = bb84.generate_bases(args.num_bits)
        logger.info("Generated Eve's bases")
    
    # Create quantum circuits
    if args.include_eve and eve_bases is not None:
        logger.info("Creating BB84 circuits with Eve's interception")
        qkd_circs = circuits.create_bb84_circuits_with_eve(
            alice_bits, alice_bases, eve_bases, bob_bases
        )
    else:
        logger.info("Creating BB84 circuits")
        qkd_circs = circuits.create_bb84_circuits(
            alice_bits, alice_bases, bob_bases
        )
    
    # Run circuits on local simulator
    logger.info(f"Running {len(qkd_circs)} circuits on local simulator")
    if args.include_eve and eve_bases is not None:
        eve_bits, bob_measured_bits = runner.run_circuits_with_eve_local_simulator(qkd_circs)
        bob_measured_bits_np = np.array(bob_measured_bits, dtype=np.uint8)
    else:
        if args.parallel:
            logger.info("Using parallel processing")
            measured_bits = runner.run_circuits_local_simulator_optimized(
                qkd_circs, use_parallel=True, max_circuits_in_memory=args.batch_size
            )
        else:
            measured_bits = runner.run_circuits_local_simulator_optimized(qkd_circs)
        
        bob_measured_bits_np = np.array(measured_bits, dtype=np.uint8)
    
    # Process results
    logger.info("Processing results")
    results = processing.batch_process_keys(
        alice_bits,
        bob_measured_bits_np,
        alice_bases,
        bob_bases,
        qber_sample_fraction=args.qber_sample,
        error_threshold=args.error_threshold,
        security_parameter=args.security_parameter
    )
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {
        'alice_bits': alice_bits.tolist(),
        'alice_bases': alice_bases.tolist(),
        'bob_bases': bob_bases.tolist(),
        'bob_measured_bits': bob_measured_bits_np.tolist(),
        'matching_indices': results['matching_indices'],
        'alice_sifted': results['alice_sifted'].tolist() if 'alice_sifted' in results else [],
        'bob_sifted': results['bob_sifted'].tolist() if 'bob_sifted' in results else [],
        'qber': float(results['qber']) if 'qber' in results else None,
        'protocol_aborted': results.get('protocol_aborted', False),
        'final_key': results['final_key'].tolist() if 'final_key' in results and isinstance(results['final_key'], np.ndarray) else []
    }
    
    # Add Eve's information if included
    if args.include_eve and eve_bases is not None:
        serializable_results['eve_bases'] = eve_bases.tolist()
        serializable_results['eve_bits'] = eve_bits.tolist() if isinstance(eve_bits, np.ndarray) else eve_bits
    
    # Save results to file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary to console
    print("\nBB84 Simulation Summary:")
    print(f"Number of qubits: {args.num_bits}")
    print(f"Eve included: {args.include_eve}")
    print(f"Matching bases: {len(results['matching_indices'])}/{args.num_bits} ({len(results['matching_indices'])/args.num_bits:.2%})")
    print(f"QBER: {serializable_results['qber']:.4f}")
    print(f"Protocol {'aborted' if serializable_results['protocol_aborted'] else 'successful'}")
    print(f"Final key length: {len(serializable_results['final_key'])}")
    print(f"Key efficiency: {len(serializable_results['final_key'])/args.num_bits:.2%}")
    print(f"Detailed results saved to: {output_path}")

def run_e91_simulation(args):
    """Run E91 protocol simulation."""
    logger.info(f"Starting E91 simulation with {args.num_pairs} entangled pairs")
    
    # Create E91 circuits
    e91_circs, bases = circuits.create_e91_circuits(args.num_pairs)
    logger.info(f"Created {len(e91_circs)} E91 circuits")
    
    # Run circuits on local simulator
    measured_bits = runner.run_circuits_local_simulator_optimized(e91_circs)
    logger.info("Completed quantum simulation")
    
    # Process results
    alice_results = []
    bob_results = []
    
    for i, result in enumerate(measured_bits):
        if result >= 0:
            # Parse the result string (format: "alice_bit bob_bit")
            result_str = str(result)
            if len(result_str) >= 2:
                alice_bit = int(result_str[0])
                bob_bit = int(result_str[1])
                alice_results.append(alice_bit)
                bob_results.append(bob_bit)
    
    alice_results_np = np.array(alice_results, dtype=np.uint8)
    bob_results_np = np.array(bob_results, dtype=np.uint8)
    
    # Calculate correlations for Bell test
    correlations = []
    for i in range(len(alice_results)):
        alice_basis = bases[i][0]
        bob_basis = bases[i][1]
        alice_result = alice_results[i]
        bob_result = bob_results[i]
        
        # Calculate correlation value (+1 for same results, -1 for different)
        correlation = 1 if alice_result == bob_result else -1
        correlations.append((alice_basis, bob_basis, correlation))
    
    # Prepare results for saving
    serializable_results = {
        'num_pairs': args.num_pairs,
        'alice_bases': bases[:, 0].tolist(),
        'bob_bases': bases[:, 1].tolist(),
        'alice_results': alice_results,
        'bob_results': bob_results,
        'correlations': correlations
    }
    
    # Save results to file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary to console
    print("\nE91 Simulation Summary:")
    print(f"Number of entangled pairs: {args.num_pairs}")
    print(f"Successful measurements: {len(alice_results)}/{args.num_pairs} ({len(alice_results)/args.num_pairs:.2%})")
    print(f"Detailed results saved to: {output_path}")

def visualize_circuits(args):
    """Visualize quantum circuits."""
    logger.info(f"Visualizing {args.num_circuits} {args.protocol} circuits")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.protocol == 'bb84':
        # Generate random bits and bases
        alice_bits, alice_bases = bb84.generate_bits_and_bases(args.num_circuits)
        bob_bases = bb84.generate_bases(args.num_circuits)
        
        # Create BB84 circuits
        qkd_circs = circuits.create_bb84_circuits(alice_bits, alice_bases, bob_bases)
        
        # Visualize circuits
        for i, circ in enumerate(qkd_circs):
            filename = output_dir / f"bb84_circuit_{i}.png"
            circuits.visualize_circuit(circ, str(filename))
            logger.info(f"Saved circuit visualization to {filename}")
    
    elif args.protocol == 'e91':
        # Create E91 circuits
        e91_circs, bases = circuits.create_e91_circuits(args.num_circuits)
        
        # Visualize circuits
        for i, circ in enumerate(e91_circs):
            filename = output_dir / f"e91_circuit_{i}.png"
            circuits.visualize_circuit(circ, str(filename))
            logger.info(f"Saved circuit visualization to {filename}")
    
    print(f"\nCircuit visualizations saved to: {output_dir}")

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    if args.command == 'bb84':
        run_bb84_simulation(args)
    elif args.command == 'e91':
        run_e91_simulation(args)
    elif args.command == 'visualize':
        visualize_circuits(args)
    else:
        print("Please specify a command. Use --help for more information.")
        sys.exit(1)

if __name__ == "__main__":
    main()
