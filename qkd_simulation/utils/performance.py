"""
Performance optimization module for QKD simulation.

This module provides functions to optimize the performance of QKD simulations,
particularly for large numbers of qubits or complex protocols.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from functools import lru_cache
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from qiskit import QuantumCircuit, transpile

def parallel_transpile(circuits: List[QuantumCircuit], 
                      backend: Any,
                      optimization_level: int = 1,
                      max_workers: Optional[int] = None) -> List[QuantumCircuit]:
    """
    Transpile quantum circuits in parallel to improve performance.
    
    Args:
        circuits: List of quantum circuits to transpile
        backend: Backend to transpile for
        optimization_level: Optimization level (0-3)
        max_workers: Maximum number of worker processes (defaults to CPU count)
        
    Returns:
        List of transpiled quantum circuits
    """
    if not circuits:
        return []
    
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    # Split circuits into chunks for parallel processing
    chunk_size = max(1, len(circuits) // max_workers)
    chunks = [circuits[i:i+chunk_size] for i in range(0, len(circuits), chunk_size)]
    
    # Define transpilation function for each chunk
    def transpile_chunk(chunk):
        return transpile(chunk, backend=backend, optimization_level=optimization_level)
    
    # Process chunks in parallel
    transpiled_chunks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        transpiled_chunks = list(executor.map(transpile_chunk, chunks))
    
    # Flatten the list of transpiled circuits
    transpiled_circuits = []
    for chunk in transpiled_chunks:
        transpiled_circuits.extend(chunk)
    
    return transpiled_circuits

def parallel_simulation(circuits: List[QuantumCircuit],
                       simulator,
                       shots: int = 1,
                       max_workers: Optional[int] = None) -> List[Dict[str, int]]:
    """
    Run quantum circuit simulations in parallel.
    
    Args:
        circuits: List of quantum circuits to simulate
        simulator: Quantum simulator instance
        shots: Number of shots per circuit
        max_workers: Maximum number of worker processes (defaults to CPU count)
        
    Returns:
        List of simulation results (counts dictionaries)
    """
    if not circuits:
        return []
    
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    # Split circuits into chunks for parallel processing
    chunk_size = max(1, len(circuits) // max_workers)
    chunks = [circuits[i:i+chunk_size] for i in range(0, len(circuits), chunk_size)]
    
    # Define simulation function for each chunk
    def simulate_chunk(chunk):
        job = simulator.run(chunk, shots=shots)
        result = job.result()
        return [result.get_counts(i) for i in range(len(chunk))]
    
    # Process chunks in parallel
    results_chunks = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_chunks = list(executor.map(simulate_chunk, chunks))
    
    # Flatten the list of results
    all_results = []
    for chunk in results_chunks:
        all_results.extend(chunk)
    
    return all_results

@lru_cache(maxsize=128)
def cached_basis_comparison(alice_bases_tuple: Tuple[int, ...], 
                           bob_bases_tuple: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Compare bases with caching for repeated operations.
    
    Args:
        alice_bases_tuple: Tuple of Alice's bases
        bob_bases_tuple: Tuple of Bob's bases
        
    Returns:
        Tuple of (matching_indices, mismatching_indices)
    """
    alice_bases = np.array(alice_bases_tuple)
    bob_bases = np.array(bob_bases_tuple)
    
    matches = (alice_bases == bob_bases)
    matching_indices = tuple(np.where(matches)[0].tolist())
    mismatching_indices = tuple(np.where(~matches)[0].tolist())
    
    return matching_indices, mismatching_indices

def optimize_circuit_creation(create_circuit_fn: Callable, 
                             num_circuits: int,
                             batch_size: int = 100,
                             **kwargs) -> List[QuantumCircuit]:
    """
    Optimize circuit creation by batching and parallel processing.
    
    Args:
        create_circuit_fn: Function that creates a single circuit
        num_circuits: Total number of circuits to create
        batch_size: Number of circuits per batch
        **kwargs: Additional arguments to pass to create_circuit_fn
        
    Returns:
        List of created quantum circuits
    """
    all_circuits = []
    
    # Process in batches to avoid memory issues
    for i in range(0, num_circuits, batch_size):
        end_idx = min(i + batch_size, num_circuits)
        batch_size_actual = end_idx - i
        
        # Create batch of circuits
        batch_circuits = []
        for j in range(batch_size_actual):
            circuit_idx = i + j
            circuit = create_circuit_fn(circuit_idx=circuit_idx, **kwargs)
            batch_circuits.append(circuit)
        
        all_circuits.extend(batch_circuits)
    
    return all_circuits

def vectorized_bit_operations(bits1: np.ndarray, bits2: np.ndarray) -> Tuple[int, int, float]:
    """
    Perform vectorized bit operations for improved performance.
    
    Args:
        bits1: First bit array
        bits2: Second bit array
        
    Returns:
        Tuple of (matching_bits, total_bits, match_percentage)
    """
    if len(bits1) != len(bits2):
        raise ValueError("Bit arrays must have the same length")
    
    # Vectorized comparison
    matches = (bits1 == bits2)
    matching_bits = np.sum(matches)
    total_bits = len(bits1)
    match_percentage = float(matching_bits) / total_bits if total_bits > 0 else 1.0
    
    return matching_bits, total_bits, match_percentage

def profile_execution_time(func: Callable) -> Callable:
    """
    Decorator to profile execution time of functions.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function with timing
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

def memory_efficient_simulation(circuits: List[QuantumCircuit],
                               simulator,
                               max_circuits_in_memory: int = 50) -> List[Dict[str, int]]:
    """
    Run simulations in a memory-efficient way by processing circuits in small batches.
    
    Args:
        circuits: List of quantum circuits to simulate
        simulator: Quantum simulator instance
        max_circuits_in_memory: Maximum number of circuits to process at once
        
    Returns:
        List of simulation results (counts dictionaries)
    """
    all_results = []
    
    # Process in small batches to limit memory usage
    for i in range(0, len(circuits), max_circuits_in_memory):
        batch = circuits[i:i+max_circuits_in_memory]
        job = simulator.run(batch, shots=1)
        result = job.result()
        
        batch_results = []
        for j in range(len(batch)):
            counts = result.get_counts(j)
            batch_results.append(counts)
        
        all_results.extend(batch_results)
        
        # Force garbage collection to free memory
        batch = None
        job = None
        result = None
        import gc
        gc.collect()
    
    return all_results
