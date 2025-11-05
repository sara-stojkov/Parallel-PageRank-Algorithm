# Parallel PageRank Algorithm

This project implements a parallel version of the PageRank algorithm, originally developed by Larry Page, designed to evaluate the relative importance of nodes (e.g., web pages) in a directed graph. The project explores parallel programming techniques to achieve better scalability and efficiency on modern multi-core processors.

 ## Key Features
- Graph Representation: Implemented with adjacency lists for memory efficiency.
- Sequential & Parallel Execution: Baseline sequential implementation + multiple parallel versions.
- Random Graph Generation: Erdős–Rényi model for controlled testing.
- Performance Evaluation: Comparison of sequential vs. parallel execution with speedup measurements.

Optimizations:
- Block-Optimized – improves cache locality.
- Vectorized – uses loop unrolling and SIMD-like optimizations.
- NUMA-Aware – optimized for non-uniform memory access architectures.
- Adaptive Grain Size – balances workload dynamically across threads.
  
## Technologies Used

- C++ – Core implementation.
- Intel Threading Building Blocks (TBB) – Task-based parallelism (parallel_for, parallel_reduce).
- Erdős–Rényi Model – Random graph generation.
- CSV Export – Benchmark results for performance analysis.

## Algorithm Overview

PageRank is an iterative algorithm defined as:

PR(A) = (1 - d) + d * Σ [ PR(T_i) / C(T_i) ]

Iterations continue until convergence (within epsilon threshold).

The parallel implementation leverages independence between node computations per iteration, making it highly suitable for task parallelism.

## Results

Parallel implementation significantly outperforms sequential execution on large graphs.

Performance depends on:
- Graph size and density.
- Number of CPU cores/threads.
- Synchronization overhead.
- Optimizations like block and adaptive grain balancing improve speedup in real-world scenarios.

## Project Structure

- Graph – Graph representation and random generation.
- PageRank – Core sequential & parallel algorithm implementations.
- PageRankBenchmark – Test suite for evaluating performance and exporting results.

## Future Work
- Explore lock-free data structures to reduce synchronization overhead.
- Extend to distributed systems (MPI, MapReduce).
- Implement weighted and personalized PageRank variants.
