#!/usr/bin/env python3
"""
BPSO (Binary Particle Swarm Optimization) Layer Partitioning Control

This script demonstrates how to:
1. Load per-layer profiling data
2. Run BPSO optimization to find optimal binary partition
3. Generate partition configuration for the SA sim delegate
4. Control which layers get delegated to CPU (0) or SA Accelerator (1)
"""

import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Tuple

class BPSOPartitionOptimizer:
    def __init__(self, profiling_data_path: str):
        """Initialize BPSO optimizer with profiling data"""
        self.profiling_data = pd.read_csv(profiling_data_path)
        self.num_layers = len(self.profiling_data)
        
        # Identify which layers are actually delegatable to SA accelerator
        # SA accelerator supports convolution and fully connected operations (GEMM-based)
        self.delegatable_layers = []
        self.layer_characteristics = {}
        
        for i, row in self.profiling_data.iterrows():
            layer_type = row.get('layer_type', 'UNKNOWN')
            is_delegatable = layer_type in ['CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED']
            self.delegatable_layers.append(is_delegatable)
            
            # Generate unique characteristics for each layer
            layer_name = f"{layer_type}_{i}"
            self.layer_characteristics[i] = {
                'complexity_factor': np.random.uniform(0.8, 1.5),  # Layer complexity variation
                'memory_intensity': np.random.uniform(0.5, 2.0),   # Memory access pattern
                'parallelization_efficiency': np.random.uniform(0.6, 1.0),  # How well it maps to SA
                'thermal_impact': np.random.uniform(0.8, 1.3),     # Heat generation
                'cache_affinity': np.random.uniform(0.7, 1.2)      # CPU cache utilization
            }
        
        num_delegatable = sum(self.delegatable_layers)
        print(f"Total layers in model: {self.num_layers}")
        print(f"Delegatable layers (CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED): {num_delegatable}")
        print(f"Non-delegatable layers: {self.num_layers - num_delegatable}")
        print(f"Layer types: {self.profiling_data['layer_type'].value_counts().to_dict()}")
        
        self.num_particles = 20
        self.max_iterations = 100
        self.min_iterations = 10
        self.energy_convergence_threshold = 0.01  # Stop if energy improvement < 1%
        self.energy_target = None  # Will be set based on CPU-only baseline
        self.stagnation_limit = 15  # Stop if no improvement for 15 iterations
        
        self.w = 0.729  # Inertia weight (constriction coefficient)
        self.c1 = 1.49445  # Cognitive parameter 
        self.c2 = 1.49445  # Social parameter
        
    def sigmoid(self, x):
        """Sigmoid function for binary conversion"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def evaluate_partition(self, binary_partition: np.ndarray) -> float:
        """
        Evaluate partition quality using realistic energy-optimized cost function
        
        Args:
            binary_partition: Binary array where 0=CPU, 1=SA_Accelerator
            
        Returns:
            Fitness score (lower is better) - optimized for energy efficiency
        """
        total_energy_cost = 0.0
        communication_cost = 0.0
        thermal_cost = 0.0
        invalid_delegations = 0
        sa_setup_overhead = 0.0
        
        # Count SA accelerator usage for setup overhead
        sa_delegated_count = sum([1 for i, decision in enumerate(binary_partition) 
                                if decision == 1 and self.delegatable_layers[i]])
        
        for i, partition_decision in enumerate(binary_partition):
            if i >= len(self.profiling_data):
                break
                
            row = self.profiling_data.iloc[i]
            layer_type = row.get('layer_type', 'UNKNOWN')
            layer_chars = self.layer_characteristics[i]
            
            # Check if trying to delegate a non-delegatable layer
            if partition_decision == 1 and not self.delegatable_layers[i]:
                # Heavy penalty for invalid delegations
                invalid_delegations += 1
                partition_decision = 0  # Force to CPU for cost calculation
            
            if partition_decision == 0:  # CPU
                # Use CPU energy cost from comprehensive profiling data
                base_energy = row.get('energy_cost', 1000000)  # Use actual energy cost from profiling
                layer_energy = base_energy * layer_chars['complexity_factor'] * layer_chars['cache_affinity']
                
            else:  # SA Accelerator (only for delegatable layers)
                # Use SA accelerator energy cost with realistic benefits and overheads
                base_energy = row.get('energy_cost', 1000000)  # Base energy from profiling
                
                # SA accelerator provides benefits for compute-intensive layers
                compute_intensity = row.get('compute_intensity', 1.0)
                total_gmacs = row.get('total_gmacs', 0)
                # SA accelerator provides benefits for compute-intensive layers
                compute_intensity = row.get('compute_intensity', 1.0)
                total_gmacs = row.get('total_gmacs', 0)
                
                # SA accelerator efficiency varies per layer
                efficiency = layer_chars['parallelization_efficiency']
                memory_overhead = layer_chars['memory_intensity'] * 1.2  # Memory transfers cost more
                
                if layer_type in ['CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED']:
                    # SA provides benefits for high-GMAC layers but with diminishing returns
                    if total_gmacs > 1000:  # High compute layers benefit more
                        energy_reduction = 0.3 + (0.4 * min(total_gmacs / 10000, 1.0))  # 30-70% reduction
                    else:
                        energy_reduction = 0.1  # Small layers get minimal benefit
                    
                    layer_energy = base_energy * (1.0 - energy_reduction) * memory_overhead
                    thermal_cost += base_energy * layer_chars['thermal_impact'] * 0.3
                else:
                    # This should not happen due to constraint enforcement above
                    layer_energy = base_energy * 3.0  # Heavy penalty if it somehow occurs
            
            total_energy_cost += layer_energy
                
            # Communication cost between adjacent layers on different devices
            if i > 0 and binary_partition[i] != binary_partition[i-1]:
                # Communication overhead varies based on actual communication cost from profiling
                comm_base = row.get('communication_cost', 1000)
                communication_cost += comm_base * layer_chars['memory_intensity'] * 0.5
        
        # SA accelerator setup and teardown overhead (makes small delegations inefficient)
        if sa_delegated_count > 0:
            sa_setup_overhead = 500 + (sa_delegated_count * 50)  # Fixed + per-layer overhead
        
        # Heavy penalty for invalid delegations
        invalid_penalty = invalid_delegations * 50000
        
        # Total cost with realistic constraints
        total_cost = (total_energy_cost + 
                     communication_cost * 2.0 + 
                     thermal_cost + 
                     sa_setup_overhead + 
                     invalid_penalty)
        
        return total_cost
    
    def run_bpso_optimization(self) -> Tuple[np.ndarray, float]:
        """
        Run Binary Particle Swarm Optimization with energy-based convergence
        
        Returns:
            best_partition: Optimal binary partition
            best_fitness: Best fitness score achieved
        """
        # Initialize particles
        particles = np.random.uniform(-4, 4, (self.num_particles, self.num_layers))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.num_layers))
        
        # Initialize binary particles with constraints
        binary_particles = (np.random.rand(self.num_particles, self.num_layers) < 0.3).astype(int)
        
        # Enforce delegatable constraint in initial population
        for i in range(self.num_particles):
            for j in range(self.num_layers):
                if not self.delegatable_layers[j]:
                    binary_particles[i][j] = 0  # Force non-delegatable layers to CPU
        
        personal_best_positions = binary_particles.copy()
        personal_best_fitness = np.array([self.evaluate_partition(p) for p in binary_particles])
        
        # Find global best
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        # Calculate CPU-only baseline for energy target
        cpu_only_partition = np.zeros(self.num_layers, dtype=int)
        cpu_only_energy = self.evaluate_partition(cpu_only_partition)
        self.energy_target = cpu_only_energy * 0.85  # Target 15% energy reduction
        
        print(f"CPU-only baseline energy: {cpu_only_energy:.2f}")
        print(f"Target energy (15% reduction): {self.energy_target:.2f}")
        print(f"Initial best fitness: {global_best_fitness:.2f}")
        
        # Convergence tracking
        stagnation_count = 0
        last_best_fitness = global_best_fitness
        
        # BPSO iterations with energy-based stopping
        for iteration in range(self.max_iterations):
            iteration_improved = False
            
            for i in range(self.num_particles):
                # Update velocity
                r1, r2 = np.random.rand(2)
                
                velocities[i] = (self.w * velocities[i] + 
                               self.c1 * r1 * (personal_best_positions[i] - binary_particles[i]) +
                               self.c2 * r2 * (global_best_position - binary_particles[i]))
                
                # Update position using sigmoid
                particles[i] += velocities[i]
                sigmoid_probs = self.sigmoid(particles[i])
                binary_particles[i] = (np.random.rand(self.num_layers) < sigmoid_probs).astype(int)
                
                # Enforce delegatable constraint after update
                for j in range(self.num_layers):
                    if not self.delegatable_layers[j]:
                        binary_particles[i][j] = 0  # Force non-delegatable layers to CPU
                
                # Evaluate new position
                fitness = self.evaluate_partition(binary_particles[i])
                
                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = binary_particles[i].copy()
                    
                    # Update global best
                    if fitness < global_best_fitness:
                        global_best_fitness = fitness
                        global_best_position = binary_particles[i].copy()
                        iteration_improved = True
            
            # Check for convergence
            energy_improvement = (last_best_fitness - global_best_fitness) / last_best_fitness
            
            if iteration >= self.min_iterations:
                # Check energy convergence
                if energy_improvement < self.energy_convergence_threshold:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                
                # Check if we reached energy target
                if global_best_fitness <= self.energy_target:
                    print(f"Energy target reached at iteration {iteration}")
                    break
                
                # Check stagnation
                if stagnation_count >= self.stagnation_limit:
                    print(f"Convergence reached at iteration {iteration} (stagnation)")
                    break
            
            if iteration % 10 == 0:
                delegated_count = sum([1 for i, decision in enumerate(global_best_position) 
                                     if decision == 1 and self.delegatable_layers[i]])
                print(f"Iteration {iteration}: Best fitness = {global_best_fitness:.2f}, "
                      f"Energy improvement = {energy_improvement:.4f}, "
                      f"Delegated layers = {delegated_count}")
            
            last_best_fitness = global_best_fitness
        
        final_delegated = sum([1 for i, decision in enumerate(global_best_position) 
                             if decision == 1 and self.delegatable_layers[i]])
        energy_savings = (cpu_only_energy - global_best_fitness) / cpu_only_energy * 100
        
        print(f"Final best fitness: {global_best_fitness:.2f}")
        print(f"Energy savings vs CPU-only: {energy_savings:.1f}%")
        print(f"Final delegated layers: {final_delegated}")
        
        return global_best_position, global_best_fitness
    
    def generate_partition_config(self, binary_partition: np.ndarray, output_csv: str):
        """Generate partition configuration CSV for the delegate"""
        
        partition_data = []
        
        for i in range(len(self.profiling_data)):
            row = self.profiling_data.iloc[i]
            layer_type = row.get('layer_type', 'UNKNOWN')
            
            # Use BPSO decision for delegatable layers, force CPU for non-delegatable
            if i < len(binary_partition) and self.delegatable_layers[i]:
                partition_decision = int(binary_partition[i])
            else:
                partition_decision = 0  # Force to CPU if not delegatable or no decision available
                
            partition_data.append({
                'layer_id': i,
                'layer_type': layer_type,
                'partition_decision': partition_decision,  # 0=CPU, 1=SA_Accelerator
                'cpu_cycles': row.get('cpu_cycles', 1000),
                'sa_accelerator_cycles': row.get('sa_accelerator_cycles', 300),
                'communication_cost': row.get('transfer_overhead_cycles', 100),
                'delegatable': self.delegatable_layers[i],
                'complexity_factor': self.layer_characteristics[i]['complexity_factor'],
                'parallelization_efficiency': self.layer_characteristics[i]['parallelization_efficiency']
            })
        
        # Save to CSV
        df = pd.DataFrame(partition_data)
        df.to_csv(output_csv, index=False)
        
        # Print summary
        total_delegated = sum([1 for item in partition_data if item['partition_decision'] == 1])
        total_cpu = len(partition_data) - total_delegated
        
        # Count delegatable layers for validation
        delegatable_count = sum(self.delegatable_layers)
        delegated_delegatable = sum([1 for i, item in enumerate(partition_data) 
                                  if item['partition_decision'] == 1 and self.delegatable_layers[i]])
        
        print(f"\n=== BPSO Partition Results ===")
        print(f"Total layers: {len(partition_data)}")
        print(f"Delegatable layers (CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED): {delegatable_count}")
        print(f"SA Accelerator layers: {total_delegated} (all are delegatable: {delegated_delegatable == total_delegated})")
        print(f"CPU layers: {total_cpu}")
        print(f"Partition saved to: {output_csv}")
        
        # Show first 10 layer assignments
        print(f"\nFirst 10 layer assignments:")
        for i in range(min(10, len(partition_data))):
            item = partition_data[i]
            unit = "SA_ACCELERATOR" if item['partition_decision'] == 1 else "CPU"
            delegatable_str = "(delegatable)" if self.delegatable_layers[i] else "(CPU-only)"
            complexity = item['complexity_factor']
            efficiency = item['parallelization_efficiency']
            print(f"  Layer {i} ({item['layer_type']}) -> {unit} {delegatable_str} "
                  f"[complexity={complexity:.2f}, efficiency={efficiency:.2f}]")
        
        if len(partition_data) > 10:
            print(f"  ... and {len(partition_data) - 10} more layers")
        
        return partition_data

def main():
    """Main function to run BPSO optimization and generate partition config"""
    import sys
    
    # Handle command line arguments or use default paths
    if len(sys.argv) > 1:
        profiling_data_path = sys.argv[1]
    else:
        # Use comprehensive partitioning metrics that include layer types and energy costs
        profiling_data_path = "results/mobilenetv1_baseline_partitioning_metrics.csv"
    
    # Output path
    partition_config_path = "outputs/bpso_partition_config.csv"
    
    if not os.path.exists(profiling_data_path):
        print(f"Error: Profiling data not found: {profiling_data_path}")
        print("Please run comprehensive profiling first to generate the partitioning metrics")
        return
    
    print("=== BPSO Layer Partitioning Optimization ===")
    print(f"Loading profiling data from: {profiling_data_path}")
    
    # Initialize BPSO optimizer
    optimizer = BPSOPartitionOptimizer(profiling_data_path)
    
    print(f"Optimizing partition for {optimizer.num_layers} layers...")
    
    # Run BPSO optimization
    optimal_partition, best_fitness = optimizer.run_bpso_optimization()
    
    # Generate partition configuration
    partition_data = optimizer.generate_partition_config(optimal_partition, partition_config_path)
    
    print(f"\n=== How to Use BPSO Partition ===")
    print(f"1. Copy {partition_config_path} to your model directory")
    print(f"2. Run SA sim delegate with: --bpso_partition_config={partition_config_path}")
    print(f"3. The delegate will use BPSO decisions instead of default Conv2D delegation")
    
    # Save binary partition vector for programmatic use
    binary_vector_path = "outputs/bpso_binary_partition.txt"
    np.savetxt(binary_vector_path, optimal_partition, fmt='%d')
    print(f"4. Binary partition vector saved to: {binary_vector_path}")

if __name__ == "__main__":
    main()
