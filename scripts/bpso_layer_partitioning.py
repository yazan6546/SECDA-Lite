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
        
        print(f"Total layers in model: {self.num_layers}")
        print(f"All layers are delegatable for optimization")
        print(f"Layer types: {self.profiling_data['layer_type'].value_counts().to_dict()}")
        
        self.num_particles = 20
        self.max_iterations = 50
        self.w = 0.5  # Inertia weight
        self.c1 = 2.0  # Cognitive parameter
        self.c2 = 2.0  # Social parameter
        
    def sigmoid(self, x):
        """Sigmoid function for binary conversion"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def evaluate_partition(self, binary_partition: np.ndarray) -> float:
        """
        Evaluate partition quality using energy-optimized cost function
        
        Args:
            binary_partition: Binary array where 0=CPU, 1=SA_Accelerator
            
        Returns:
            Fitness score (lower is better) - optimized for energy efficiency
        """
        total_energy_cost = 0.0
        communication_cost = 0.0
        
        for i, partition_decision in enumerate(binary_partition):
            if i >= len(self.profiling_data):
                break
                
            row = self.profiling_data.iloc[i]
            layer_type = row.get('layer_type', 'UNKNOWN')
            
            if partition_decision == 0:  # CPU
                # Use CPU energy cost
                layer_energy = row.get('cpu_energy_nj', row.get('cpu_cycles', 1000) * 2.0)
                
            else:  # SA Accelerator
                # Use SA accelerator energy cost
                layer_energy = row.get('sa_energy_nj', row.get('sa_accelerator_cycles', 300) * 1.5)
                
                # SA accelerator provides energy savings for most operations
                if layer_type in ['CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED']:
                    layer_energy *= 0.3  # 70% energy reduction for compute-intensive ops
                elif layer_type in ['RELU', 'RELU6', 'ADD', 'MUL']:
                    layer_energy *= 0.8  # 20% energy reduction for element-wise ops  
                else:
                    layer_energy *= 1.1  # 10% energy penalty for other ops (small overhead)
            
            total_energy_cost += layer_energy
                
            # Communication cost between adjacent layers on different devices
            if i > 0 and binary_partition[i] != binary_partition[i-1]:
                communication_cost += row.get('transfer_overhead_cycles', 100) * 0.3
        
        # Total cost optimized for energy efficiency
        total_cost = total_energy_cost + communication_cost * 1.5
        
        return total_cost
    
    def run_bpso_optimization(self) -> Tuple[np.ndarray, float]:
        """
        Run Binary Particle Swarm Optimization
        
        Returns:
            best_partition: Optimal binary partition
            best_fitness: Best fitness score achieved
        """
        # Initialize particles
        particles = np.random.uniform(-4, 4, (self.num_particles, self.num_layers))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.num_layers))
        
        # Convert to binary and evaluate
        binary_particles = (np.random.rand(self.num_particles, self.num_layers) < 0.5).astype(int)
        personal_best_positions = binary_particles.copy()
        personal_best_fitness = np.array([self.evaluate_partition(p) for p in binary_particles])
        
        # Find global best
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        print(f"Initial best fitness: {global_best_fitness:.2f}")
        
        # BPSO iterations
        for iteration in range(self.max_iterations):
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
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best fitness = {global_best_fitness:.2f}")
        
        print(f"Final best fitness: {global_best_fitness:.2f}")
        return global_best_position, global_best_fitness
    
    def generate_partition_config(self, binary_partition: np.ndarray, output_csv: str):
        """Generate partition configuration CSV for the delegate"""
        
        partition_data = []
        
        for i in range(len(self.profiling_data)):
            row = self.profiling_data.iloc[i]
            layer_type = row.get('layer_type', 'UNKNOWN')
            
            # Use BPSO decision for all layers
            if i < len(binary_partition):
                partition_decision = int(binary_partition[i])
            else:
                partition_decision = 0  # Default to CPU if no decision available
                
            partition_data.append({
                'layer_id': i,
                'layer_type': layer_type,
                'partition_decision': partition_decision,  # 0=CPU, 1=SA_Accelerator
                'cpu_cycles': row.get('cpu_cycles', 1000),
                'sa_accelerator_cycles': row.get('sa_accelerator_cycles', 300),
                'communication_cost': row.get('transfer_overhead_cycles', 100)
            })
        
        # Save to CSV
        df = pd.DataFrame(partition_data)
        df.to_csv(output_csv, index=False)
        
        # Print summary
        total_delegated = sum([1 for item in partition_data if item['partition_decision'] == 1])
        total_cpu = len(partition_data) - total_delegated
        
        print(f"\n=== BPSO Partition Results ===")
        print(f"Total layers: {len(partition_data)}")
        print(f"SA Accelerator layers: {total_delegated}")
        print(f"CPU layers: {total_cpu}")
        print(f"Partition saved to: {output_csv}")
        
        # Show first 10 layer assignments
        print(f"\nFirst 10 layer assignments:")
        for i in range(min(10, len(partition_data))):
            item = partition_data[i]
            unit = "SA_ACCELERATOR" if item['partition_decision'] == 1 else "CPU"
            print(f"  Layer {i} ({item['layer_type']}) -> {unit}")
        
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
        profiling_data_path = "outputs/pbso_dag_profiling.csv"  # Default path from workspace root
    
    # Output path
    partition_config_path = "outputs/bpso_partition_config.csv"
    
    if not os.path.exists(profiling_data_path):
        print(f"Error: Profiling data not found: {profiling_data_path}")
        print("Please run per_layer_profiling.py first to generate the profiling data")
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
