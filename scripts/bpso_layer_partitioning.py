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
        # SA accelerator only supports convolution operations
        self.delegatable_layers = []
        self.layer_type_to_delegatable = {}
        
        for i, row in self.profiling_data.iterrows():
            layer_type = row.get('layer_type', 'UNKNOWN')
            is_delegatable = layer_type in ['CONV_2D', 'DEPTHWISE_CONV_2D']
            self.delegatable_layers.append(is_delegatable)
            self.layer_type_to_delegatable[layer_type] = is_delegatable
        
        num_delegatable = sum(self.delegatable_layers)
        print(f"Total layers in model: {self.num_layers}")
        print(f"Delegatable layers (CONV_2D, DEPTHWISE_CONV_2D): {num_delegatable}")
        print(f"Non-delegatable layers: {self.num_layers - num_delegatable}")
        print(f"Layer types: {self.profiling_data['layer_type'].value_counts().to_dict()}")
        print(f"Delegatable by type: {self.layer_type_to_delegatable}")
        
        # Enhanced BPSO parameters for better optimization
        self.num_particles = 50  # Increased from 20 for better exploration
        self.max_iterations = 200  # Increased from 50 for deeper search
        self.min_iterations = 30  # Minimum iterations before early termination
        
        # Adaptive parameters that change during optimization
        self.w_max = 0.9  # Maximum inertia weight (exploration)
        self.w_min = 0.4  # Minimum inertia weight (exploitation)
        self.c1 = 2.0  # Cognitive parameter 
        self.c2 = 2.0  # Social parameter
        
        # Advanced termination conditions
        self.convergence_threshold = 1e-6  # Stop if improvement < threshold
        self.stagnation_limit = 25  # Stop if no improvement for N iterations
        self.target_fitness = None  # Will be set based on a target performance goal
        
    def sigmoid(self, x):
        """Sigmoid function for binary conversion"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def evaluate_partition(self, binary_partition: np.ndarray) -> float:
        """
        Enhanced partition evaluation with realistic energy modeling and better cost function
        
        Args:
            binary_partition: Binary array where 0=CPU, 1=SA_Accelerator
            
        Returns:
            Fitness score (lower is better) - optimized for realistic energy efficiency
        """
        total_energy_cost = 0.0
        communication_cost = 0.0
        setup_overhead_cost = 0.0
        invalid_delegations = 0
        context_switch_cost = 0.0
        
        # Count SA accelerator delegations for overhead calculations
        sa_delegated_count = sum([1 for i, decision in enumerate(binary_partition) 
                                if decision == 1 and self.delegatable_layers[i]])
        
        for i, partition_decision in enumerate(binary_partition):
            if i >= len(self.profiling_data):
                break
                
            row = self.profiling_data.iloc[i]
            layer_type = row.get('layer_type', 'UNKNOWN')
            
            # Get realistic energy/cycle data from profiling
            cpu_cycles = row.get('total_cycles', 1000)
            energy_cost = row.get('energy_cost', cpu_cycles * 2.0)
            
            # Check if trying to delegate a non-delegatable layer
            if partition_decision == 1 and not self.delegatable_layers[i]:
                # Heavy penalty for invalid delegations
                invalid_delegations += 1
                partition_decision = 0  # Force to CPU for cost calculation
            
            if partition_decision == 0:  # CPU execution
                # CPU energy cost with realistic scaling
                layer_energy = energy_cost
                
            else:  # SA Accelerator execution (only for delegatable layers)
                # SA accelerator provides significant benefits for convolution operations
                if layer_type in ['CONV_2D', 'DEPTHWISE_CONV_2D']:
                    # More realistic energy savings based on actual profiling data
                    compute_cycles = row.get('compute_cycles', cpu_cycles * 0.6)
                    total_gmacs = row.get('total_gmacs', 0)
                    
                    # Energy savings depend on computational complexity
                    if total_gmacs > 1000:  # High-compute layers benefit more
                        energy_reduction = 0.15  # 85% energy reduction for high-GMAC ops
                    elif total_gmacs > 100:  # Medium-compute layers
                        energy_reduction = 0.25  # 75% energy reduction
                    else:  # Low-compute layers get less benefit
                        energy_reduction = 0.4   # 60% energy reduction
                    
                    layer_energy = energy_cost * energy_reduction
                    
                    # Add realistic SA accelerator setup overhead per layer
                    setup_overhead_cost += energy_cost * 0.02  # 2% overhead per delegated layer
                else:
                    # This should not happen due to constraint enforcement above
                    layer_energy = energy_cost * 3.0  # Heavy penalty if it somehow occurs
            
            total_energy_cost += layer_energy
                
            # More realistic communication cost between adjacent layers on different devices
            if i > 0 and binary_partition[i] != binary_partition[i-1]:
                # Communication overhead varies based on layer output size and complexity
                comm_base_cost = row.get('communication_cost', cpu_cycles * 0.1)
                buffer_size = row.get('buffer_requirement', 1024)
                
                # Higher communication cost for larger tensors
                communication_cost += comm_base_cost * (1.0 + buffer_size / 10000.0)
                
                # Context switching cost between CPU and SA accelerator
                context_switch_cost += energy_cost * 0.05  # 5% context switch penalty
        
        # SA accelerator initialization overhead (makes small delegations inefficient)
        if sa_delegated_count > 0:
            setup_overhead_cost += energy_cost * 0.1 * sa_delegated_count  # Per-layer setup cost
            setup_overhead_cost += 5000  # Fixed SA accelerator initialization cost
        
        # Heavy penalty for invalid delegations
        invalid_penalty = invalid_delegations * 50000
        
        # Penalty for too few SA delegations (underutilization)
        if sa_delegated_count < 3 and sa_delegated_count > 0:
            setup_overhead_cost += 10000  # Penalty for underutilizing SA accelerator
        
        # Total cost with more realistic factors
        total_cost = (total_energy_cost + 
                     communication_cost * 2.0 +  # Communication is expensive
                     setup_overhead_cost + 
                     context_switch_cost +
                     invalid_penalty)
        
        return total_cost
    
    def run_bpso_optimization(self) -> Tuple[np.ndarray, float]:
        """
        Enhanced Binary Particle Swarm Optimization with advanced termination conditions
        
        Returns:
            best_partition: Optimal binary partition
            best_fitness: Best fitness score achieved
        """
        print(f"Starting enhanced BPSO optimization...")
        print(f"Particles: {self.num_particles}, Max iterations: {self.max_iterations}")
        
        # Calculate CPU-only baseline for target setting
        cpu_only_partition = np.zeros(self.num_layers, dtype=int)
        cpu_baseline_fitness = self.evaluate_partition(cpu_only_partition)
        self.target_fitness = cpu_baseline_fitness * 0.6  # Target 40% improvement over CPU-only
        
        print(f"CPU-only baseline fitness: {cpu_baseline_fitness:.2f}")
        print(f"Target fitness (40% improvement): {self.target_fitness:.2f}")
        
        # Initialize particles with better strategy
        particles = np.random.uniform(-6, 6, (self.num_particles, self.num_layers))
        velocities = np.random.uniform(-2, 2, (self.num_particles, self.num_layers))
        
        # Initialize binary particles with smarter initialization
        binary_particles = np.zeros((self.num_particles, self.num_layers), dtype=int)
        
        for i in range(self.num_particles):
            for j in range(self.num_layers):
                if self.delegatable_layers[j]:
                    # Initialize delegatable layers with higher probability of delegation
                    binary_particles[i][j] = 1 if np.random.rand() < 0.7 else 0
                else:
                    binary_particles[i][j] = 0  # Force non-delegatable layers to CPU
        
        personal_best_positions = binary_particles.copy()
        personal_best_fitness = np.array([self.evaluate_partition(p) for p in binary_particles])
        
        # Find global best
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        print(f"Initial best fitness: {global_best_fitness:.2f}")
        
        # Advanced termination tracking
        stagnation_count = 0
        previous_best_fitness = global_best_fitness
        fitness_history = [global_best_fitness]
        
        # BPSO iterations with adaptive parameters
        for iteration in range(self.max_iterations):
            # Adaptive inertia weight (linearly decreasing from w_max to w_min)
            w = self.w_max - (self.w_max - self.w_min) * iteration / self.max_iterations
            
            for i in range(self.num_particles):
                # Update velocity with adaptive inertia weight
                r1, r2 = np.random.rand(2)
                
                velocities[i] = (w * velocities[i] + 
                               self.c1 * r1 * (personal_best_positions[i] - binary_particles[i]) +
                               self.c2 * r2 * (global_best_position - binary_particles[i]))
                
                # Velocity clamping to prevent explosion
                velocities[i] = np.clip(velocities[i], -6, 6)
                
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
                        stagnation_count = 0  # Reset stagnation counter
            
            # Track fitness improvement
            fitness_history.append(global_best_fitness)
            fitness_improvement = previous_best_fitness - global_best_fitness
            
            # Check for stagnation
            if fitness_improvement < self.convergence_threshold:
                stagnation_count += 1
            else:
                stagnation_count = 0
                
            previous_best_fitness = global_best_fitness
            
            # Enhanced progress reporting
            if iteration % 10 == 0 or iteration < 10:
                sa_delegated = sum([1 for i, decision in enumerate(global_best_position) 
                                  if decision == 1 and self.delegatable_layers[i]])
                improvement_vs_baseline = ((cpu_baseline_fitness - global_best_fitness) / cpu_baseline_fitness) * 100
                print(f"Iteration {iteration:3d}: Fitness = {global_best_fitness:12.2f}, "
                      f"Improvement = {improvement_vs_baseline:5.1f}%, "
                      f"SA layers = {sa_delegated:2d}, "
                      f"Inertia = {w:.3f}")
            
            # Advanced termination conditions
            if iteration >= self.min_iterations:
                # Early termination if target reached
                if global_best_fitness <= self.target_fitness:
                    print(f"Target fitness reached at iteration {iteration}!")
                    break
                    
                # Early termination if stagnated
                if stagnation_count >= self.stagnation_limit:
                    print(f"Optimization stagnated for {self.stagnation_limit} iterations at iteration {iteration}")
                    break
                    
                # Convergence check (very small improvements over recent iterations)
                if len(fitness_history) >= 20:
                    recent_improvement = fitness_history[-20] - fitness_history[-1]
                    if recent_improvement < self.convergence_threshold * 20:
                        print(f"Converged at iteration {iteration} (improvement: {recent_improvement:.6f})")
                        break
        
        final_improvement = ((cpu_baseline_fitness - global_best_fitness) / cpu_baseline_fitness) * 100
        sa_delegated_final = sum([1 for i, decision in enumerate(global_best_position) 
                                if decision == 1 and self.delegatable_layers[i]])
        
        print(f"Final best fitness: {global_best_fitness:.2f}")
        print(f"Energy savings vs CPU-only: {final_improvement:.1f}%")
        print(f"Final SA delegated layers: {sa_delegated_final}")
        
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
                'communication_cost': row.get('transfer_overhead_cycles', 100)
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
        print(f"Delegatable layers (CONV_2D, DEPTHWISE_CONV_2D): {delegatable_count}")
        print(f"SA Accelerator layers: {total_delegated} (all are delegatable: {delegated_delegatable == total_delegated})")
        print(f"CPU layers: {total_cpu}")
        print(f"Partition saved to: {output_csv}")
        
        # Show first 10 layer assignments with more details
        print(f"\nFirst 10 layer assignments:")
        for i in range(min(10, len(partition_data))):
            item = partition_data[i]
            unit = "SA_ACCELERATOR" if item['partition_decision'] == 1 else "CPU"
            delegatable_str = "(delegatable)" if self.delegatable_layers[i] else "(CPU-only)"
            
            # Get additional info from profiling data
            if i < len(self.profiling_data):
                row = self.profiling_data.iloc[i]
                energy_cost = row.get('energy_cost', 0)
                total_gmacs = row.get('total_gmacs', 0)
                complexity_str = f"energy={energy_cost//1000}k, gmacs={total_gmacs}"
            else:
                complexity_str = "no_data"
            
            print(f"  Layer {i:2d} ({item['layer_type']:15s}) -> {unit:13s} {delegatable_str:13s} [{complexity_str}]")
        
        if len(partition_data) > 10:
            print(f"  ... and {len(partition_data) - 10} more layers")
        
        return partition_data

def main():
    """Main function to run BPSO optimization and generate partition config"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="BPSO Layer Partitioning Optimization")
    parser.add_argument("--model", default="mobilenetv1.tflite", 
                       choices=["mobilenetv1.tflite", "mobilenetv2.tflite", "resnet18v1.tflite", "resnet50v2.tflite"],
                       help="Model to optimize partitioning for")
    parser.add_argument("--profiling_data", 
                       help="Path to profiling data CSV (auto-detected if not provided)")
    parser.add_argument("--output_config", default="outputs/bpso_partition_config.csv",
                       help="Output path for partition configuration")
    
    # Parse command line arguments or use defaults for backward compatibility
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        # Backward compatibility: if first arg doesn't start with --, treat as profiling data path
        args = argparse.Namespace()
        args.profiling_data = sys.argv[1]
        args.model = "mobilenetv1.tflite" 
        args.output_config = "outputs/bpso_partition_config.csv"
    else:
        args = parser.parse_args()
    
    # Determine profiling data path
    if args.profiling_data:
        profiling_data_path = args.profiling_data
    else:
        # Auto-detect based on model name
        model_name = args.model.replace('.tflite', '')
        profiling_data_path = f"results/{model_name}_baseline_partitioning_metrics.csv"
    
    # Output path
    partition_config_path = args.output_config
    
    if not os.path.exists(profiling_data_path):
        print(f"Error: Profiling data not found: {profiling_data_path}")
        print("Please run comprehensive profiling first to generate the partitioning metrics")
        print(f"Example: python3 scripts/comprehensive_bpso_profile.py --models {args.model}")
        return
    
    print("=== BPSO Layer Partitioning Optimization ===")
    print(f"Model: {args.model}")
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
    print(f"4. Model optimized: {args.model}")
    
    # Save binary partition vector for programmatic use
    model_name = args.model.replace('.tflite', '')
    binary_vector_path = f"outputs/{model_name}_bpso_binary_partition.txt"
    np.savetxt(binary_vector_path, optimal_partition, fmt='%d')
    print(f"5. Binary partition vector saved to: {binary_vector_path}")
    
    print(f"\n=== Multi-Model Usage ===")
    print(f"To optimize other models:")
    for model in ["mobilenetv1.tflite", "mobilenetv2.tflite", "resnet18v1.tflite", "resnet50v2.tflite"]:
        if model != args.model:
            print(f"  python3 scripts/bpso_layer_partitioning.py --model {model}")

if __name__ == "__main__":
    main()
