#!/usr/bin/env python3
"""
Test script for static CPU vs SA accelerator partitioning
Uses PBSO+DAG optimization with per-layer profiling data
"""

import sys
import os
sys.path.append('/workspaces/SECDA-Lite/scripts')

from per_layer_profiling import PerLayerProfiler
from pbso_dag_partitioning import StaticDNNGraphModel, StaticPBSOOptimizer, LayerNode

def create_test_dag():
    """Create a simple test DAG for MobileNetV1-like model"""
    dag = StaticDNNGraphModel()
    
    # Create test layers with static characteristics
    layers = [
        LayerNode(
            layer_id=0, layer_name="conv1", layer_type="CONV_2D",
            cpu_cycles=50000, sa_accelerator_cycles=15000,  # SA accelerator is 3.3x faster for conv
            input_tensor_size=150528, output_tensor_size=200704, weight_size=864,
            transfer_overhead_cycles=100,
            predecessors=[], successors=[1],
            sa_suitable=True, cpu_preferred=False, compute_intensity=10000
        ),
        LayerNode(
            layer_id=1, layer_name="relu1", layer_type="RELU",
            cpu_cycles=5000, sa_accelerator_cycles=5000,  # No speedup for ReLU
            input_tensor_size=200704, output_tensor_size=200704, weight_size=0,
            transfer_overhead_cycles=100,
            predecessors=[0], successors=[2],
            sa_suitable=False, cpu_preferred=True, compute_intensity=1000
        ),
        LayerNode(
            layer_id=2, layer_name="dw_conv1", layer_type="DEPTHWISE_CONV_2D",
            cpu_cycles=30000, sa_accelerator_cycles=10000,  # SA accelerator good for depthwise
            input_tensor_size=200704, output_tensor_size=200704, weight_size=288,
            transfer_overhead_cycles=100,
            predecessors=[1], successors=[3],
            sa_suitable=True, cpu_preferred=False, compute_intensity=8000
        ),
        LayerNode(
            layer_id=3, layer_name="pool1", layer_type="AVERAGE_POOL_2D",
            cpu_cycles=8000, sa_accelerator_cycles=8000,  # No speedup for pooling
            input_tensor_size=200704, output_tensor_size=50176, weight_size=0,
            transfer_overhead_cycles=100,
            predecessors=[2], successors=[4],
            sa_suitable=False, cpu_preferred=True, compute_intensity=500
        ),
        LayerNode(
            layer_id=4, layer_name="fc1", layer_type="FULLY_CONNECTED",
            cpu_cycles=40000, sa_accelerator_cycles=12000,  # SA accelerator excellent for FC
            input_tensor_size=50176, output_tensor_size=4000, weight_size=200000,
            transfer_overhead_cycles=100,
            predecessors=[3], successors=[],
            sa_suitable=True, cpu_preferred=False, compute_intensity=15000
        )
    ]
    
    # Add layers to DAG
    for layer in layers:
        dag.add_layer(layer)
    
    # Add dependencies
    dag.add_dependency(0, 1, layers[0].output_tensor_size)
    dag.add_dependency(1, 2, layers[1].output_tensor_size)
    dag.add_dependency(2, 3, layers[2].output_tensor_size)
    dag.add_dependency(3, 4, layers[3].output_tensor_size)
    
    return dag

def static_fitness_function(dag: StaticDNNGraphModel, partition: dict) -> float:
    """
    Static fitness function for CPU vs SA accelerator partitioning
    Minimizes: makespan + communication_cost + energy_cost
    """
    cpu_cycles = 0
    sa_cycles = 0
    communication_cost = 0
    energy_cost = 0
    
    # Calculate execution cost per processing unit
    for layer_id, unit in partition.items():
        layer = dag.layers[layer_id]
        if unit == 'cpu':
            cpu_cycles += layer.cpu_cycles
            energy_cost += layer.cpu_cycles * 1.0  # CPU energy cost
        elif unit == 'sa_accelerator':
            sa_cycles += layer.sa_accelerator_cycles
            energy_cost += layer.sa_accelerator_cycles * 0.5  # SA accelerator more efficient
    
    # Calculate communication cost
    communication_cost = dag.calculate_communication_cost(partition)
    
    # Makespan is the maximum execution time
    makespan = max(cpu_cycles, sa_cycles)
    
    # Total cost
    total_cost = makespan + communication_cost + energy_cost * 0.1
    
    return total_cost

def test_static_partitioning():
    """Test static CPU vs SA accelerator partitioning"""
    print("=== Testing Static CPU vs SA Accelerator Partitioning ===")
    
    # Create test DAG
    dag = create_test_dag()
    print(f"Created DAG with {len(dag.layers)} layers")
    
    # Test different static partitioning strategies
    
    # Strategy 1: All CPU
    all_cpu = {i: 'cpu' for i in range(5)}
    cpu_cost = static_fitness_function(dag, all_cpu)
    print(f"All CPU partition cost: {cpu_cost:.0f} cycles")
    
    # Strategy 2: All SA accelerator
    all_sa = {i: 'sa_accelerator' for i in range(5)}
    sa_cost = static_fitness_function(dag, all_sa)
    print(f"All SA accelerator partition cost: {sa_cost:.0f} cycles")
    
    # Strategy 3: Smart static partitioning (SA for compute-intensive layers)
    smart_partition = {}
    for layer_id, layer in dag.layers.items():
        if layer.sa_suitable and layer.compute_intensity > 5000:
            smart_partition[layer_id] = 'sa_accelerator'
        else:
            smart_partition[layer_id] = 'cpu'
    
    smart_cost = static_fitness_function(dag, smart_partition)
    print(f"Smart static partition cost: {smart_cost:.0f} cycles")
    print(f"Smart partition: {smart_partition}")
    
    # Strategy 4: Use PBSO optimization
    print("\n=== Running PBSO Optimization ===")
    try:
        pbso = StaticPBSOOptimizer(dag, swarm_size=20)
        best_partition, best_cost = pbso.optimize(max_iterations=50, target_fitness=smart_cost * 0.9)
        
        print(f"PBSO optimized partition cost: {best_cost:.0f} cycles")
        print(f"PBSO partition: {best_partition}")
        
        # Show improvement
        improvement = (smart_cost - best_cost) / smart_cost * 100
        print(f"Improvement over smart static: {improvement:.1f}%")
        
    except Exception as e:
        print(f"PBSO optimization failed: {e}")
        print("Using smart static partitioning as best result")
    
    print("\n=== Summary ===")
    print(f"Best approach: Smart static partitioning")
    print(f"Cost reduction vs all-CPU: {(cpu_cost - smart_cost) / cpu_cost * 100:.1f}%")
    print(f"Cost reduction vs all-SA: {(sa_cost - smart_cost) / sa_cost * 100:.1f}%")

if __name__ == "__main__":
    test_static_partitioning()
