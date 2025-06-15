#!/usr/bin/env python3
"""
PBSO (Population-Based Swarm Optimization) with DAG modeling for static DNN partiticlass StaticStaticPBSOOptimizer:
    """Population-Based Swarm Optimization for static CPU vs SA accelerator partitioning"""
    
    def __init__(self, dag_model: StaticStaticDNNGraphModel, swarm_size: int = 50):g
Static CPU vs SA (Systolic Array) accelerator partitioning only
Requires per-layer profiling data for accurate optimization
"""

import numpy as np
import pandas as pd
import networkx as nx
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json

@dataclass
class LayerNode:
    """DAG node representing a neural network layer for static partitioning"""
    layer_id: int
    layer_name: str
    layer_type: str
    
    # Static execution costs (clock cycles only)
    cpu_cycles: int
    sa_accelerator_cycles: int  # SA accelerator cycles
    
    # Memory requirements for static allocation
    input_tensor_size: int
    output_tensor_size: int
    weight_size: int
    
    # Static communication costs
    transfer_overhead_cycles: int  # Fixed transfer cost between CPU and SA
    
    # Dependencies (static DAG structure)
    predecessors: List[int]
    successors: List[int]
    
    # Static partitioning characteristics
    sa_suitable: bool      # Layer type suitable for SA accelerator
    cpu_preferred: bool    # Layer type preferred for CPU
    compute_intensity: int # FLOPS equivalent for static analysis

class StaticDNNGraphModel:
    """Static DAG representation of neural network for CPU vs SA accelerator partitioning"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.layers: Dict[int, LayerNode] = {}
        self.processing_units = ['cpu', 'sa_accelerator']  # Only CPU and SA accelerator
        
    def add_layer(self, layer: LayerNode):
        """Add layer node to DAG"""
        self.layers[layer.layer_id] = layer
        self.graph.add_node(layer.layer_id, **layer.__dict__)
        
    def add_dependency(self, from_layer: int, to_layer: int, data_size: int):
        """Add edge representing data dependency"""
        self.graph.add_edge(from_layer, to_layer, data_size=data_size)
        
    def get_critical_path(self) -> List[int]:
        """Find critical path in the DAG"""
        try:
            return nx.dag_longest_path(self.graph, weight='cpu_cycles')
        except:
            return list(nx.topological_sort(self.graph))
    
    def calculate_communication_cost(self, partition: Dict[int, str]) -> int:
        """Calculate total communication cost for static CPU vs SA accelerator partition"""
        total_cost = 0
        for edge in self.graph.edges(data=True):
            from_node, to_node, data = edge
            if partition[from_node] != partition[to_node]:
                # Different processing units - add static transfer cost
                layer = self.layers[from_node]
                transfer_cost = layer.transfer_overhead_cycles  # Fixed SA accelerator transfer cost
                total_cost += transfer_cost
        return total_cost
    
    def get_transfer_cost_per_byte(self, from_unit: str, to_unit: str) -> float:
        """Get static transfer cost between CPU and SA accelerator"""
        transfer_costs = {
            ('cpu', 'sa_accelerator'): 100,  # CPU to SA accelerator (cycles)
            ('sa_accelerator', 'cpu'): 100,  # SA accelerator to CPU (cycles)
        }
        return transfer_costs.get((from_unit, to_unit), 0.0)

class PBSOParticle:
    """Particle in PBSO representing a partition solution"""
    
    def __init__(self, num_layers: int, num_processing_units: int):
        self.position = np.random.randint(0, num_processing_units, num_layers)
        self.velocity = np.random.randn(num_layers) * 0.1
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.fitness = float('inf')
    
    def update_velocity(self, global_best_position: np.ndarray, w: float, c1: float, c2: float):
        """Update particle velocity"""
        r1, r2 = np.random.random(2)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social
    
    def update_position(self, num_processing_units: int):
        """Update particle position with constraints"""
        self.position = self.position + self.velocity
        # Clip to valid processing unit range
        self.position = np.clip(np.round(self.position), 0, num_processing_units - 1).astype(int)

class StaticPBSOOptimizer:
    """Population-Based Swarm Optimization for DAG partitioning"""
    
    def __init__(self, dag_model: StaticDNNGraphModel, swarm_size: int = 50):
        self.dag_model = dag_model
        self.swarm_size = swarm_size
        self.num_layers = len(dag_model.layers)
        self.num_processing_units = len(dag_model.processing_units)
        
        # PBSO parameters
        self.w = 0.9  # Inertia weight
        self.c1 = 2.0  # Cognitive parameter
        self.c2 = 2.0  # Social parameter
        self.max_iterations = 100
        
        # Initialize swarm
        self.swarm = [PBSOParticle(self.num_layers, self.num_processing_units) 
                      for _ in range(swarm_size)]
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        
    def evaluate_fitness(self, partition: np.ndarray) -> float:
        """Evaluate fitness of a partition solution"""
        partition_dict = {i: self.dag_model.processing_units[partition[i]] 
                         for i in range(len(partition))}
        
        # Calculate execution costs per processing unit
        execution_costs = {unit: 0 for unit in self.dag_model.processing_units}
        
        for layer_id, unit in partition_dict.items():
            layer = self.dag_model.layers[layer_id]
            if unit == 'cpu':
                execution_costs[unit] += layer.cpu_cycles
            elif unit == 'systemc_delegate':
                execution_costs[unit] += layer.systemc_delegate_cycles
        
        # Calculate communication costs
        comm_cost = self.dag_model.calculate_communication_cost(partition_dict)
        
        # Calculate memory constraints violations
        memory_penalty = self.calculate_memory_penalty(partition_dict)
        
        # Multi-objective fitness function
        makespan = max(execution_costs.values())  # Critical path length
        total_cost = sum(execution_costs.values()) + comm_cost
        
        # Fitness = weighted sum of objectives
        fitness = (0.6 * makespan +           # Minimize maximum execution time
                  0.3 * comm_cost +           # Minimize communication overhead  
                  0.1 * total_cost +          # Minimize total cost
                  1000 * memory_penalty)      # Heavy penalty for constraint violations
        
        return fitness
    
    def calculate_memory_penalty(self, partition_dict: Dict[int, str]) -> float:
        """Calculate penalty for memory constraint violations"""
        memory_limits = {
            'cpu': 8192,      # 8GB
            'systemc_delegate': 2048,  # 2GB SystemC delegate memory
        }
        
        memory_usage = {unit: 0 for unit in self.dag_model.processing_units}
        
        for layer_id, unit in partition_dict.items():
            layer = self.dag_model.layers[layer_id]
            memory_usage[unit] += (layer.input_memory + 
                                 layer.output_memory + 
                                 layer.weight_memory)
        
        penalty = 0
        for unit, usage in memory_usage.items():
            if usage > memory_limits[unit]:
                penalty += (usage - memory_limits[unit]) / memory_limits[unit]
        
        return penalty
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """Run PBSO optimization"""
        # Evaluate initial population
        for particle in self.swarm:
            particle.fitness = self.evaluate_fitness(particle.position)
            if particle.fitness < particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position.copy()
            
            if particle.fitness < self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            for particle in self.swarm:
                # Update velocity and position
                particle.update_velocity(self.global_best_position, 
                                       self.w, self.c1, self.c2)
                particle.update_position(self.num_processing_units)
                
                # Evaluate new position
                particle.fitness = self.evaluate_fitness(particle.position)
                
                # Update personal best
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
            
            # Decrease inertia weight
            self.w = max(0.4, self.w * 0.99)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.global_best_fitness:.2f}")
        
        return self.global_best_position, self.global_best_fitness

def load_per_layer_profiling_data(csv_file: str) -> StaticDNNGraphModel:
    """Load per-layer profiling data and create DAG model"""
    dag_model = StaticDNNGraphModel()
    
    # Load profiling data
    df = pd.read_csv(csv_file)
    
    # Create layer nodes from profiling data
    for _, row in df.iterrows():
        layer = LayerNode(
            layer_id=row['layer_id'],
            layer_name=row['layer_name'],
            layer_type=row['layer_type'],
            cpu_cycles=row['cpu_cycles'],
            systemc_delegate_cycles=row['accelerator_cycles'],
            input_memory=row['input_memory_kb'] * 1024,
            output_memory=row['output_memory_kb'] * 1024,
            weight_memory=row['weight_memory_kb'] * 1024,
            input_tensor_size=row['input_tensor_size'],
            output_tensor_size=row['output_tensor_size'],
            predecessors=[],  # Will be filled from dependencies
            successors=[],
            parallelizable=row.get('parallelizable', True),
            memory_bound=row.get('memory_bound', False),
            compute_bound=row.get('compute_bound', True)
        )
        dag_model.add_layer(layer)
    
    # Add dependencies based on layer sequence
    layer_ids = sorted(dag_model.layers.keys())
    for i in range(len(layer_ids) - 1):
        current_layer = layer_ids[i]
        next_layer = layer_ids[i + 1]
        data_size = dag_model.layers[current_layer].output_tensor_size
        dag_model.add_dependency(current_layer, next_layer, data_size)
    
    return dag_model

def load_profiling_data_to_dag(csv_file: str) -> StaticDNNGraphModel:
    """Load per-layer profiling data and create DAG model"""
    
    # Read CSV data
    df = pd.read_csv(csv_file)
    
    # Create DAG model
    dag_model = StaticDNNGraphModel()
    
    # Add layers to the model
    for _, row in df.iterrows():
        layer = LayerNode(
            layer_id=int(row['layer_id']),
            layer_name=str(row['layer_name']),
            layer_type=str(row['layer_type']),
            cpu_cycles=int(row['cpu_cycles']),
            systemc_delegate_cycles=int(row['systemc_delegate_cycles']),
            input_memory=int(row.get('input_memory_kb', 0) * 1024),
            output_memory=int(row.get('output_memory_kb', 0) * 1024),
            weight_memory=int(row.get('weight_memory_kb', 0) * 1024),
            input_tensor_size=int(row.get('input_tensor_size', 0)),
            output_tensor_size=int(row.get('output_tensor_size', 0)),
            predecessors=[],  # Will be filled based on layer sequence
            successors=[],    # Will be filled based on layer sequence
            parallelizable=bool(row.get('parallelizable', False)),
            memory_bound=bool(row.get('memory_bound', False)),
            compute_bound=bool(row.get('compute_bound', False))
        )
        
        dag_model.add_layer(layer)
    
    # Add sequential dependencies (each layer depends on previous)
    sorted_layers = sorted(dag_model.layers.keys())
    for i in range(len(sorted_layers) - 1):
        current_layer_id = sorted_layers[i]
        next_layer_id = sorted_layers[i + 1]
        
        # Get output tensor size for communication cost
        current_layer = dag_model.layers[current_layer_id]
        data_size = current_layer.output_tensor_size
        
        dag_model.add_dependency(current_layer_id, next_layer_id, data_size)
    
    return dag_model

def run_pbso_partitioning(profiling_csv: str) -> Dict:
    """Run PBSO optimization on per-layer profiling data"""
    print("Loading per-layer profiling data...")
    dag_model = load_profiling_data_to_dag(profiling_csv)
    
    print(f"Created DAG with {len(dag_model.layers)} layers")
    print(f"Critical path: {dag_model.get_critical_path()}")
    
    print("Running PBSO optimization...")
    optimizer = StaticPBSOOptimizer(dag_model, swarm_size=50)
    best_partition, best_fitness = optimizer.optimize()
    
    # Convert result to readable format
    partition_result = {}
    for i, unit_idx in enumerate(best_partition):
        unit_name = dag_model.processing_units[unit_idx]
        layer_name = dag_model.layers[i].layer_name
        partition_result[layer_name] = unit_name
    
    # Calculate detailed metrics
    partition_dict = {i: dag_model.processing_units[best_partition[i]] 
                     for i in range(len(best_partition))}
    
    execution_costs = {unit: 0 for unit in dag_model.processing_units}
    for layer_id, unit in partition_dict.items():
        layer = dag_model.layers[layer_id]
        if unit == 'cpu':
            execution_costs[unit] += layer.cpu_cycles
        elif unit == 'accelerator':
            execution_costs[unit] += layer.accelerator_cycles
        elif unit == 'gpu':
            execution_costs[unit] += layer.gpu_cycles
    
    comm_cost = dag_model.calculate_communication_cost(partition_dict)
    
    results = {
        'partition': partition_result,
        'fitness': best_fitness,
        'execution_costs': execution_costs,
        'communication_cost': comm_cost,
        'makespan': max(execution_costs.values()),
        'total_execution_cycles': sum(execution_costs.values())
    }
    
    return results

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "outputs/per_layer_profiling.csv"
    
    results = run_pbso_partitioning(csv_file)
    
    print("\n" + "="*60)
    print("PBSO PARTITIONING RESULTS")
    print("="*60)
    
    print(f"Best fitness: {results['fitness']:.2f}")
    print(f"Makespan: {results['makespan']:,} cycles")
    print(f"Communication cost: {results['communication_cost']:,} cycles")
    print(f"Total execution: {results['total_execution_cycles']:,} cycles")
    
    print("\nPartition assignment:")
    for layer, unit in results['partition'].items():
        print(f"  {layer}: {unit}")
    
    print("\nExecution costs per unit:")
    for unit, cost in results['execution_costs'].items():
        print(f"  {unit}: {cost:,} cycles")
    
    # Save results
    with open('outputs/pbso_partitioning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to outputs/pbso_partitioning_results.json")
