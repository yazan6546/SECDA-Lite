#!/usr/bin/env python3
"""
Per-Layer Profiling for SECDA-Lite DNN Partitioning Algorithms

This script provides detailed per-layer profiling metrics including:
- Clock cycles per layer/operation
- Memory access costs per layer
- Data movement costs
- Compute intensity per layer
- Resource utilization per layer

Perfect for partitioning algorithms that need to make decisions at layer granularity.
"""

import subprocess
import os
import re
import json
import csv
from typing import Dict, List, Tuple, Any
import pandas as pd
from pathlib import Path

class LayerProfiler:
    def __init__(self, workspace_dir: str = "/root/Workspace/tensorflow"):
        self.workspace_dir = workspace_dir
        self.models_dir = f"{workspace_dir}/models"
        self.outputs_dir = f"{workspace_dir}/outputs"
        self.results_dir = f"{workspace_dir}/results"
        
        # Ensure output directories exist
        os.makedirs(self.outputs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Layer type mappings for better readability
        self.layer_type_map = {
            'kTfLiteBuiltinConv2d': 'CONV2D',
            'kTfLiteBuiltinDepthwiseConv2d': 'DEPTHWISE_CONV2D',
            'kTfLiteBuiltinAveragePool2d': 'AVERAGE_POOL2D',
            'kTfLiteBuiltinMaxPool2d': 'MAX_POOL2D',
            'kTfLiteBuiltinRelu': 'RELU',
            'kTfLiteBuiltinRelu6': 'RELU6',
            'kTfLiteBuiltinReshape': 'RESHAPE',
            'kTfLiteBuiltinSoftmax': 'SOFTMAX',
            'kTfLiteBuiltinFullyConnected': 'FULLY_CONNECTED',
            'kTfLiteBuiltinAdd': 'ADD',
            'kTfLiteBuiltinMul': 'MUL'
        }

    def get_layer_info_from_tflite_verbose(self, model_path: str, binary_path: str) -> List[Dict]:
        """Extract layer information from TFLite model using verbose output"""
        cmd = [
            binary_path,
            f"--model={model_path}",
            "--image=test_images/grace_hopper.bmp",
            "--verbose=3"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_dir)
            layers = []
            
            # Parse the verbose output to extract layer information
            lines = result.stderr.split('\n') + result.stdout.split('\n')
            
            layer_idx = 0
            for line in lines:
                if 'Node' in line and 'inputs:' in line:
                    # Extract layer information from verbose output
                    layer_info = {
                        'layer_idx': layer_idx,
                        'layer_name': f"layer_{layer_idx}",
                        'layer_type': 'UNKNOWN',
                        'input_shape': None,
                        'output_shape': None,
                        'params': 0
                    }
                    
                    # Try to extract operation type
                    if 'CONV_2D' in line:
                        layer_info['layer_type'] = 'CONV2D'
                    elif 'DEPTHWISE_CONV_2D' in line:
                        layer_info['layer_type'] = 'DEPTHWISE_CONV2D'
                    elif 'AVERAGE_POOL_2D' in line:
                        layer_info['layer_type'] = 'AVERAGE_POOL2D'
                    elif 'MAX_POOL_2D' in line:
                        layer_info['layer_type'] = 'MAX_POOL2D'
                    elif 'FULLY_CONNECTED' in line:
                        layer_info['layer_type'] = 'FULLY_CONNECTED'
                    
                    layers.append(layer_info)
                    layer_idx += 1
                    
            return layers
            
        except Exception as e:
            print(f"Error extracting layer info: {e}")
            return []

    def parse_systemc_profiling_per_layer(self, csv_file: str) -> Dict[str, Dict]:
        """Parse SystemC profiling CSV to extract per-layer metrics"""
        layer_metrics = {}
        
        if not os.path.exists(csv_file):
            print(f"Warning: Profiling CSV {csv_file} not found")
            return layer_metrics
            
        try:
            df = pd.read_csv(csv_file)
            
            for idx, row in df.iterrows():
                layer_name = f"layer_{idx}"
                
                # Extract cycle counts and costs per layer
                layer_metrics[layer_name] = {
                    'clock_cycles': row.get('ClockCycles', 0),
                    'process_cycles': row.get('ProcessCycles', 0),
                    'memory_read_cycles': row.get('MemoryReadCycles', 0),
                    'memory_write_cycles': row.get('MemoryWriteCycles', 0),
                    'compute_cycles': row.get('ComputeCycles', 0),
                    'data_movement_cycles': row.get('DataMovementCycles', 0),
                    'buffer_usage': row.get('BufferUsage', 0),
                    'power_cost': row.get('PowerCost', 0),
                    'energy_cost': row.get('EnergyCost', 0)
                }
                
        except Exception as e:
            print(f"Error parsing SystemC profiling: {e}")
            
        return layer_metrics

    def profile_model_per_layer(self, model_name: str, delegate_type: str = "sa_sim") -> Dict:
        """Profile a model and return per-layer metrics for partitioning algorithms"""
        
        model_path = f"{self.models_dir}/{model_name}"
        binary_path = f"bazel-bin/tensorflow/lite/delegates/utils/{delegate_type}_delegate/label_image_plus_{delegate_type}_delegate"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        print(f"Profiling {model_name} with {delegate_type} delegate...")
        
        # Clean previous profiling data
        csv_output = f"{self.outputs_dir}/{delegate_type}_sim.csv"
        if os.path.exists(csv_output):
            os.remove(csv_output)
            
        # Run inference with delegate enabled
        cmd = [
            binary_path,
            f"--model={model_path}",
            "--image=test_images/grace_hopper.bmp",
            f"--use_{delegate_type}_delegate=true",
            "--verbose=1"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_dir)
            
            # Get layer information
            layers = self.get_layer_info_from_tflite_verbose(model_path, binary_path)
            
            # Parse SystemC profiling data
            layer_metrics = self.parse_systemc_profiling_per_layer(csv_output)
            
            # Combine layer info with profiling metrics
            profile_data = {
                'model_name': model_name,
                'delegate_type': delegate_type,
                'total_layers': len(layers),
                'layers': {}
            }
            
            for i, layer in enumerate(layers):
                layer_name = f"layer_{i}"
                layer_metrics_data = layer_metrics.get(layer_name, {})
                
                profile_data['layers'][layer_name] = {
                    'layer_info': layer,
                    'performance_metrics': layer_metrics_data,
                    'partitioning_metrics': self.calculate_partitioning_metrics(layer, layer_metrics_data)
                }
                
            return profile_data
            
        except Exception as e:
            print(f"Error profiling model: {e}")
            return {}

    def calculate_partitioning_metrics(self, layer_info: Dict, perf_metrics: Dict) -> Dict:
        """Calculate specific metrics useful for partitioning algorithms"""
        
        # Base metrics
        clock_cycles = perf_metrics.get('clock_cycles', 0)
        memory_cycles = perf_metrics.get('memory_read_cycles', 0) + perf_metrics.get('memory_write_cycles', 0)
        compute_cycles = perf_metrics.get('compute_cycles', 0)
        
        # Partitioning-specific metrics
        partitioning_metrics = {
            # Core cycle metrics (what you want for partitioning)
            'total_cycles': clock_cycles,
            'compute_cycles': compute_cycles,
            'memory_access_cycles': memory_cycles,
            'data_movement_cycles': perf_metrics.get('data_movement_cycles', 0),
            
            # Compute intensity (compute/memory ratio) - important for device selection
            'compute_intensity': compute_cycles / max(memory_cycles, 1),
            
            # Operation complexity (for scheduling)
            'operation_weight': clock_cycles * (1 + layer_info.get('params', 0) / 1000),
            
            # Resource requirements
            'buffer_requirement': perf_metrics.get('buffer_usage', 0),
            'energy_cost': perf_metrics.get('energy_cost', 0),
            
            # Layer characteristics for partitioning decisions
            'layer_type': layer_info.get('layer_type', 'UNKNOWN'),
            'is_compute_intensive': compute_cycles > memory_cycles,
            'is_memory_intensive': memory_cycles > compute_cycles,
            
            # Communication cost (for edge partitioning)
            'communication_cost': self.estimate_communication_cost(layer_info),
            
            # Parallelization potential
            'parallelization_factor': self.estimate_parallelization_factor(layer_info)
        }
        
        return partitioning_metrics

    def estimate_communication_cost(self, layer_info: Dict) -> int:
        """Estimate communication cost for layer boundary crossing"""
        # This would be based on output tensor size and data movement
        # For now, use a simple heuristic based on layer type
        layer_type = layer_info.get('layer_type', 'UNKNOWN')
        
        comm_costs = {
            'CONV2D': 100,
            'DEPTHWISE_CONV2D': 50,
            'FULLY_CONNECTED': 200,
            'AVERAGE_POOL2D': 25,
            'MAX_POOL2D': 25,
            'RELU': 10,
            'RELU6': 10,
            'RESHAPE': 5,
            'SOFTMAX': 30
        }
        
        return comm_costs.get(layer_type, 50)

    def estimate_parallelization_factor(self, layer_info: Dict) -> float:
        """Estimate how well a layer can be parallelized"""
        layer_type = layer_info.get('layer_type', 'UNKNOWN')
        
        parallel_factors = {
            'CONV2D': 0.8,         # Highly parallelizable
            'DEPTHWISE_CONV2D': 0.9, # Very parallelizable
            'FULLY_CONNECTED': 0.7,  # Moderately parallelizable
            'AVERAGE_POOL2D': 0.9,   # Highly parallelizable
            'MAX_POOL2D': 0.9,       # Highly parallelizable
            'RELU': 1.0,             # Perfectly parallelizable
            'RELU6': 1.0,            # Perfectly parallelizable
            'RESHAPE': 0.3,          # Limited parallelization
            'SOFTMAX': 0.4           # Sequential dependencies
        }
        
        return parallel_factors.get(layer_type, 0.5)

    def generate_partitioning_report(self, profile_data: Dict) -> str:
        """Generate a detailed report for partitioning algorithms"""
        
        model_name = profile_data['model_name']
        report_file = f"{self.results_dir}/{model_name}_partitioning_profile.json"
        
        # Save detailed JSON for algorithmic use
        with open(report_file, 'w') as f:
            json.dump(profile_data, f, indent=2)
            
        # Create CSV for easy analysis
        csv_file = f"{self.results_dir}/{model_name}_partitioning_metrics.csv"
        
        rows = []
        for layer_name, layer_data in profile_data['layers'].items():
            row = {
                'layer_name': layer_name,
                'layer_type': layer_data['layer_info']['layer_type'],
                **layer_data['partitioning_metrics']
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        
        print(f"Partitioning profile saved to: {report_file}")
        print(f"Partitioning metrics CSV saved to: {csv_file}")
        
        return report_file

    def profile_all_models_for_partitioning(self):
        """Profile all available models for partitioning algorithm development"""
        
        models = [
            'mobilenetv1.tflite',
            'mobilenetv2.tflite',
            'efficientnet_lite.tflite',
            'resnet18v1.tflite'
        ]
        
        delegates = ['sa_sim', 'vm_sim', 'bert_sim']
        
        all_profiles = {}
        
        for model in models:
            if os.path.exists(f"{self.models_dir}/{model}"):
                print(f"\n=== Profiling {model} ===")
                
                for delegate in delegates:
                    try:
                        profile_data = self.profile_model_per_layer(model, delegate)
                        if profile_data:
                            key = f"{model}_{delegate}"
                            all_profiles[key] = profile_data
                            self.generate_partitioning_report(profile_data)
                            
                    except Exception as e:
                        print(f"Failed to profile {model} with {delegate}: {e}")
                        
        # Generate comparative analysis
        self.generate_comparative_analysis(all_profiles)
        
        return all_profiles

    def generate_comparative_analysis(self, all_profiles: Dict):
        """Generate comparative analysis across models and delegates"""
        
        comparison_file = f"{self.results_dir}/partitioning_comparative_analysis.csv"
        
        rows = []
        for profile_key, profile_data in all_profiles.items():
            model_name = profile_data['model_name']
            delegate_type = profile_data['delegate_type']
            
            # Aggregate metrics across layers
            total_cycles = 0
            total_compute_cycles = 0
            total_memory_cycles = 0
            total_energy = 0
            layer_count = len(profile_data['layers'])
            
            for layer_name, layer_data in profile_data['layers'].items():
                metrics = layer_data['partitioning_metrics']
                total_cycles += metrics.get('total_cycles', 0)
                total_compute_cycles += metrics.get('compute_cycles', 0)
                total_memory_cycles += metrics.get('memory_access_cycles', 0)
                total_energy += metrics.get('energy_cost', 0)
                
            row = {
                'model': model_name,
                'delegate': delegate_type,
                'total_layers': layer_count,
                'total_cycles': total_cycles,
                'total_compute_cycles': total_compute_cycles,
                'total_memory_cycles': total_memory_cycles,
                'total_energy_cost': total_energy,
                'avg_cycles_per_layer': total_cycles / max(layer_count, 1),
                'compute_memory_ratio': total_compute_cycles / max(total_memory_cycles, 1)
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        df.to_csv(comparison_file, index=False)
        
        print(f"Comparative analysis saved to: {comparison_file}")

    def extract_pbso_dag_data(self, profile_data: Dict) -> pd.DataFrame:
        """Extract per-layer profiling data for PBSO+DAG optimization with static CPU vs SA accelerator partitioning"""
        
        dag_data = []
        
        for layer_name, layer_info in profile_data['layers'].items():
            metrics = layer_info['partitioning_metrics']
            
            # Static partitioning data for PBSO optimization (CPU vs SA accelerator only)
            layer_data = {
                'layer_id': layer_info['layer_id'],
                'layer_name': layer_name,
                'layer_type': metrics['layer_type'],
                
                # Static execution costs (clock cycles only - no timing)
                'cpu_cycles': metrics['total_cycles'],  # CPU baseline cycles
                'sa_accelerator_cycles': metrics.get('accelerator_cycles', int(metrics['total_cycles'] * 0.3)),  # SA accelerator cycles
                
                # Memory requirements for static allocation (bytes)
                'input_tensor_size': metrics.get('input_memory_size', 1024),
                'output_tensor_size': metrics.get('output_memory_size', 1024),
                'weight_size': metrics.get('weight_memory_size', 0),
                
                # Communication cost between layers (for static cut points)
                'transfer_overhead_cycles': 100,  # Fixed transfer cost between CPU and SA accelerator
                
                # Layer characteristics for static partitioning decisions
                'compute_intensity': metrics.get('compute_cycles', 1000),  # FLOPS equivalent
                'memory_accesses': metrics.get('memory_access_cycles', 500),
                
                # SA accelerator suitability (static analysis)
                'sa_suitable': metrics['layer_type'] in ['CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED'],
                'cpu_preferred': metrics['layer_type'] in ['POOLING', 'RELU', 'SOFTMAX', 'RESHAPE'],
                
                # Static energy costs
                'cpu_energy_nj': metrics.get('energy_cost', 100),
                'sa_energy_nj': metrics.get('energy_cost', 50) if metrics['layer_type'] in ['CONV_2D', 'DEPTHWISE_CONV_2D'] else 100,
            }
            
            dag_data.append(layer_data)
        
        # Create DataFrame and save to CSV for PBSO algorithm
        df = pd.DataFrame(dag_data)
        
        # Sort by layer_id to maintain DAG order
        df = df.sort_values('layer_id').reset_index(drop=True)
        
        # Save to CSV for PBSO algorithm consumption
        pbso_csv_file = f"{self.outputs_dir}/pbso_dag_profiling.csv"
        df.to_csv(pbso_csv_file, index=False)
        
        print(f"PBSO DAG profiling data saved to: {pbso_csv_file}")
        return df

def main():
    """Main function for per-layer profiling"""
    
    profiler = LayerProfiler()
    
    # Test with a single model first
    print("=== Testing per-layer profiling with MobileNetV1 ===")
    try:
        profile_data = profiler.profile_model_per_layer('mobilenetv1.tflite', 'sa_sim')
        if profile_data:
            report_file = profiler.generate_partitioning_report(profile_data)
            
            # Generate PBSO DAG data for optimization
            pbso_df = profiler.extract_pbso_dag_data(profile_data)
            
            # Print summary for verification
            print(f"\nSummary for PBSO+DAG partitioning algorithm:")
            print(f"Model: {profile_data['model_name']}")
            print(f"Total layers: {profile_data['total_layers']}")
            print(f"PBSO DAG data shape: {pbso_df.shape}")
            
            print(f"\nFirst 5 layers for DAG modeling:")
            for _, row in pbso_df.head().iterrows():
                print(f"  {row['layer_name']}: {row['layer_type']} - CPU:{row['cpu_cycles']} Accel:{row['accelerator_cycles']} cycles")
                
            # Show key metrics for partitioning
            print(f"\nKey metrics for partitioning:")
            print(f"  Total CPU cycles: {pbso_df['cpu_cycles'].sum():,}")
            print(f"  Total Accelerator cycles: {pbso_df['accelerator_cycles'].sum():,}")
            print(f"  Memory bound layers: {pbso_df['memory_bound'].sum()}")
            print(f"  Compute bound layers: {pbso_df['compute_bound'].sum()}")
                
    except Exception as e:
        print(f"Error in test profiling: {e}")
        
    # Uncomment to profile all models
    # print("\n=== Profiling all models for partitioning ===")
    # profiler.profile_all_models_for_partitioning()

if __name__ == "__main__":
    main()
