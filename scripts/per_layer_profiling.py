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
            f"--tflite_model={model_path}",
            "--image=test_images/grace_hopper.bmp",
            "--labels=tensorflow/lite/examples/ios/camera/data/labels.txt",
            "-p", "1",
            "--verbose", "1"
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, cwd=self.workspace_dir)
            layers = []
            
            # Parse the verbose output to extract layer information
            lines = result.stderr.split('\n') + result.stdout.split('\n')
            
            layer_idx = 0
            for line in lines:
                # Look for pattern: "Node   X Operator Builtin Code Y OPERATION_NAME"
                if line.strip().startswith('Node') and 'Operator Builtin Code' in line:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        try:
                            node_id = int(parts[1])
                            builtin_code = int(parts[5])
                            op_name = parts[6] if len(parts) > 6 else 'UNKNOWN'
                            is_delegated = '(delegated)' in line
                            
                            layer_info = {
                                'layer_idx': node_id,
                                'layer_name': f"layer_{node_id}",
                                'layer_type': op_name,
                                'op_type': op_name,
                                'builtin_code': builtin_code,
                                'is_delegated': is_delegated,
                                'input_shape': None,
                                'output_shape': None,
                                'params': 0
                            }
                            
                            layers.append(layer_info)
                            
                        except (ValueError, IndexError):
                            continue
                            
            print(f"DEBUG: Extracted {len(layers)} layers from verbose output")
            return layers
            
        except Exception as e:
            print(f"Error extracting layer info: {e}")
            return []

    def parse_systemc_profiling_per_layer(self, csv_file: str, num_delegated_layers: int = 15) -> Dict[str, Dict]:
        """Parse SystemC profiling CSV to extract per-layer metrics by aggregating SystemC simulation steps"""
        layer_metrics = {}
        
        if not os.path.exists(csv_file):
            print(f"Warning: Profiling CSV {csv_file} not found")
            return layer_metrics
            
        try:
            df = pd.read_csv(csv_file)
            print(f"DEBUG: CSV has {len(df)} rows, aggregating for {num_delegated_layers} delegated layers")
            
            # Calculate metrics per layer by dividing total rows by number of delegated layers
            rows_per_layer = len(df) // num_delegated_layers if num_delegated_layers > 0 else 1
            
            for layer_idx in range(num_delegated_layers):
                start_row = layer_idx * rows_per_layer
                end_row = start_row + rows_per_layer
                
                # Handle last layer to include any remaining rows
                if layer_idx == num_delegated_layers - 1:
                    end_row = len(df)
                
                layer_data = df.iloc[start_row:end_row]
                layer_name = f"delegated_layer_{layer_idx}"
                
                # Aggregate SystemC metrics for this layer
                layer_metrics[layer_name] = {
                    # Sum up all cycles for this layer
                    'read_cycles': int(layer_data['read_cycles'].sum()),
                    'process_cycles': int(layer_data['process_cycles'].sum()),
                    'idle_cycles': int(layer_data['idle'].sum()),
                    'gemmw_cycles': int(layer_data['gemmw'].sum()),
                    'gemm_cycles': int(layer_data['gemm'].sum()),
                    'wstall_cycles': int(layer_data['wstall'].sum()),
                    
                    # Maximum buffer usage for this layer
                    'max_input_buffer': int(layer_data['inputbuf_p'].max()),
                    'max_weight_buffer': int(layer_data['weightbuf_p'].max()),
                    
                    # Total operations for this layer
                    'total_gmacs': int(layer_data['gmacs'].sum()),
                    'total_outputs': int(layer_data['gouts'].sum()),
                    
                    # Calculated metrics
                    'total_cycles': int(layer_data['read_cycles'].sum() + 
                                      layer_data['process_cycles'].sum() + 
                                      layer_data['idle'].sum()),
                    'effective_cycles': int(layer_data['read_cycles'].sum() + 
                                          layer_data['process_cycles'].sum()),
                    
                    # Efficiency metrics (percentages)
                    'compute_efficiency': (layer_data['process_cycles'].sum() / 
                                         max(1, layer_data['read_cycles'].sum() + 
                                             layer_data['process_cycles'].sum() + 
                                             layer_data['idle'].sum())) * 100,
                    
                    'memory_efficiency': (layer_data['read_cycles'].sum() / 
                                        max(1, layer_data['read_cycles'].sum() + 
                                            layer_data['wstall'].sum())) * 100,
                    
                    # Performance metrics
                    'gmacs_per_cycle': layer_data['gmacs'].sum() / max(1, layer_data['process_cycles'].sum()),
                    
                    # Cost estimation for optimization algorithms
                    'execution_cost': int(layer_data['read_cycles'].sum() * 1.0 + 
                                        layer_data['process_cycles'].sum() * 2.0 + 
                                        layer_data['idle'].sum() * 0.5),
                    
                    'memory_cost': int(layer_data['inputbuf_p'].max() + 
                                     layer_data['weightbuf_p'].max()),
                    
                    'communication_cost': int(layer_data['read_cycles'].sum() * 0.1)
                }
                
        except Exception as e:
            print(f"Error parsing SystemC profiling: {e}")
            
        return layer_metrics

    def profile_model_per_layer(self, model_file: str, delegate_type: str = "sa_sim", 
                               output_model_name: str = None, csv_suffix: str = "") -> Dict:
        """Profile a model and return per-layer metrics for partitioning algorithms"""
        
        # Use output_model_name for file naming, or derive from model_file
        if output_model_name:
            model_name = output_model_name
        else:
            model_name = model_file.replace('.tflite', '')
            
        print(f"DEBUG: Starting profile_model_per_layer for {model_file} -> {model_name}")
        
        model_path = f"{self.models_dir}/{model_file}"
        
        # Handle different delegate types
        if delegate_type is None or delegate_type == 'baseline':
            # No delegate - use baseline benchmark model
            binary_path = f"{self.workspace_dir}/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model_plus_flex"
            delegate_flag = ""
        else:
            binary_path = f"{self.workspace_dir}/bazel-bin/tensorflow/lite/delegates/utils/{delegate_type}_delegate/label_image_plus_{delegate_type}_delegate"
            delegate_flag = f"--use_{delegate_type}_delegate=true"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        if not os.path.exists(binary_path):
            raise FileNotFoundError(f"Binary not found: {binary_path}")
        
        print(f"DEBUG: Using binary: {binary_path}")
        print(f"DEBUG: Delegate type: {delegate_type}")
        
        # Use unique CSV output file to avoid conflicts between different modes
        csv_filename = f"sa_sim{csv_suffix}.csv" if csv_suffix else "sa_sim.csv"
        csv_output = f"{self.outputs_dir}/{csv_filename}"
        print(f"DEBUG: Using CSV output: {csv_output}")
        
        # Clean previous profiling data for this specific CSV
        if os.path.exists(csv_output):
            os.remove(csv_output)
            
        # Run inference with appropriate configuration
        if delegate_type is None or delegate_type == 'baseline':
            # Baseline mode - use benchmark_model
            cmd = [
                binary_path,
                f"--graph={model_path}",
                "--num_runs=1",
                "--enable_op_profiling=true"
            ]
        else:
            # Delegate mode - use label_image
            cmd = [
                binary_path,
                f"--tflite_model={model_path}",
                "--image=test_images/grace_hopper.bmp",
                "--labels=tensorflow/lite/examples/ios/camera/data/labels.txt",
                delegate_flag,
                "-p", "1",
                "--verbose", "1"
            ]
            
            # Add BPSO config if this is a BPSO run
            if csv_suffix == "_bpso" and os.path.exists(f"{self.outputs_dir}/bpso_partition_config.csv"):
                cmd.append(f"--bpso_partition_config={self.outputs_dir}/bpso_partition_config.csv")
                print(f"DEBUG: Added BPSO config to command")
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, cwd=self.workspace_dir)
            
            # Extract number of delegated layers from output
            num_delegated_layers = 0
            output_text = result.stdout + result.stderr
            for line in output_text.split('\n'):
                if "nodes delegated out of" in line:
                    # Extract: "INFO: SASimDelegate delegate: 15 nodes delegated out of 39 nodes"
                    parts = line.split("nodes delegated out of")
                    if len(parts) > 0:
                        try:
                            num_delegated_layers = int(parts[0].split()[-1])
                            print(f"DEBUG: Found {num_delegated_layers} delegated layers")
                            break
                        except ValueError:
                            pass
            
            if num_delegated_layers == 0:
                print("DEBUG: No delegated layers found, using default estimate of 15")
                num_delegated_layers = 15
            
            # Get layer information
            layers = self.get_layer_info_from_tflite_verbose(model_path, binary_path)
            
            # Parse SystemC profiling data with correct number of delegated layers
            layer_metrics = self.parse_systemc_profiling_per_layer(csv_output, num_delegated_layers)
            
            # Combine layer info with profiling metrics
            profile_data = {
                'model_name': model_name,
                'delegate_type': delegate_type,
                'total_layers': len(layers),
                'delegated_layers': num_delegated_layers,
                'layers': {}
            }
            
            # Map delegated layers to TFLite layers
            delegated_layer_idx = 0
            
            for i, layer in enumerate(layers):
                layer_name = f"layer_{i}"
                
                # Check if this layer is delegated (Conv2D layers are typically delegated)
                is_delegated = 'CONV_2D' in layer.get('op_type', '')
                
                if is_delegated and delegated_layer_idx < len(layer_metrics):
                    # Use SystemC profiling data for delegated layers
                    delegated_name = f"delegated_layer_{delegated_layer_idx}"
                    layer_metrics_data = layer_metrics.get(delegated_name, {})
                    # Update layer info to reflect delegation status
                    layer['is_delegated'] = True
                    delegated_layer_idx += 1
                else:
                    # Provide baseline CPU estimates for non-delegated layers
                    layer_metrics_data = self.estimate_cpu_baseline(layer)
                    layer['is_delegated'] = False
                
                profile_data['layers'][layer_name] = {
                    'layer_id': i,  # Add layer_id here
                    'layer_info': layer,
                    'is_delegated': is_delegated,
                    'performance_metrics': layer_metrics_data,
                    'partitioning_metrics': self.calculate_partitioning_metrics(layer, layer_metrics_data)
                }
                
            return profile_data
            
        except Exception as e:
            print(f"Error profiling model: {e}")
            return {}

    def calculate_partitioning_metrics(self, layer_info: Dict, perf_metrics: Dict) -> Dict:
        """Calculate specific metrics useful for partitioning algorithms using SystemC data"""
        
        # Extract SystemC profiling data (use actual column names from SystemC)
        total_cycles = perf_metrics.get('total_cycles', 0)
        process_cycles = perf_metrics.get('process_cycles', 0)
        read_cycles = perf_metrics.get('read_cycles', 0)
        memory_access_cycles = read_cycles + perf_metrics.get('wstall_cycles', 0)
        
        # Partitioning-specific metrics for PBSO algorithms
        partitioning_metrics = {
            # Core cycle metrics (what PBSO needs for cost functions)
            'total_cycles': total_cycles,
            'compute_cycles': process_cycles,  # Use process_cycles as compute
            'memory_access_cycles': memory_access_cycles,
            'data_movement_cycles': read_cycles,  # Data movement includes read operations
            
            # Compute intensity (compute/memory ratio) - critical for device selection
            'compute_intensity': process_cycles / max(memory_access_cycles, 1),
            
            # Operation complexity (for scheduling priority)
            'operation_weight': total_cycles * (1 + layer_info.get('params', 0) / 1000),
            
            # Resource requirements (from SystemC buffer data)
            'buffer_requirement': perf_metrics.get('max_input_buffer', 0) + perf_metrics.get('max_weight_buffer', 0),
            'energy_cost': perf_metrics.get('execution_cost', 0),  # Use execution_cost as energy proxy
            
            # Layer characteristics for partitioning decisions
            'layer_type': layer_info.get('layer_type', 'UNKNOWN'),
            'is_compute_intensive': process_cycles > memory_access_cycles,
            'is_memory_intensive': memory_access_cycles > process_cycles,
            
            # Communication cost (for edge partitioning) - use SystemC communication cost if available
            'communication_cost': perf_metrics.get('communication_cost', self.estimate_communication_cost(layer_info)),
            
            # Parallelization potential based on operation type
            'parallelization_factor': self.estimate_parallelization_factor(layer_info),
            
            # Additional SystemC-specific metrics for advanced partitioning
            'gmacs_per_cycle': perf_metrics.get('gmacs_per_cycle', 0),
            'compute_efficiency': perf_metrics.get('compute_efficiency', 0),
            'memory_efficiency': perf_metrics.get('memory_efficiency', 0),
            'total_gmacs': perf_metrics.get('total_gmacs', 0),
            
            # Cost breakdown for multi-objective optimization
            'execution_cost_breakdown': {
                'compute_cost': process_cycles * 2.0,  # Higher weight for compute
                'memory_cost': perf_metrics.get('memory_cost', 0),
                'communication_cost': perf_metrics.get('communication_cost', 0)
            }
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
        
        delegates = ['sa_sim']
        
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
                
                # Layer characteristics for partitioning algorithms
                'memory_bound': 1 if metrics.get('memory_accesses', 500) > metrics.get('compute_cycles', 1000) else 0,
                'compute_bound': 1 if metrics.get('compute_cycles', 1000) > metrics.get('memory_accesses', 500) else 0,
                
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

    def estimate_cpu_baseline(self, layer_info: Dict) -> Dict:
        """Estimate CPU baseline performance for non-delegated layers with realistic values"""
        
        op_type = layer_info.get('op_type', 'UNKNOWN')
        
        # Realistic CPU baseline estimates - much higher to match delegate SystemC data scale
        # These values are scaled to be comparable with SystemC simulation results
        cpu_baselines = {
            'QUANTIZE': {'cycles': 50000, 'memory': 2048, 'compute': 25000},
            'DEQUANTIZE': {'cycles': 60000, 'memory': 2048, 'compute': 30000},
            'PAD': {'cycles': 25000, 'memory': 1024, 'compute': 5000},
            'RELU': {'cycles': 40000, 'memory': 1024, 'compute': 20000},
            'RELU6': {'cycles': 45000, 'memory': 1024, 'compute': 22500},
            'ADD': {'cycles': 30000, 'memory': 2048, 'compute': 15000},
            'MUL': {'cycles': 35000, 'memory': 2048, 'compute': 17500},
            'RESHAPE': {'cycles': 20000, 'memory': 512, 'compute': 2500},
            'POOLING': {'cycles': 100000, 'memory': 4096, 'compute': 50000},
            'AVERAGE_POOL_2D': {'cycles': 125000, 'memory': 4096, 'compute': 62500},
            'MAX_POOL_2D': {'cycles': 100000, 'memory': 4096, 'compute': 50000},
            'SOFTMAX': {'cycles': 250000, 'memory': 8192, 'compute': 200000},
            'FULLY_CONNECTED': {'cycles': 500000, 'memory': 16384, 'compute': 400000},
            'DEPTHWISE_CONV_2D': {'cycles': 1000000, 'memory': 32768, 'compute': 750000},
            'CONV_2D': {'cycles': 2500000, 'memory': 65536, 'compute': 2000000}  # Expensive on CPU
        }
        
        baseline = cpu_baselines.get(op_type, {'cycles': 50000, 'memory': 2048, 'compute': 25000})
        
        return {
            'read_cycles': int(baseline['cycles'] * 0.3),
            'process_cycles': int(baseline['cycles'] * 0.6),
            'idle_cycles': int(baseline['cycles'] * 0.1),
            'gemmw_cycles': 0,  # No GEMM on CPU
            'gemm_cycles': 0,   # No GEMM on CPU
            'wstall_cycles': int(baseline['cycles'] * 0.05),
            'max_input_buffer': baseline['memory'],
            'max_weight_buffer': baseline['memory'] // 2,
            'total_gmacs': baseline['compute'] // 10,
            'total_outputs': baseline['compute'] // 20,
            'total_cycles': baseline['cycles'],
            'effective_cycles': int(baseline['cycles'] * 0.9),
            'compute_efficiency': 60.0,  # CPU efficiency
            'memory_efficiency': 80.0,   # CPU memory efficiency
            'gmacs_per_cycle': (baseline['compute'] // 10) / max(1, baseline['cycles']),
            'execution_cost': int(baseline['cycles'] * 2.0),  # CPU is less efficient, higher weight
            'memory_cost': baseline['memory'],
            'communication_cost': int(baseline['cycles'] * 0.05)
        }

def main():
    """Main function for per-layer profiling"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Per-layer profiling for SECDA-Lite models")
    parser.add_argument("--model", default="mobilenetv1.tflite", 
                       help="Model file to profile (relative to models dir)")
    parser.add_argument("--delegate", default="sa_sim",
                       help="Delegate type: 'sa_sim', 'baseline', or 'none'")
    parser.add_argument("--model_name", default=None,
                       help="Output model name (used for file naming)")
    parser.add_argument("--output_dir", default="outputs",
                       help="Output directory for results")
    parser.add_argument("--bpso_config", default=None,
                       help="Path to BPSO partition config file")
    
    args = parser.parse_args()
    
    # Use model name from args or derive from model file
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = args.model.replace('.tflite', '')
    
    print(f"=== Per-layer profiling ===")
    print(f"Model: {args.model}")
    print(f"Delegate: {args.delegate}")
    print(f"Model name: {model_name}")
    print(f"BPSO config: {args.bpso_config}")
    
    profiler = LayerProfiler()
    
    # Set delegate mode for profiling
    delegate_type = args.delegate
    if delegate_type == 'none' or delegate_type == 'baseline':
        delegate_type = None  # No delegation for baseline
    
    try:
        profile_data = profiler.profile_model_per_layer(args.model, delegate_type, model_name)
        if profile_data:
            report_file = profiler.generate_partitioning_report(profile_data)
            
            # Generate PBSO DAG data for optimization
            pbso_df = profiler.extract_pbso_dag_data(profile_data)
            
            # Print summary for verification
            print(f"\nSummary for {delegate_type or 'baseline'} profiling:")
            print(f"Model: {profile_data['model_name']}")
            print(f"Total layers: {profile_data['total_layers']}")
            print(f"PBSO DAG data shape: {pbso_df.shape}")
            
            print(f"\nFirst 5 layers:")
            for _, row in pbso_df.head().iterrows():
                print(f"  {row['layer_name']}: {row['layer_type']} - CPU:{row['cpu_cycles']} SA:{row['sa_accelerator_cycles']} cycles")
                
            # Show key metrics for partitioning
            print(f"\nKey metrics for partitioning:")
            print(f"  Total CPU cycles: {pbso_df['cpu_cycles'].sum():,}")
            print(f"  Total SA Accelerator cycles: {pbso_df['sa_accelerator_cycles'].sum():,}")
            print(f"  Memory bound layers: {pbso_df['memory_bound'].sum()}")
            print(f"  Compute bound layers: {pbso_df['compute_bound'].sum()}")
            
            print(f"\nProfile saved to: {report_file}")
                
    except Exception as e:
        print(f"Error in profiling: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
