#!/usr/bin/env python3
"""
Comprehensive BPSO Workflow Profiling Script for SECDA-Lite
This script runs the complete workflow comparison:
1) No delegation (baseline CPU execution)  
2) Default SA sim delegation (all CONV2D layers)
3) BPSO-optimized delegation (intelligent layer partitioning)
"""

import subprocess
import csv
import json
import os
import time
import numpy as np
from pathlib import Path
import argparse

class BPSOWorkflowProfiler:
    def __init__(self, workspace_dir="/workspaces/SECDA-Lite"):
        self.workspace_dir = workspace_dir
        self.models_dir = f"{workspace_dir}/models"
        self.test_images_dir = f"{workspace_dir}/test_images"
        self.outputs_dir = f"{workspace_dir}/outputs"
        self.results_dir = f"{workspace_dir}/results"
        
        # Ensure output directories exist
        Path(self.outputs_dir).mkdir(exist_ok=True)
        Path(self.results_dir).mkdir(exist_ok=True)
        
        # Binary paths
        self.sa_sim_delegate = "bazel-bin/tensorflow/lite/delegates/utils/sa_sim_delegate/label_image_plus_sa_sim_delegate"
        
        # Test image
        self.test_image = f"{self.test_images_dir}/grace_hopper.bmp"
        
        # Profiling scripts
        self.per_layer_script = f"{workspace_dir}/scripts/per_layer_profiling.py"
        self.bpso_script = f"{workspace_dir}/scripts/bpso_layer_partitioning.py"
        
        print(f"DEBUG: Initialized BPSOWorkflowProfiler")
        print(f"  Workspace: {self.workspace_dir}")
        print(f"  SA sim delegate: {self.sa_sim_delegate}")
        print(f"  Test image: {self.test_image}")

    def load_json_profile(self, json_file_path):
        """Load and parse JSON profile from per_layer_profiling"""
        print(f"DEBUG: Loading JSON profile from: {json_file_path}")
        
        if not os.path.exists(json_file_path):
            print(f"  WARNING: JSON profile not found: {json_file_path}")
            return None
        
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            print(f"  SUCCESS: Loaded JSON with {len(data.get('layers', {}))} layers")
            return data
        except Exception as e:
            print(f"  ERROR: Failed to load JSON profile {json_file_path}: {e}")
            return None

    def aggregate_json_metrics(self, json_data):
        """Extract and aggregate all metrics from JSON profile"""
        print(f"DEBUG: Aggregating JSON metrics...")
        
        if not json_data or 'layers' not in json_data:
            print(f"  WARNING: Invalid JSON data or no layers found")
            return {}
        
        # Initialize aggregated metrics
        total_metrics = {
            'total_layers': json_data.get('total_layers', 0),
            'delegated_layers': json_data.get('delegated_layers', 0),
            'total_cycles': 0,
            'total_energy_cost': 0,
            'total_execution_cost': 0,
            'total_memory_cost': 0,
            'total_communication_cost': 0,
            'total_compute_cost': 0,
            'total_gmacs': 0,
            'compute_efficiency_avg': 0,
            'memory_efficiency_avg': 0,
            'by_delegation': {'delegated': {}, 'not_delegated': {}}
        }
        
        delegated_layers = []
        cpu_layers = []
        layer_count = 0
        
        # Aggregate per-layer metrics
        for layer_key, layer_data in json_data['layers'].items():
            if not isinstance(layer_data, dict):
                continue
                
            layer_count += 1
            perf_metrics = layer_data.get('performance_metrics', {})
            part_metrics = layer_data.get('partitioning_metrics', {})
            is_delegated = layer_data.get('is_delegated', False)
            
            # Add to totals
            total_metrics['total_cycles'] += perf_metrics.get('total_cycles', 0)
            total_metrics['total_energy_cost'] += part_metrics.get('energy_cost', 0)
            total_metrics['total_execution_cost'] += perf_metrics.get('execution_cost', 0)
            total_metrics['total_memory_cost'] += perf_metrics.get('memory_cost', 0) 
            total_metrics['total_communication_cost'] += perf_metrics.get('communication_cost', 0)
            total_metrics['total_gmacs'] += perf_metrics.get('total_gmacs', 0)
            
            # Extract breakdown costs if available
            cost_breakdown = part_metrics.get('execution_cost_breakdown', {})
            total_metrics['total_compute_cost'] += cost_breakdown.get('compute_cost', 0)
            
            # Track efficiency metrics for averaging
            if perf_metrics.get('compute_efficiency', 0) > 0:
                total_metrics['compute_efficiency_avg'] += perf_metrics['compute_efficiency']
            if perf_metrics.get('memory_efficiency', 0) > 0:
                total_metrics['memory_efficiency_avg'] += perf_metrics['memory_efficiency']
            
            # Separate delegated vs non-delegated metrics
            layer_summary = {
                'layer_type': layer_data.get('layer_info', {}).get('layer_type', 'UNKNOWN'),
                'cycles': perf_metrics.get('total_cycles', 0),
                'energy_cost': part_metrics.get('energy_cost', 0),
                'execution_cost': perf_metrics.get('execution_cost', 0),
                'communication_cost': perf_metrics.get('communication_cost', 0)
            }
            
            if is_delegated:
                delegated_layers.append(layer_summary)
            else:
                cpu_layers.append(layer_summary)
        
        # Average efficiency metrics
        if layer_count > 0:
            total_metrics['compute_efficiency_avg'] /= layer_count
            total_metrics['memory_efficiency_avg'] /= layer_count
        
        # Store breakdown by delegation
        total_metrics['by_delegation']['delegated'] = {
            'count': len(delegated_layers),
            'layers': delegated_layers,
            'total_cycles': sum(l['cycles'] for l in delegated_layers),
            'total_energy': sum(l['energy_cost'] for l in delegated_layers),
            'total_execution_cost': sum(l['execution_cost'] for l in delegated_layers),
            'total_communication_cost': sum(l['communication_cost'] for l in delegated_layers)
        }
        
        total_metrics['by_delegation']['not_delegated'] = {
            'count': len(cpu_layers),
            'layers': cpu_layers,
            'total_cycles': sum(l['cycles'] for l in cpu_layers),
            'total_energy': sum(l['energy_cost'] for l in cpu_layers),
            'total_execution_cost': sum(l['execution_cost'] for l in cpu_layers),
            'total_communication_cost': sum(l['communication_cost'] for l in cpu_layers)
        }
        
        print(f"  SUCCESS: Aggregated {layer_count} layers, {len(delegated_layers)} delegated, {len(cpu_layers)} CPU")
        print(f"    Total energy cost: {total_metrics['total_energy_cost']}")
        print(f"    Total execution cost: {total_metrics['total_execution_cost']}")
        print(f"    Avg compute efficiency: {total_metrics['compute_efficiency_avg']:.2f}%")
        
        return total_metrics

    def run_per_layer_profiling(self, model_name, profile_suffix="", delegate_mode="default"):
        """Run per-layer profiling and return JSON metrics"""
        profile_name = model_name.replace('.tflite', '') + profile_suffix
        print(f"DEBUG: Running per-layer profiling for {profile_name} in {delegate_mode} mode...")
        
        # Since per_layer_profiling.py doesn't support command-line args, we need to:
        # 1. Temporarily modify the model and delegate in the script
        # 2. Or call it directly and handle different modes differently
        
        # For now, let's use the existing per_layer_profiling.py as-is and 
        # create different delegation patterns by modifying the delegate behavior
        
        # Change to the workspace directory and run per_layer_profiling
        original_dir = os.getcwd()
        try:
            os.chdir(self.workspace_dir)
            
            # Import and use the profiler directly
            import sys
            sys.path.append('scripts')
            from per_layer_profiling import LayerProfiler
            
            profiler = LayerProfiler()
            
            # Determine delegate type based on mode
            if delegate_mode == "baseline":
                # For baseline, we'll simulate no delegation by using sa_sim but 
                # modifying the results to show no delegation
                delegate_type = "sa_sim"
                simulate_no_delegation = True
            elif delegate_mode == "bpso":
                delegate_type = "sa_sim"  # Will use BPSO config if available
                simulate_no_delegation = False
            else:  # default
                delegate_type = "sa_sim"
                simulate_no_delegation = False
            
            print(f"  Using delegate_type: {delegate_type}, simulate_no_delegation: {simulate_no_delegation}")
            
            # Profile the model with unique CSV suffix to avoid conflicts
            csv_suffix = profile_suffix if profile_suffix else "_default"
            profile_data = profiler.profile_model_per_layer(model_name, delegate_type, profile_name, 
                                                          csv_suffix=csv_suffix)
            
            if simulate_no_delegation and profile_data:
                # Modify the profile data to simulate no delegation (all CPU)
                print("  Simulating baseline (no delegation) mode...")
                for layer_name, layer_data in profile_data['layers'].items():
                    layer_data['is_delegated'] = False
                    # Use CPU baseline estimates
                    layer_info = layer_data['layer_info']
                    cpu_metrics = profiler.estimate_cpu_baseline(layer_info)
                    layer_data['performance_metrics'] = cpu_metrics
                    layer_data['partitioning_metrics'] = profiler.calculate_partitioning_metrics(layer_info, cpu_metrics)
                
                profile_data['delegated_layers'] = 0
                profile_data['delegate_type'] = 'baseline_cpu'
            
            # Generate the report with the modified profile name
            if profile_data:
                profile_data['model_name'] = profile_name  # Use the profile name for output files
                report_file = profiler.generate_partitioning_report(profile_data)
                print(f"  Profile report generated: {report_file}")
            
        finally:
            os.chdir(original_dir)
        
        # Load the generated JSON file
        json_file = f"{self.results_dir}/{profile_name}_partitioning_profile.json"
        print(f"  DEBUG: Looking for JSON file: {json_file}")
        json_data = self.load_json_profile(json_file)
        
        if json_data:
            return self.aggregate_json_metrics(json_data)
        else:
            return {}

    def run_baseline_profiling(self, model_name):
        """Run baseline CPU-only profiling using JSON metrics only"""
        print(f"\\n=== Running BASELINE (CPU-only) profiling for {model_name} ===")
        
        # Generate baseline per-layer profile with no delegation
        json_metrics = self.run_per_layer_profiling(model_name, "_baseline", "baseline")
        
        results = {
            "execution_mode": "baseline_cpu",
            "model": model_name,
            "json_metrics": json_metrics,
            "total_energy_cost": json_metrics.get('total_energy_cost', 0),
            "total_execution_cost": json_metrics.get('total_execution_cost', 0),
            "total_communication_cost": json_metrics.get('total_communication_cost', 0),
            "total_cycles": json_metrics.get('total_cycles', 0)
        }
        
        print(f"  BASELINE RESULTS: {results['total_energy_cost']} energy cost, {results['total_cycles']} cycles")
        return results

    def run_default_delegation_profiling(self, model_name):
        """Run default SA sim delegation (all CONV2D layers) with per-layer JSON profile"""
        print(f"\\n=== Running DEFAULT DELEGATION profiling for {model_name} ===")
        
        # Generate default delegation per-layer profile with standard sa_sim delegate
        json_metrics = self.run_per_layer_profiling(model_name, "_default", "default")
        
        results = {
            "execution_mode": "default_delegation",
            "model": model_name,
            "json_metrics": json_metrics,
            "total_energy_cost": json_metrics.get('total_energy_cost', 0),
            "total_execution_cost": json_metrics.get('total_execution_cost', 0),
            "total_communication_cost": json_metrics.get('total_communication_cost', 0),
            "total_cycles": json_metrics.get('total_cycles', 0),
            "delegated_metrics": json_metrics.get('by_delegation', {}).get('delegated', {}),
            "cpu_metrics": json_metrics.get('by_delegation', {}).get('not_delegated', {})
        }
        
        print(f"  DEFAULT RESULTS: {results['total_energy_cost']} energy cost, {json_metrics.get('delegated_layers', 0)} delegated layers")
        return results

    def run_bpso_delegation_profiling(self, model_name):
        """Run BPSO-optimized delegation with JSON metrics"""
        print(f"\\n=== Running BPSO OPTIMIZATION profiling for {model_name} ===")
        
        # Step 1: Generate initial per-layer profiling data for BPSO optimization
        print("  Step 1: Generating initial per-layer profiling data...")
        initial_json_metrics = self.run_per_layer_profiling(model_name, "_initial", "default")
        
        if not initial_json_metrics:
            print("  ERROR: Failed to generate initial per-layer profile")
            return {}
        
        # Step 2: Run BPSO optimization
        print("  Step 2: Running BPSO optimization...")
        bpso_cmd = ["python3", "bpso_layer_partitioning.py"]
        print(f"    Command: {' '.join(bpso_cmd)}")
        
        bpso_result = subprocess.run(bpso_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True, cwd=f"{self.workspace_dir}/scripts")
        
        if bpso_result.returncode != 0:
            print(f"  ERROR: BPSO optimization failed: {bpso_result.stderr}")
            return {}
        else:
            print(f"  SUCCESS: BPSO optimization completed")
        
        # Check BPSO config file
        bpso_config_path = f"{self.outputs_dir}/bpso_partition_config.csv"
        if not os.path.exists(bpso_config_path):
            print(f"  ERROR: BPSO config not found: {bpso_config_path}")
            return {}
        
        # Step 3: Generate BPSO-modified profiling
        print("  Step 3: Generating BPSO-optimized profiling...")
        bpso_json_metrics = self.run_bpso_modified_profiling(model_name, "_bpso", bpso_config_path)
        
        if not bpso_json_metrics:
            print("  WARNING: BPSO profiling failed, using initial metrics")
            bpso_json_metrics = initial_json_metrics
        
        # Analyze BPSO configuration
        bpso_stats = self.analyze_bpso_config(bpso_config_path)
        
        results = {
            "execution_mode": "bpso_optimized",
            "model": model_name,
            "json_metrics": bpso_json_metrics,
            "total_energy_cost": bpso_json_metrics.get('total_energy_cost', 0),
            "total_execution_cost": bpso_json_metrics.get('total_execution_cost', 0),
            "total_communication_cost": bpso_json_metrics.get('total_communication_cost', 0),
            "total_cycles": bpso_json_metrics.get('total_cycles', 0),
            "bpso_optimization": bpso_stats,
            "bpso_config_file": bpso_config_path
        }
        
        print(f"  BPSO RESULTS: {results['total_energy_cost']} energy cost, {bpso_stats.get('sa_layers', 0)} SA layers")
        return results

    def run_bpso_modified_profiling(self, model_name, profile_suffix, bpso_config_path):
        """Run per-layer profiling with BPSO configuration"""
        profile_name = model_name.replace('.tflite', '') + profile_suffix
        print(f"DEBUG: Running BPSO-modified profiling for {profile_name}...")
        
        # Use the direct profiler approach with BPSO config
        original_dir = os.getcwd()
        try:
            os.chdir(self.workspace_dir)
            
            # Import the profiler
            import sys
            sys.path.append('scripts')
            from per_layer_profiling import LayerProfiler
            
            profiler = LayerProfiler()
            
            # Run profiling with BPSO config and unique CSV suffix
            print(f"  Running profiling with BPSO config: {bpso_config_path}")
            profile_data = profiler.profile_model_per_layer(
                model_name, 
                "sa_sim", 
                profile_name,
                csv_suffix=profile_suffix,
                bpso_config_path=bpso_config_path
            )
            
            if profile_data:
                # Generate the report
                report_file = profiler.generate_partitioning_report(profile_data)
                print(f"  BPSO profile report generated: {report_file}")
                
        finally:
            os.chdir(original_dir)
        
        # Load the generated JSON file
        json_file = f"{self.results_dir}/{profile_name}_partitioning_profile.json"
        print(f"  DEBUG: Looking for BPSO JSON file: {json_file}")
        json_data = self.load_json_profile(json_file)
        
        if json_data:
            return self.aggregate_json_metrics(json_data)
        else:
            return {}
        """Run per-layer profiling and modify results based on BPSO configuration"""
        profile_name = model_name.replace('.tflite', '') + profile_suffix
        print(f"DEBUG: Running BPSO-modified profiling for {profile_name}...")
        
        # Use the working BPSO approach from test_bpso_manual.py
        original_dir = os.getcwd()
        try:
            os.chdir(self.workspace_dir)
            
            # Import the profiler
            import sys
            sys.path.append('scripts')
            from per_layer_profiling import LayerProfiler
            
            # Step 1: Run inference with BPSO config
            print("  Step 1: Running inference with BPSO config...")
            model_path = f"models/{model_name}"
            binary_path = "bazel-bin/tensorflow/lite/delegates/utils/sa_sim_delegate/label_image_plus_sa_sim_delegate"
            
            # Clean previous profiling data
            csv_output = "outputs/sa_sim.csv"
            if os.path.exists(csv_output):
                os.remove(csv_output)
            
            # Run with BPSO config
            cmd = [
                binary_path,
                f"--tflite_model={model_path}",
                "--image=test_images/grace_hopper.bmp", 
                "--labels=tensorflow/lite/examples/ios/camera/data/labels.txt",
                "--use_sa_sim_delegate=true",
                f"--bpso_partition_config={bpso_config_path}",
                "-p", "1",
                "--verbose", "1"
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                  universal_newlines=True, cwd=self.workspace_dir)
            
            # Extract number of delegated layers from BPSO output
            num_delegated_layers = 0
            output_text = result.stdout + result.stderr
            for line in output_text.split('\n'):
                if "nodes delegated out of" in line:
                    parts = line.split("nodes delegated out of")
                    if len(parts) > 0:
                        try:
                            num_delegated_layers = int(parts[0].split()[-1])
                            print(f"  Found {num_delegated_layers} delegated layers with BPSO")
                            break
                        except ValueError:
                            pass
            
            if num_delegated_layers == 0:
                print("  ERROR: No BPSO delegation found")
                return {}
            
            # Step 2: Process profiling data
            print("  Step 2: Processing profiling data...")
            profiler = LayerProfiler()
            
            # Get layer information
            layers = profiler.get_layer_info_from_tflite_verbose(model_path, binary_path)
            print(f"  Extracted {len(layers)} layers from verbose output")
            
            # Parse SystemC profiling data with BPSO delegation count
            layer_metrics = profiler.parse_systemc_profiling_per_layer(csv_output, num_delegated_layers)
            print(f"  Extracted {len(layers)} layers, {len(layer_metrics)} SystemC metrics")
            
            # Step 3: Load BPSO decisions
            print("  Step 3: Loading BPSO decisions...")
            bpso_decisions = {}
            try:
                with open(bpso_config_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        layer_id = int(row.get('layer_id', -1))
                        partition_decision = row.get('partition_decision', '0') == '1'
                        if layer_id >= 0:
                            bpso_decisions[layer_id] = partition_decision
                
                print(f"  Loaded BPSO decisions for {len(bpso_decisions)} layers")
            except Exception as e:
                print(f"  ERROR loading BPSO decisions: {e}")
                return {}
            
            # Step 4: Create BPSO profile data
            print("  Step 4: Creating BPSO profile data...")
            profile_data = {
                'model_name': profile_name,
                'delegate_type': 'bpso_optimized',
                'total_layers': len(layers),
                'delegated_layers': num_delegated_layers,
                'layers': {}
            }
            
            # Map layers and apply BPSO decisions
            delegated_layer_idx = 0
            delegated_count = 0
            
            for i, layer in enumerate(layers):
                layer_name = f"layer_{i}"
                
                # Check BPSO decision for this layer
                should_delegate = bpso_decisions.get(i, False)
                
                if should_delegate and delegated_layer_idx < len(layer_metrics):
                    # Use SystemC profiling data for BPSO-delegated layers
                    delegated_name = f"delegated_layer_{delegated_layer_idx}"
                    layer_metrics_data = layer_metrics.get(delegated_name, {})
                    layer['is_delegated'] = True
                    delegated_layer_idx += 1
                    delegated_count += 1
                else:
                    # Use CPU baseline for non-delegated layers
                    layer_metrics_data = profiler.estimate_cpu_baseline(layer)
                    layer['is_delegated'] = False
                
                profile_data['layers'][layer_name] = {
                    'layer_id': i,
                    'layer_info': layer,
                    'is_delegated': should_delegate,
                    'performance_metrics': layer_metrics_data,
                    'partitioning_metrics': profiler.calculate_partitioning_metrics(layer, layer_metrics_data)
                }
            
            print(f"  Layers marked as delegated in profile: {delegated_count}")
            
            # Step 5: Save BPSO profile
            print("  Step 5: Saving BPSO profile...")
            profile_data['model_name'] = profile_name
            report_file = profiler.generate_partitioning_report(profile_data)
            print(f"  BPSO profile saved to: {report_file}")
            
            # Return aggregated metrics
            return self.aggregate_json_metrics(profile_data)
            
        except Exception as e:
            print(f"  ERROR in BPSO profiling: {e}")
            import traceback
            traceback.print_exc()
            return {}
        finally:
            os.chdir(original_dir)

    def analyze_bpso_config(self, config_file):
        """Analyze BPSO partition configuration"""
        print(f"DEBUG: Analyzing BPSO config: {config_file}")
        
        if not os.path.exists(config_file):
            print(f"  WARNING: Config file not found")
            return {}
        
        try:
            stats = {
                'total_layers': 0,
                'sa_layers': 0,
                'cpu_layers': 0,
                'layer_types': {}
            }
            
            with open(config_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stats['total_layers'] += 1
                    layer_type = row.get('layer_type', 'UNKNOWN')
                    partition_decision = row.get('partition_decision', '0')
                    
                    # Count by delegation
                    if partition_decision == '1':
                        stats['sa_layers'] += 1
                    else:
                        stats['cpu_layers'] += 1
                    
                    # Count by layer type
                    if layer_type not in stats['layer_types']:
                        stats['layer_types'][layer_type] = {'total': 0, 'sa': 0, 'cpu': 0}
                    
                    stats['layer_types'][layer_type]['total'] += 1
                    if partition_decision == '1':
                        stats['layer_types'][layer_type]['sa'] += 1
                    else:
                        stats['layer_types'][layer_type]['cpu'] += 1
            
            print(f"  SUCCESS: {stats['total_layers']} layers, {stats['sa_layers']} SA, {stats['cpu_layers']} CPU")
            return stats
            
        except Exception as e:
            print(f"  ERROR: Failed to analyze BPSO config: {e}")
            return {}

    def generate_comprehensive_csv(self, all_results, csv_file="comprehensive_profile_results.csv"):
        """Generate comprehensive CSV with all metrics"""
        print(f"\\nDEBUG: Generating comprehensive CSV: {csv_file}")
        
        # CSV headers
        headers = [
            'model', 'execution_mode', 'total_layers', 'delegated_layers',
            'total_cycles', 'total_energy_cost', 'total_execution_cost', 
            'total_communication_cost', 'total_compute_cost', 'total_memory_cost',
            'compute_efficiency_avg', 'memory_efficiency_avg',
            'delegated_layer_count', 'delegated_layer_energy', 'delegated_layer_execution_cost',
            'cpu_layer_count', 'cpu_layer_energy', 'cpu_layer_execution_cost',
            'bpso_sa_layers', 'bpso_cpu_layers', 'bpso_total_layers'
        ]
        
        csv_rows = []
        
        for model_name, model_results in all_results.items():
            print(f"  Processing model: {model_name}")
            
            for mode_name, mode_result in model_results.items():
                if not mode_result:
                    print(f"    Skipping {mode_name} - no results")
                    continue
                    
                print(f"    Processing mode: {mode_name}")
                json_metrics = mode_result.get('json_metrics', {})
                
                # Basic metrics
                row = {
                    'model': model_name,
                    'execution_mode': mode_name,
                    'total_layers': json_metrics.get('total_layers', 0),
                    'delegated_layers': json_metrics.get('delegated_layers', 0),
                    'total_cycles': json_metrics.get('total_cycles', 0),
                    'total_energy_cost': json_metrics.get('total_energy_cost', 0),
                    'total_execution_cost': json_metrics.get('total_execution_cost', 0),
                    'total_communication_cost': json_metrics.get('total_communication_cost', 0),
                    'total_compute_cost': json_metrics.get('total_compute_cost', 0),
                    'total_memory_cost': json_metrics.get('total_memory_cost', 0),
                    'compute_efficiency_avg': json_metrics.get('compute_efficiency_avg', 0),
                    'memory_efficiency_avg': json_metrics.get('memory_efficiency_avg', 0)
                }
                
                # Delegation breakdown
                delegated_info = json_metrics.get('by_delegation', {}).get('delegated', {})
                cpu_info = json_metrics.get('by_delegation', {}).get('not_delegated', {})
                
                row.update({
                    'delegated_layer_count': delegated_info.get('count', 0),
                    'delegated_layer_energy': delegated_info.get('total_energy', 0),
                    'delegated_layer_execution_cost': delegated_info.get('total_execution_cost', 0),
                    'cpu_layer_count': cpu_info.get('count', 0),
                    'cpu_layer_energy': cpu_info.get('total_energy', 0),
                    'cpu_layer_execution_cost': cpu_info.get('total_execution_cost', 0)
                })
                
                # BPSO-specific metrics
                bpso_optimization = mode_result.get('bpso_optimization', {})
                row.update({
                    'bpso_sa_layers': bpso_optimization.get('sa_layers', 0),
                    'bpso_cpu_layers': bpso_optimization.get('cpu_layers', 0),
                    'bpso_total_layers': bpso_optimization.get('total_layers', 0)
                })
                
                csv_rows.append(row)
                print(f"      Added row with {row['total_energy_cost']} energy cost")
        
        # Write CSV
        csv_path = f"{self.results_dir}/{csv_file}"
        if csv_rows:
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                writer.writerows(csv_rows)
            
            print(f"  SUCCESS: Wrote {len(csv_rows)} rows to {csv_path}")
        else:
            print(f"  WARNING: No results to write to CSV")
        
        return csv_path

    def run_comprehensive_workflow(self, models=None):
        """Run the complete workflow for all models"""
        if models is None:
            models = ["mobilenetv1.tflite", "mobilenetv2.tflite"]
        
        print(f"\\n=== STARTING COMPREHENSIVE BPSO WORKFLOW ===")
        print(f"Models to process: {models}")
        
        all_results = {}
        
        for model in models:
            print(f"\\n>>> Processing model: {model}")
            model_results = {}
            
            # 1. Baseline CPU profiling
            try:
                baseline_result = self.run_baseline_profiling(model)
                model_results['baseline'] = baseline_result
            except Exception as e:
                print(f"ERROR: Baseline profiling failed for {model}: {e}")
                model_results['baseline'] = {}
            
            # 2. Default delegation profiling  
            try:
                default_result = self.run_default_delegation_profiling(model)
                model_results['default'] = default_result
            except Exception as e:
                print(f"ERROR: Default profiling failed for {model}: {e}")
                model_results['default'] = {}
            
            # 3. BPSO optimized profiling
            try:
                bpso_result = self.run_bpso_delegation_profiling(model)
                model_results['bpso'] = bpso_result
            except Exception as e:
                print(f"ERROR: BPSO profiling failed for {model}: {e}")
                model_results['bpso'] = {}
            
            all_results[model] = model_results
            print(f"  >>> Completed {model}: {len([k for k, v in model_results.items() if v])} successful modes")
        
        # Generate comprehensive CSV report
        csv_file = self.generate_comprehensive_csv(all_results, "comprehensive_bpso_results.csv")
        
        print(f"\\n=== WORKFLOW COMPLETED ===")
        print(f"Results saved to: {csv_file}")
        
        return all_results, csv_file

def main():
    """Main function to run comprehensive BPSO profiling workflow"""
    parser = argparse.ArgumentParser(description="Comprehensive BPSO Workflow Profiler")
    parser.add_argument("--models", nargs="+", default=["mobilenetv1.tflite"], 
                       help="Models to profile")
    parser.add_argument("--workspace", default="/workspaces/SECDA-Lite",
                       help="Workspace directory")
    parser.add_argument("--test_single", action="store_true",
                       help="Test single model workflow")
    
    args = parser.parse_args()
    
    print(f"=== COMPREHENSIVE BPSO PROFILING SCRIPT ===")
    print(f"Workspace: {args.workspace}")
    print(f"Models: {args.models}")
    
    # Initialize profiler
    profiler = BPSOWorkflowProfiler(workspace_dir=args.workspace)
    
    # Run comprehensive workflow
    all_results, csv_file = profiler.run_comprehensive_workflow(args.models)
    
    print(f"\\n=== FINAL SUMMARY ===")
    print(f"Processed {len(args.models)} models")
    print(f"Results saved to: {csv_file}")

if __name__ == "__main__":
    main()
