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
import pandas as pd
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
        
        # Test models for comprehensive evaluation
        self.models = [
            "mobilenetv1.tflite",
            "mobilenetv2.tflite"
        ]
        
        # Binary paths - use SA sim delegate for all tests (with/without delegation flag)
        self.sa_sim_delegate = "bazel-bin/tensorflow/lite/delegates/utils/sa_sim_delegate/label_image_plus_sa_sim_delegate"
        
        # Test image
        self.test_image = f"{self.test_images_dir}/grace_hopper.bmp"
        
        # Profiling scripts
        self.per_layer_script = f"{workspace_dir}/scripts/per_layer_profiling.py"
        self.bpso_script = f"{workspace_dir}/scripts/bpso_layer_partitioning.py"

    def load_json_profile(self, json_file_path):
        """Load and parse JSON profile from per_layer_profiling"""
        if not os.path.exists(json_file_path):
            print(f"Warning: JSON profile not found: {json_file_path}")
            return None
        
        try:
            with open(json_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON profile {json_file_path}: {e}")
            return None

    def aggregate_json_metrics(self, json_data):
        """Extract and aggregate all metrics from JSON profile"""
        if not json_data or 'layers' not in json_data:
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
        
        # Aggregate per-layer metrics
        for layer_key, layer_data in json_data['layers'].items():
            if not isinstance(layer_data, dict):
                continue
                
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
        layer_count = len(json_data['layers'])
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
        
    def extract_inference_time(self, output_text):
        """Extract inference time from TFLite output"""
        for line in output_text.split('\n'):
            if "average time:" in line.lower():
                try:
                    # Extract number from "average time: X ms" or "average time: X.X ms"
                    time_str = line.split(":")[-1].strip().replace("ms", "").strip()
                    return float(time_str)
                except:
                    continue
        return 0.0

    def extract_delegate_info(self, output_text):
        """Extract delegation information from verbose output"""
        delegated_nodes = 0
        total_nodes = 0
        partitions = 0
        
        for line in output_text.split('\n'):
            if "nodes delegated out of" in line:
                try:
                    # Parse: "8 nodes delegated out of 39 nodes with 8 partitions"
                    parts = line.split()
                    delegated_nodes = int(parts[0])
                    total_nodes = int(parts[5])
                    partitions = int(parts[8])
                except:
                    continue
        
        return {
            "delegated_nodes": delegated_nodes,
            "total_nodes": total_nodes,
            "partitions": partitions,
            "delegation_ratio": delegated_nodes / total_nodes if total_nodes > 0 else 0
        }

    def run_baseline_profiling(self, model_name, runs=3):
        """Run baseline CPU-only profiling with per-layer JSON profile"""
        print(f"\\n=== Running BASELINE (CPU-only) profiling for {model_name} ===")
        
        # Generate baseline per-layer profile first
        print("  Generating baseline per-layer profile...")
        profile_cmd = [
            "python3", "scripts/per_layer_profiling.py",
            "--model", f"models/{model_name}",
            "--output_dir", "outputs",
            "--model_name", model_name.replace('.tflite', '_baseline')
        ]
        
        profile_result = subprocess.run(profile_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                       universal_newlines=True, cwd=self.workspace_dir)
        if profile_result.returncode != 0:
            print(f"  Warning: Baseline profiling failed: {profile_result.stderr}")
        
    def run_baseline_profiling(self, model_name, runs=1):
        """Run baseline CPU-only profiling using JSON metrics only"""
        print(f"\\n=== Running BASELINE (CPU-only) profiling for {model_name} ===")
        
        # Generate baseline per-layer profile 
        print("  Generating baseline per-layer JSON profile...")
        profile_cmd = [
            "python3", "scripts/per_layer_profiling.py",
            "--model", f"models/{model_name}",
            "--output_dir", "outputs",
            "--model_name", model_name.replace('.tflite', '_baseline')
        ]
        
        profile_result = subprocess.run(profile_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                       universal_newlines=True, cwd=self.workspace_dir)
        if profile_result.returncode != 0:
            print(f"  Error: Baseline profiling failed: {profile_result.stderr}")
            return {}
        
        # Load and aggregate JSON metrics
        json_file = f"{self.workspace_dir}/results/{model_name.replace('.tflite', '_baseline')}_partitioning_profile.json"
        json_data = self.load_json_profile(json_file)
        if not json_data:
            print(f"  Error: Could not load JSON profile: {json_file}")
            return {}
        
        json_metrics = self.aggregate_json_metrics(json_data)
        print(f"  ✓ Loaded baseline metrics: {json_metrics.get('total_layers', 0)} layers, all CPU")
        
        results = {
            "execution_mode": "baseline_cpu",
            "model": model_name,
            "json_data": json_data,
            "json_metrics": json_metrics,
            "summary": {
                "total_layers": json_metrics.get('total_layers', 0),
                "delegated_layers": 0,  # Baseline has no delegation
                "total_cycles": json_metrics.get('total_cycles', 0),
                "total_energy_cost": json_metrics.get('total_energy_cost', 0),
                "total_execution_cost": json_metrics.get('total_execution_cost', 0),
                "total_communication_cost": json_metrics.get('total_communication_cost', 0),
                "total_compute_cost": json_metrics.get('total_compute_cost', 0),
                "total_memory_cost": json_metrics.get('total_memory_cost', 0),
                "compute_efficiency_avg": json_metrics.get('compute_efficiency_avg', 0),
                "memory_efficiency_avg": json_metrics.get('memory_efficiency_avg', 0)
            }
        }
        
        return results
        
        return results

    def run_default_delegation_profiling(self, model_name, runs=1):
        """Run default SA sim delegation using JSON metrics only"""
        print(f"\\n=== Running DEFAULT DELEGATION profiling for {model_name} ===")
        
        # Generate default delegation per-layer profile
        print("  Generating default delegation per-layer profile...")
        profile_cmd = [
            "python3", "scripts/per_layer_profiling.py",
            "--model", f"models/{model_name}",
            "--output_dir", "outputs", 
            "--model_name", model_name.replace('.tflite', '_default')
        ]
        
        profile_result = subprocess.run(profile_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                       universal_newlines=True, cwd=self.workspace_dir)
        if profile_result.returncode != 0:
            print(f"  Error: Default profiling failed: {profile_result.stderr}")
            return {}
        
        # Load and aggregate JSON metrics
        json_file = f"{self.workspace_dir}/results/{model_name.replace('.tflite', '_default')}_partitioning_profile.json"
        json_data = self.load_json_profile(json_file)
        if not json_data:
            print(f"  Error: Could not load JSON profile: {json_file}")
            return {}
        
        json_metrics = self.aggregate_json_metrics(json_data)
        print(f"  ✓ Loaded default metrics: {json_metrics.get('total_layers', 0)} layers, {json_metrics.get('delegated_layers', 0)} delegated")
        
        results = {
            "execution_mode": "default_delegation",
            "model": model_name,
            "json_data": json_data,
            "json_metrics": json_metrics,
            "summary": {
                "total_layers": json_metrics.get('total_layers', 0),
                "delegated_layers": json_metrics.get('delegated_layers', 0),
                "total_cycles": json_metrics.get('total_cycles', 0),
                "total_energy_cost": json_metrics.get('total_energy_cost', 0),
                "total_execution_cost": json_metrics.get('total_execution_cost', 0),
                "total_communication_cost": json_metrics.get('total_communication_cost', 0),
                "total_compute_cost": json_metrics.get('total_compute_cost', 0),
                "total_memory_cost": json_metrics.get('total_memory_cost', 0),
                "compute_efficiency_avg": json_metrics.get('compute_efficiency_avg', 0),
                "memory_efficiency_avg": json_metrics.get('memory_efficiency_avg', 0),
                "delegated_metrics": json_metrics.get('by_delegation', {}).get('delegated', {}),
                "cpu_metrics": json_metrics.get('by_delegation', {}).get('not_delegated', {})
            }
        }
        
        return results

    def run_bpso_delegation_profiling(self, model_name, runs=3):
        """Run BPSO-optimized delegation with JSON metrics"""
        print(f"\\n=== Running BPSO OPTIMIZATION profiling for {model_name} ===")
        
        # Step 1: Generate per-layer profiling data
        print("  Step 1: Generating per-layer profiling data...")
        per_layer_cmd = [
            "python3", self.per_layer_script,
            "--model", f"{self.models_dir}/{model_name}",
            "--output_dir", self.outputs_dir,
            "--model_name", model_name.replace(".tflite", "")
        ]
        
        subprocess.run(per_layer_cmd, cwd=self.workspace_dir)
        
        # Step 2: Run BPSO optimization
        print("  Step 2: Running BPSO optimization...")
        bpso_cmd = ["python3", self.bpso_script]
        subprocess.run(bpso_cmd, cwd=f"{self.workspace_dir}/scripts")
        
        # Load JSON profile with original per-layer data  
        json_file = f"{self.workspace_dir}/results/{model_name.replace('.tflite', '')}_partitioning_profile.json"
        json_metrics = {}
        json_data = self.load_json_profile(json_file)
        if json_data:
            json_metrics = self.aggregate_json_metrics(json_data)
            print(f"  Loaded BPSO JSON metrics: {json_metrics.get('total_layers', 0)} layers")
        
        # Step 3: Profile with BPSO configuration
        print("  Step 3: Profiling with BPSO partition configuration...")
        
        bpso_config_path = f"{self.outputs_dir}/bpso_partition_config.csv"
        
        results = {
            "execution_mode": "bpso_optimized",
            "model": model_name,
            "runs": [],
            "avg_inference_time_ms": 0,
            "std_inference_time_ms": 0,
            "delegation_info": {},
            "bpso_optimization": {},
            "json_metrics": json_metrics,
            "total_energy_cost": json_metrics.get('total_energy_cost', 0),
            "total_execution_cost": json_metrics.get('total_execution_cost', 0), 
            "total_communication_cost": json_metrics.get('total_communication_cost', 0),
            "total_cycles": json_metrics.get('total_cycles', 0),
            "bpso_config_file": bpso_config_path
        }
        
        # Load BPSO configuration for analysis
        bpso_stats = self.analyze_bpso_config(bpso_config_path)
        results["bpso_optimization"] = bpso_stats
        
        for run in range(runs):
            print(f"  BPSO delegation run {run + 1}/{runs}")
            
            cmd = [
                self.sa_sim_delegate,
                f"--image={self.test_image}",
                f"--tflite_model={self.models_dir}/{model_name}",
                "--use_sa_sim_delegate=true",
                f"--bpso_partition_config={bpso_config_path}",
                "--verbose=1"
            ]
            
            start_time = time.time()
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                   universal_newlines=True, cwd=self.workspace_dir)
            end_time = time.time()
            
            wall_time = (end_time - start_time) * 1000
            inference_time = self.extract_inference_time(result.stdout)
            delegation_info = self.extract_delegate_info(result.stdout)
            
            run_data = {
                "run": run + 1,
                "inference_time_ms": inference_time,
                "wall_time_ms": wall_time,
                "delegation_info": delegation_info,
                "returncode": result.returncode
            }
            
            results["runs"].append(run_data)
        
        # Calculate statistics
        times = [r["inference_time_ms"] for r in results["runs"] if r["inference_time_ms"] > 0]
        if times:
            results["avg_inference_time_ms"] = np.mean(times)
            results["std_inference_time_ms"] = np.std(times)
            
            # Get delegation info
            for run_data in results["runs"]:
                if run_data["delegation_info"] and run_data["delegation_info"]["total_nodes"] > 0:
                    results["delegation_info"] = run_data["delegation_info"]
                    break
            
            # Communication overhead calculation
            partitions = results["delegation_info"].get("partitions", 0)
            results["communication_overhead_ms"] = partitions * 8  # BPSO reduces overhead
            
            # Energy calculation with BPSO efficiency
            delegation_ratio = results["delegation_info"].get("delegation_ratio", 0)
            cpu_energy = np.mean(times) * 0.5 * (1 - delegation_ratio)
            sa_energy = np.mean(times) * 0.12 * delegation_ratio  # BPSO optimized SA efficiency
            results["total_energy_estimate"] = cpu_energy + sa_energy
        
        return results

    def analyze_bpso_config(self, config_file):
        """Analyze BPSO partition configuration"""
        if not os.path.exists(config_file):
            return {"error": "BPSO config file not found"}
        
        stats = {
            "total_layers": 0,
            "sa_accelerator_layers": 0,
            "cpu_layers": 0,
            "total_cpu_cycles": 0,
            "total_sa_cycles": 0,
            "total_communication_cost": 0,
            "optimization_ratio": 0
        }
        
        try:
            df = pd.read_csv(config_file)
            stats["total_layers"] = len(df)
            stats["sa_accelerator_layers"] = (df["partition_decision"] == 1).sum()
            stats["cpu_layers"] = (df["partition_decision"] == 0).sum()
            stats["total_cpu_cycles"] = df["cpu_cycles"].sum()
            stats["total_sa_cycles"] = df["sa_accelerator_cycles"].sum()
            stats["total_communication_cost"] = df["communication_cost"].sum()
            stats["optimization_ratio"] = stats["sa_accelerator_layers"] / stats["total_layers"]
            
        except Exception as e:
            stats["error"] = str(e)
        
        return stats

    def generate_comprehensive_report(self, all_results, output_file):
        """Generate comprehensive comparison report"""
        print(f"\\n=== Generating comprehensive report: {output_file} ===")
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models_tested": list(set([r["model"] for r in all_results])),
            "execution_modes": list(set([r["execution_mode"] for r in all_results])),
            "detailed_results": all_results,
            "summary_comparison": {},
            "performance_analysis": {},
            "energy_analysis": {},
            "recommendations": []
        }
        
        # Group results by model
        for model in report["models_tested"]:
            model_results = [r for r in all_results if r["model"] == model]
            
            model_summary = {
                "baseline_cpu": None,
                "default_delegation": None,
                "bpso_optimized": None,
                "performance_improvement": {},
                "energy_savings": {}
            }
            
            # Extract results for each execution mode
            for result in model_results:
                mode = result["execution_mode"]
                model_summary[mode] = {
                    "avg_inference_time_ms": result.get("avg_inference_time_ms", 0),
                    "delegation_info": result.get("delegation_info", {}),
                    "total_energy_estimate": result.get("total_energy_estimate", 0),
                    "communication_overhead_ms": result.get("communication_overhead_ms", 0)
                }
            
            # Calculate improvements
            baseline_time = model_summary["baseline_cpu"]["avg_inference_time_ms"] if model_summary["baseline_cpu"] else 0
            default_time = model_summary["default_delegation"]["avg_inference_time_ms"] if model_summary["default_delegation"] else 0
            bpso_time = model_summary["bpso_optimized"]["avg_inference_time_ms"] if model_summary["bpso_optimized"] else 0
            
            if baseline_time > 0:
                if default_time > 0:
                    model_summary["performance_improvement"]["default_vs_baseline"] = ((baseline_time - default_time) / baseline_time) * 100
                if bpso_time > 0:
                    model_summary["performance_improvement"]["bpso_vs_baseline"] = ((baseline_time - bpso_time) / baseline_time) * 100
                    if default_time > 0:
                        model_summary["performance_improvement"]["bpso_vs_default"] = ((default_time - bpso_time) / default_time) * 100
            
            # Energy analysis
            baseline_energy = model_summary["baseline_cpu"]["total_energy_estimate"] if model_summary["baseline_cpu"] else 0
            default_energy = model_summary["default_delegation"]["total_energy_estimate"] if model_summary["default_delegation"] else 0
            bpso_energy = model_summary["bpso_optimized"]["total_energy_estimate"] if model_summary["bpso_optimized"] else 0
            
            if baseline_energy > 0:
                if default_energy > 0:
                    model_summary["energy_savings"]["default_vs_baseline"] = ((baseline_energy - default_energy) / baseline_energy) * 100
                if bpso_energy > 0:
                    model_summary["energy_savings"]["bpso_vs_baseline"] = ((baseline_energy - bpso_energy) / baseline_energy) * 100
                    if default_energy > 0:
                        model_summary["energy_savings"]["bpso_vs_default"] = ((default_energy - bpso_energy) / default_energy) * 100
            
            report["summary_comparison"][model] = model_summary
        
        # Generate recommendations
        report["recommendations"] = self.generate_recommendations(report)
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate CSV summary
        self.generate_csv_summary(report, output_file.replace('.json', '_summary.csv'))
        
        return report

    def generate_csv_summary(self, all_results, csv_file):
        """Generate comprehensive CSV summary with aggregated JSON metrics"""
        print(f"\\n=== Generating Comprehensive CSV Summary: {csv_file} ===")
        
        # Prepare CSV headers
        headers = [
            'model', 'execution_mode', 'total_layers', 'delegated_layers',
            'total_cycles', 'total_energy_cost', 'total_execution_cost', 
            'total_communication_cost', 'total_compute_cost', 'total_memory_cost',
            'compute_efficiency_avg', 'memory_efficiency_avg',
            'delegated_layer_cycles', 'delegated_layer_energy', 'delegated_layer_execution_cost',
            'cpu_layer_cycles', 'cpu_layer_energy', 'cpu_layer_execution_cost',
            'performance_summary', 'partitioning_summary'
        ]
        
        csv_rows = []
        
        for model_name, model_results in all_results.items():
            for mode_name, mode_result in model_results.items():
                if not mode_result or 'summary' not in mode_result:
                    continue
                    
                summary = mode_result['summary']
                json_metrics = mode_result.get('json_metrics', {})
                
                # Basic metrics
                row = {
                    'model': model_name,
                    'execution_mode': mode_name,
                    'total_layers': summary.get('total_layers', 0),
                    'delegated_layers': summary.get('delegated_layers', 0),
                    'total_cycles': summary.get('total_cycles', 0),
                    'total_energy_cost': summary.get('total_energy_cost', 0),
                    'total_execution_cost': summary.get('total_execution_cost', 0),
                    'total_communication_cost': summary.get('total_communication_cost', 0),
                    'total_compute_cost': summary.get('total_compute_cost', 0),
                    'total_memory_cost': summary.get('total_memory_cost', 0),
                    'compute_efficiency_avg': f"{summary.get('compute_efficiency_avg', 0):.2f}",
                    'memory_efficiency_avg': f"{summary.get('memory_efficiency_avg', 0):.2f}"
                }
                
                # Delegation breakdown
                delegated_metrics = summary.get('delegated_metrics', {})
                cpu_metrics = summary.get('cpu_metrics', {})
                
                row.update({
                    'delegated_layer_cycles': delegated_metrics.get('total_cycles', 0),
                    'delegated_layer_energy': delegated_metrics.get('total_energy', 0),
                    'delegated_layer_execution_cost': delegated_metrics.get('total_execution_cost', 0),
                    'cpu_layer_cycles': cpu_metrics.get('total_cycles', 0),
                    'cpu_layer_energy': cpu_metrics.get('total_energy', 0),
                    'cpu_layer_execution_cost': cpu_metrics.get('total_execution_cost', 0)
                })
                
                # Generate performance and partitioning summaries from raw JSON
                json_data = mode_result.get('json_data', {})
                if json_data and 'layers' in json_data:
                    perf_summary = self.summarize_performance_metrics(json_data)
                    part_summary = self.summarize_partitioning_metrics(json_data)
                    
                    row.update({
                        'performance_summary': json.dumps(perf_summary),
                        'partitioning_summary': json.dumps(part_summary)
                    })
                else:
                    row.update({
                        'performance_summary': '{}',
                        'partitioning_summary': '{}'
                    })
                
                csv_rows.append(row)
        
        # Write CSV
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"  ✓ Comprehensive CSV written: {len(csv_rows)} rows")
        print(f"  ✓ File saved: {csv_file}")
        return csv_file

    def summarize_performance_metrics(self, json_data):
        """Summarize performance_metrics from JSON data"""
        total_read_cycles = 0
        total_process_cycles = 0
        total_idle_cycles = 0
        total_gmacs = 0
        total_effective_cycles = 0
        
        for layer_key, layer_data in json_data.get('layers', {}).items():
            if isinstance(layer_data, dict):
                perf = layer_data.get('performance_metrics', {})
                total_read_cycles += perf.get('read_cycles', 0)
                total_process_cycles += perf.get('process_cycles', 0)
                total_idle_cycles += perf.get('idle_cycles', 0)
                total_gmacs += perf.get('total_gmacs', 0)
                total_effective_cycles += perf.get('effective_cycles', 0)
        
        total_cycles = total_read_cycles + total_process_cycles + total_idle_cycles
        return {
            'total_read_cycles': total_read_cycles,
            'total_process_cycles': total_process_cycles,
            'total_idle_cycles': total_idle_cycles,
            'total_gmacs': total_gmacs,
            'total_effective_cycles': total_effective_cycles,
            'utilization_ratio': round(total_effective_cycles / max(total_cycles, 1), 4)
        }

    def summarize_partitioning_metrics(self, json_data):
        """Summarize partitioning_metrics from JSON data"""
        compute_intensive_layers = 0
        memory_intensive_layers = 0
        total_buffer_requirement = 0
        total_data_movement_cycles = 0
        
        for layer_key, layer_data in json_data.get('layers', {}).items():
            if isinstance(layer_data, dict):
                part = layer_data.get('partitioning_metrics', {})
                if part.get('is_compute_intensive', False):
                    compute_intensive_layers += 1
                if part.get('is_memory_intensive', False):
                    memory_intensive_layers += 1
                total_buffer_requirement += part.get('buffer_requirement', 0)
                total_data_movement_cycles += part.get('data_movement_cycles', 0)
        
        return {
            'compute_intensive_layers': compute_intensive_layers,
            'memory_intensive_layers': memory_intensive_layers,
            'total_buffer_requirement': total_buffer_requirement,
            'total_data_movement_cycles': total_data_movement_cycles
        }

    def generate_recommendations(self, report):
        """Generate optimization recommendations"""
        recommendations = []
        
        for model, summary in report["summary_comparison"].items():
            if summary["bpso_optimized"] and summary["default_delegation"]:
                bpso_improvement = summary["performance_improvement"].get("bpso_vs_default", 0)
                
                if bpso_improvement > 10:
                    recommendations.append(f"{model}: BPSO optimization provides {bpso_improvement:.1f}% performance improvement over default delegation")
                elif bpso_improvement > 0:
                    recommendations.append(f"{model}: BPSO optimization provides modest {bpso_improvement:.1f}% improvement")
                else:
                    recommendations.append(f"{model}: Consider tuning BPSO parameters - current performance is suboptimal")
        
        return recommendations

    def run_comprehensive_workflow(self, models=None):
        """Run the complete comprehensive workflow"""
        if models is None:
            models = self.models
        
        print("=== SECDA-Lite Comprehensive BPSO Workflow Profiling ===")
        print(f"Testing models: {models}")
        print(f"Output directory: {self.results_dir}")
        
        all_results = []
        
        for model in models:
            print(f"\\n{'='*60}")
            print(f"PROCESSING MODEL: {model}")
            print(f"{'='*60}")
            
            # 1. Baseline profiling
            baseline_results = self.run_baseline_profiling(model)
            all_results.append(baseline_results)
            
            # 2. Default delegation profiling
            default_results = self.run_default_delegation_profiling(model)
            all_results.append(default_results)
            
            # 3. BPSO optimization profiling
            bpso_results = self.run_bpso_delegation_profiling(model)
            all_results.append(bpso_results)
        
        # Generate comprehensive report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"{self.results_dir}/comprehensive_bpso_profile_{timestamp}.json"
        
        final_report = self.generate_comprehensive_report(all_results, report_file)
        
        print(f"\\n{'='*60}")
        print("COMPREHENSIVE PROFILING COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {report_file}")
        print(f"CSV summary: {report_file.replace('.json', '_summary.csv')}")
        
        # Print key findings
        self.print_key_findings(final_report)
        
        return final_report

    def print_key_findings(self, report):
        """Print key findings summary"""
        print("\\n=== KEY FINDINGS ===")
        
        for model, summary in report["summary_comparison"].items():
            print(f"\\n{model}:")
            
            if summary["baseline_cpu"]:
                print(f"  Baseline CPU: {summary['baseline_cpu']['avg_inference_time_ms']:.1f}ms")
            
            if summary["default_delegation"]:
                print(f"  Default Delegation: {summary['default_delegation']['avg_inference_time_ms']:.1f}ms "
                      f"({summary['default_delegation']['delegation_info'].get('delegated_nodes', 0)} nodes)")
            
            if summary["bpso_optimized"]:
                print(f"  BPSO Optimized: {summary['bpso_optimized']['avg_inference_time_ms']:.1f}ms "
                      f"({summary['bpso_optimized']['delegation_info'].get('delegated_nodes', 0)} nodes)")
            
            # Performance improvements
            if "bpso_vs_default" in summary["performance_improvement"]:
                improvement = summary["performance_improvement"]["bpso_vs_default"]
                print(f"  BPSO vs Default: {improvement:+.1f}% performance change")

def main():
    parser = argparse.ArgumentParser(description="SECDA-Lite Comprehensive BPSO Workflow Profiler")
    parser.add_argument("--models", nargs="+", default=["mobilenetv1.tflite", "mobilenetv2.tflite"],
                       help="Models to test")
    parser.add_argument("--workspace", default="/workspaces/SECDA-Lite",
                       help="Workspace directory")
    
    args = parser.parse_args()
    
    profiler = BPSOWorkflowProfiler(workspace_dir=args.workspace)
    
    try:
        # Run comprehensive workflow
        results = profiler.run_comprehensive_workflow(models=args.models)
        print("\\nProfiling completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
