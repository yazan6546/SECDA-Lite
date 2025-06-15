#!/usr/bin/env python3
"""
Comprehensive DNN Profiling Script for SECDA-Lite
This script profiles every stage of the DNN workflow with detailed cost and cycle information
for objective function optimization.
"""

import subprocess
import csv
import json
import os
import time
import pandas as pd
from pathlib import Path

class SECDAProfiler:
    def __init__(self, workspace_dir="/workspaces/SECDA-Lite"):
        self.workspace_dir = workspace_dir
        self.models_dir = f"{workspace_dir}/models"
        self.test_images_dir = f"{workspace_dir}/test_images"
        self.outputs_dir = f"{workspace_dir}/outputs"
        self.results_dir = f"{workspace_dir}/results"
        
        # Ensure output directories exist
        Path(self.outputs_dir).mkdir(exist_ok=True)
        Path(self.results_dir).mkdir(exist_ok=True)
        
        # Available models
        self.models = [
            "mobilenetv1.tflite",
            "mobilenetv2.tflite", 
            "inceptionv1.tflite",
            "inceptionv3.tflite",
            "resnet18v1.tflite",
            "resnet50v2.tflite",
            "efficientnet_lite.tflite",
            "mobilebert_quant.tflite",
            "albert_int8.tflite"
        ]
        
        # Available delegates
        self.delegates = {
            "sa_sim": "bazel-bin/tensorflow/lite/delegates/utils/sa_sim_delegate/label_image_plus_sa_sim_delegate",
            "vm_sim": "bazel-bin/tensorflow/lite/delegates/utils/vm_sim_delegate/label_image_plus_vm_sim_delegate", 
            "bert_sim": "bazel-bin/tensorflow/lite/delegates/utils/bert_sim_delegate/label_image_plus_bert_sim_delegate"
        }
        
        # Test image
        self.test_image = f"{self.test_images_dir}/grace_hopper.bmp"
        self.labels_file = "imagenet_classification_eval_plus_sa_sim_delegate"

    def run_baseline_profiling(self, model_name, runs=5):
        """Run baseline TFLite profiling without any delegates"""
        print(f"Running baseline profiling for {model_name}...")
        
        baseline_results = {
            "model": model_name,
            "delegate": "none",
            "runs": [],
            "avg_inference_time_ms": 0,
            "std_inference_time_ms": 0,
            "min_inference_time_ms": 0,
            "max_inference_time_ms": 0
        }
        
        # Use standard label_image for baseline
        cmd = [
            "bazel-bin/tensorflow/lite/examples/label_image/label_image",
            f"--image={self.test_image}",
            f"--model={self.models_dir}/{model_name}",
            f"--labels={self.labels_file}",
            "--verbose=1"
        ]
        
        times = []
        for run in range(runs):
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_dir)
            end_time = time.time()
            
            # Parse timing from output
            inference_time = self._extract_inference_time(result.stdout)
            wall_time = (end_time - start_time) * 1000  # Convert to ms
            
            run_data = {
                "run": run + 1,
                "inference_time_ms": inference_time,
                "wall_time_ms": wall_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
            baseline_results["runs"].append(run_data)
            if inference_time > 0:
                times.append(inference_time)
            
        if times:
            baseline_results["avg_inference_time_ms"] = sum(times) / len(times)
            baseline_results["std_inference_time_ms"] = (sum((x - baseline_results["avg_inference_time_ms"])**2 for x in times) / len(times))**0.5
            baseline_results["min_inference_time_ms"] = min(times)
            baseline_results["max_inference_time_ms"] = max(times)
            
        return baseline_results

    def run_delegate_profiling(self, model_name, delegate_name, runs=5):
        """Run detailed profiling with SECDA delegates focusing on SystemC cycles"""
        print(f"Running {delegate_name} delegate profiling for {model_name}...")
        
        delegate_results = {
            "model": model_name,
            "delegate": delegate_name,
            "runs": [],
            "systemc_cycles_summary": {},
            "performance_summary": {},
            "detailed_per_operation": {}
        }
        
        # Clean previous outputs
        self._clean_output_files(delegate_name)
        
        binary = self.delegates[delegate_name]
        
        for run in range(runs):
            print(f"  Run {run + 1}/{runs}")
            
            # Run with delegate enabled
            cmd = [
                binary,
                f"--image={self.test_image}",
                f"--model={self.models_dir}/{model_name}",
                f"--labels={self.labels_file}",
                f"--use_{delegate_name}_delegate=true",
                "--verbose=3"
            ]
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.workspace_dir)
            end_time = time.time()
            
            wall_time = (end_time - start_time) * 1000
            
            # Read SystemC profiling data (cycles, not simulation time)
            systemc_data = self._read_systemc_profiling_data(delegate_name)
            
            run_data = {
                "run": run + 1,
                "wall_time_ms": wall_time,  # Real execution time
                "systemc_cycles": systemc_data,  # SystemC cycle counts
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
            delegate_results["runs"].append(run_data)
            
        # Aggregate results focusing on cycle metrics
        self._aggregate_delegate_results(delegate_results)
        
        return delegate_results

    def _extract_inference_time(self, output):
        """Extract inference time from tool output"""
        lines = output.split('\n')
        for line in lines:
            if 'average time' in line.lower() or 'inference' in line.lower():
                # Look for timing patterns
                import re
                time_match = re.search(r'(\d+\.?\d*)\s*ms', line)
                if time_match:
                    return float(time_match.group(1))
        return 0

    def _clean_output_files(self, delegate_name):
        """Clean previous profiling output files"""
        output_files = [
            f"{self.outputs_dir}/{delegate_name}.csv",
            f"{self.outputs_dir}/{delegate_name}_model.csv"
        ]
        
        for file_path in output_files:
            if os.path.exists(file_path):
                os.remove(file_path)

    def _read_systemc_profiling_data(self, delegate_name):
        """Read and parse SystemC profiling CSV data focusing on clock cycles"""
        csv_file = f"{self.outputs_dir}/{delegate_name}.csv"
        model_csv_file = f"{self.outputs_dir}/{delegate_name}_model.csv"
        
        profiling_data = {
            "per_operation_cycles": {},
            "total_systemc_cycles": 0,
            "cycle_breakdown": {},
            "performance_counters": {}
        }
        
        # Read per-operation cycle data
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                total_cycles = 0
                
                for _, row in df.iterrows():
                    op_name = row.get('layer_name', row.get('operation', f"op_{len(profiling_data['per_operation_cycles'])}"))
                    
                    # Extract SystemC cycle counts (not simulation time)
                    cycles = int(row.get('total_cycles', row.get('cycles', row.get('systemc_cycles', 0))))
                    compute_cycles = int(row.get('compute_cycles', 0))
                    stall_cycles = int(row.get('stall_cycles', 0))
                    memory_cycles = int(row.get('memory_cycles', 0))
                    
                    op_data = {
                        'systemc_cycles': cycles,
                        'compute_cycles': compute_cycles,
                        'stall_cycles': stall_cycles,
                        'memory_cycles': memory_cycles,
                        'mac_operations': int(row.get('mac_ops', row.get('operations', 0))),
                        'memory_reads': int(row.get('memory_reads', 0)),
                        'memory_writes': int(row.get('memory_writes', 0)),
                        'cache_hits': int(row.get('cache_hits', 0)),
                        'cache_misses': int(row.get('cache_misses', 0)),
                        'utilization': float(row.get('utilization', 0.0)) if row.get('utilization') else 0.0
                    }
                    
                    profiling_data["per_operation_cycles"][op_name] = op_data
                    total_cycles += cycles
                    
                profiling_data["total_systemc_cycles"] = total_cycles
                
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        
        # Read model-level cycle summary
        if os.path.exists(model_csv_file):
            try:
                df = pd.read_csv(model_csv_file)
                if not df.empty:
                    model_data = df.iloc[0].to_dict()
                    
                    # Extract cycle-based metrics (ignore inference_time_ms)
                    profiling_data["cycle_breakdown"] = {
                        'total_cycles': int(model_data.get('total_cycles', 0)),
                        'compute_cycles': int(model_data.get('compute_cycles', 0)),
                        'memory_cycles': int(model_data.get('memory_cycles', 0)),
                        'stall_cycles': int(model_data.get('stall_cycles', 0)),
                        'idle_cycles': int(model_data.get('idle_cycles', 0))
                    }
                    
                    profiling_data["performance_counters"] = {
                        'total_operations': int(model_data.get('total_operations', 0)),
                        'total_memory_accesses': int(model_data.get('total_memory_accesses', 0)),
                        'cache_hit_rate': float(model_data.get('cache_hit_rate', 0.0)),
                        'average_utilization': float(model_data.get('average_utilization', 0.0)),
                        'peak_utilization': float(model_data.get('peak_utilization', 0.0))
                    }
                    
            except Exception as e:
                print(f"Error reading {model_csv_file}: {e}")
                
        return profiling_data

    def _aggregate_delegate_results(self, delegate_results):
        """Aggregate results across multiple runs focusing on SystemC cycles"""
        runs = delegate_results["runs"]
        if not runs:
            return
            
        # Remove inference_time_ms aggregation as it's meaningless for simulation
        
        # Aggregate SystemC cycle metrics
        cycle_metrics = {}
        performance_counters = {}
        
        for run in runs:
            systemc_data = run["systemc_cycles"]
            
            # Aggregate cycle breakdown
            if systemc_data.get("cycle_breakdown"):
                for metric, value in systemc_data["cycle_breakdown"].items():
                    if metric not in cycle_metrics:
                        cycle_metrics[metric] = []
                    if isinstance(value, (int, float)) and value > 0:
                        cycle_metrics[metric].append(value)
            
            # Aggregate performance counters
            if systemc_data.get("performance_counters"):
                for metric, value in systemc_data["performance_counters"].items():
                    if metric not in performance_counters:
                        performance_counters[metric] = []
                    if isinstance(value, (int, float)):
                        performance_counters[metric].append(value)
        
        # Calculate cycle statistics
        delegate_results["systemc_cycles_summary"] = {}
        for metric, values in cycle_metrics.items():
            if values:
                delegate_results["systemc_cycles_summary"][metric] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5 if len(values) > 1 else 0
                }
        
        # Calculate performance counter statistics
        delegate_results["performance_summary"] = {}
        for metric, values in performance_counters.items():
            if values:
                delegate_results["performance_summary"][metric] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5 if len(values) > 1 else 0
                }

    def run_comprehensive_profiling(self):
        """Run comprehensive profiling across all models and delegates"""
        print("Starting comprehensive DNN profiling...")
        
        all_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "workspace": self.workspace_dir,
            "models_tested": [],
            "delegates_tested": list(self.delegates.keys()),
            "baseline_results": {},
            "delegate_results": {},
            "comparison_analysis": {}
        }
        
        # Test each model
        for model in self.models:
            model_path = f"{self.models_dir}/{model}"
            if not os.path.exists(model_path):
                print(f"Model {model} not found, skipping...")
                continue
                
            print(f"\n{'='*60}")
            print(f"Testing model: {model}")
            print(f"{'='*60}")
            
            all_results["models_tested"].append(model)
            
            # Run baseline profiling
            baseline_result = self.run_baseline_profiling(model)
            all_results["baseline_results"][model] = baseline_result
            
            # Run delegate profiling
            all_results["delegate_results"][model] = {}
            
            for delegate_name in self.delegates.keys():
                # Check if delegate binary exists
                binary_path = f"{self.workspace_dir}/{self.delegates[delegate_name]}"
                if not os.path.exists(binary_path):
                    print(f"Delegate binary {binary_path} not found, skipping {delegate_name}...")
                    continue
                    
                delegate_result = self.run_delegate_profiling(model, delegate_name)
                all_results["delegate_results"][model][delegate_name] = delegate_result
        
        # Generate comparison analysis
        self._generate_comparison_analysis(all_results)
        
        # Save results
        self._save_results(all_results)
        
        return all_results

    def _generate_comparison_analysis(self, all_results):
        """Generate detailed comparison analysis for objective function"""
        comparison = {}
        
        for model in all_results["models_tested"]:
            model_comparison = {
                "baseline_time_ms": 0,
                "delegate_performance": {},
                "speedup_factors": {},
                "efficiency_metrics": {},
                "resource_utilization": {}
            }
            
            # Get baseline time
            baseline = all_results["baseline_results"].get(model, {})
            model_comparison["baseline_time_ms"] = baseline.get("avg_inference_time_ms", 0)
            
            # Compare each delegate
            delegate_results = all_results["delegate_results"].get(model, {})
            
            for delegate_name, delegate_data in delegate_results.items():
                delegate_time = delegate_data.get("avg_inference_time_ms", 0)
                
                # Calculate speedup
                speedup = model_comparison["baseline_time_ms"] / delegate_time if delegate_time > 0 else 0
                
                model_comparison["delegate_performance"][delegate_name] = delegate_time
                model_comparison["speedup_factors"][delegate_name] = speedup
                
                # Extract efficiency metrics from SystemC data
                systemc_metrics = delegate_data.get("systemc_metrics", {})
                
                efficiency_data = {}
                if "read_cycles" in systemc_metrics:
                    efficiency_data["read_cycles"] = systemc_metrics["read_cycles"]
                if "process_cycles" in systemc_metrics:
                    efficiency_data["process_cycles"] = systemc_metrics["process_cycles"]
                if "gmacs" in systemc_metrics:
                    efficiency_data["gmacs"] = systemc_metrics["gmacs"]
                if "idle" in systemc_metrics:
                    efficiency_data["idle_cycles"] = systemc_metrics["idle"]
                
                model_comparison["efficiency_metrics"][delegate_name] = efficiency_data
                
                # Calculate resource utilization
                total_cycles = 0
                active_cycles = 0
                
                for metric_name, metric_data in systemc_metrics.items():
                    if "cycles" in metric_name and isinstance(metric_data, dict):
                        cycles = metric_data.get("avg", 0)
                        total_cycles += cycles
                        if metric_name != "idle":
                            active_cycles += cycles
                
                utilization = active_cycles / total_cycles if total_cycles > 0 else 0
                model_comparison["resource_utilization"][delegate_name] = {
                    "total_cycles": total_cycles,
                    "active_cycles": active_cycles,
                    "utilization_percent": utilization * 100
                }
            
            comparison[model] = model_comparison
        
        all_results["comparison_analysis"] = comparison

    def _save_results(self, all_results):
        """Save comprehensive results to files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = f"{self.results_dir}/comprehensive_profiling_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save CSV summary for objective function
        csv_file = f"{self.results_dir}/objective_function_data_{timestamp}.csv"
        self._create_objective_function_csv(all_results, csv_file)
        
        # Save detailed report
        report_file = f"{self.results_dir}/profiling_report_{timestamp}.txt"
        self._create_detailed_report(all_results, report_file)
        
        print(f"\nResults saved:")
        print(f"  JSON: {json_file}")
        print(f"  CSV:  {csv_file}")
        print(f"  Report: {report_file}")

    def _create_objective_function_csv(self, all_results, csv_file):
        """Create CSV file optimized for objective function analysis"""
        rows = []
        
        for model in all_results["models_tested"]:
            baseline_time = all_results["comparison_analysis"][model]["baseline_time_ms"]
            
            # Add baseline row
            rows.append({
                "model": model,
                "delegate": "baseline",
                "inference_time_ms": baseline_time,
                "speedup_factor": 1.0,
                "read_cycles": 0,
                "process_cycles": 0,
                "idle_cycles": 0,
                "total_cycles": 0,
                "gmacs": 0,
                "utilization_percent": 0,
                "cost_metric": baseline_time  # Simple cost = time for baseline
            })
            
            # Add delegate rows
            delegate_results = all_results["delegate_results"].get(model, {})
            comparison = all_results["comparison_analysis"][model]
            
            for delegate_name, delegate_data in delegate_results.items():
                delegate_time = comparison["delegate_performance"].get(delegate_name, 0)
                speedup = comparison["speedup_factors"].get(delegate_name, 0)
                efficiency = comparison["efficiency_metrics"].get(delegate_name, {})
                utilization = comparison["resource_utilization"].get(delegate_name, {})
                
                # Extract cycle counts
                read_cycles = efficiency.get("read_cycles", {}).get("avg", 0)
                process_cycles = efficiency.get("process_cycles", {}).get("avg", 0)
                idle_cycles = efficiency.get("idle_cycles", {}).get("avg", 0)
                total_cycles = utilization.get("total_cycles", 0)
                gmacs = efficiency.get("gmacs", {}).get("avg", 0)
                util_percent = utilization.get("utilization_percent", 0)
                
                # Calculate cost metric (weighted combination of time and resource usage)
                cost_metric = delegate_time + (idle_cycles * 0.001) + (total_cycles * 0.0001)
                
                rows.append({
                    "model": model,
                    "delegate": delegate_name,
                    "inference_time_ms": delegate_time,
                    "speedup_factor": speedup,
                    "read_cycles": read_cycles,
                    "process_cycles": process_cycles,
                    "idle_cycles": idle_cycles,
                    "total_cycles": total_cycles,
                    "gmacs": gmacs,
                    "utilization_percent": util_percent,
                    "cost_metric": cost_metric
                })
        
        # Write CSV
        if rows:
            fieldnames = rows[0].keys()
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    def _create_detailed_report(self, all_results, report_file):
        """Create detailed text report"""
        with open(report_file, 'w') as f:
            f.write("SECDA-Lite Comprehensive DNN Profiling Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {all_results['timestamp']}\n")
            f.write(f"Workspace: {all_results['workspace']}\n")
            f.write(f"Models tested: {len(all_results['models_tested'])}\n")
            f.write(f"Delegates tested: {', '.join(all_results['delegates_tested'])}\n\n")
            
            # Summary table
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Model':<20} {'Baseline (ms)':<15} {'Best Delegate':<15} {'Best Time (ms)':<15} {'Speedup':<10}\n")
            f.write("-" * 80 + "\n")
            
            for model in all_results["models_tested"]:
                comparison = all_results["comparison_analysis"][model]
                baseline_time = comparison["baseline_time_ms"]
                
                best_delegate = "none"
                best_time = baseline_time
                best_speedup = 1.0
                
                for delegate, time_ms in comparison["delegate_performance"].items():
                    if time_ms > 0 and time_ms < best_time:
                        best_delegate = delegate
                        best_time = time_ms
                        best_speedup = comparison["speedup_factors"][delegate]
                
                f.write(f"{model:<20} {baseline_time:<15.2f} {best_delegate:<15} {best_time:<15.2f} {best_speedup:<10.2f}\n")
            
            f.write("\n\nDETAILED SYSTEMC METRICS\n")
            f.write("-" * 40 + "\n")
            
            for model in all_results["models_tested"]:
                f.write(f"\nModel: {model}\n")
                f.write("-" * 20 + "\n")
                
                delegate_results = all_results["delegate_results"].get(model, {})
                for delegate_name, delegate_data in delegate_results.items():
                    f.write(f"\n  {delegate_name} delegate:\n")
                    systemc_metrics = delegate_data.get("systemc_metrics", {})
                    
                    for metric_name, metric_data in systemc_metrics.items():
                        if isinstance(metric_data, dict):
                            f.write(f"    {metric_name}: avg={metric_data.get('avg', 0):.2f}, "
                                   f"min={metric_data.get('min', 0):.2f}, "
                                   f"max={metric_data.get('max', 0):.2f}\n")
    
    def extract_objective_function_metrics(self, delegate_results):
        """Extract key metrics for objective function optimization"""
        if not delegate_results.get("systemc_cycles_summary"):
            return {}
            
        cycles_summary = delegate_results["systemc_cycles_summary"]
        perf_summary = delegate_results.get("performance_summary", {})
        
        # Key metrics for objective function
        objective_metrics = {
            # Primary performance metrics (in cycles, not time)
            "total_execution_cycles": cycles_summary.get("total_cycles", {}).get("avg", 0),
            "compute_cycles": cycles_summary.get("compute_cycles", {}).get("avg", 0),
            "memory_cycles": cycles_summary.get("memory_cycles", {}).get("avg", 0),
            "stall_cycles": cycles_summary.get("stall_cycles", {}).get("avg", 0),
            "idle_cycles": cycles_summary.get("idle_cycles", {}).get("avg", 0),
            
            # Efficiency metrics
            "compute_efficiency": 0,  # compute_cycles / total_cycles
            "memory_efficiency": 0,  # Useful memory ops / total memory cycles
            "utilization": perf_summary.get("average_utilization", {}).get("avg", 0),
            
            # Cost metrics for optimization
            "operations_per_cycle": 0,  # total_operations / total_cycles
            "cache_hit_rate": perf_summary.get("cache_hit_rate", {}).get("avg", 0),
            "memory_bandwidth_utilization": 0,
            
            # Hardware resource costs
            "total_operations": perf_summary.get("total_operations", {}).get("avg", 0),
            "memory_accesses": perf_summary.get("total_memory_accesses", {}).get("avg", 0),
            
            # Variability metrics (for robustness)
            "performance_variance": cycles_summary.get("total_cycles", {}).get("std", 0),
            "peak_utilization": perf_summary.get("peak_utilization", {}).get("avg", 0)
        }
        
        # Calculate derived efficiency metrics
        total_cycles = objective_metrics["total_execution_cycles"]
        if total_cycles > 0:
            objective_metrics["compute_efficiency"] = objective_metrics["compute_cycles"] / total_cycles
            
            total_ops = objective_metrics["total_operations"]
            if total_ops > 0:
                objective_metrics["operations_per_cycle"] = total_ops / total_cycles
                
        # Calculate memory efficiency
        memory_cycles = objective_metrics["memory_cycles"]
        memory_accesses = objective_metrics["memory_accesses"]
        if memory_cycles > 0 and memory_accesses > 0:
            # Assuming each memory access should ideally take 1 cycle
            objective_metrics["memory_efficiency"] = min(1.0, memory_accesses / memory_cycles)
            
        return objective_metrics

    def generate_objective_function_report(self, all_results):
        """Generate a report focused on objective function metrics"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_metrics": {},
            "delegate_comparison": {},
            "optimization_recommendations": []
        }
        
        for model_name in all_results["models_tested"]:
            report["model_metrics"][model_name] = {}
            
            # Get baseline reference (TFLite CPU cycles converted from time)
            baseline = all_results["baseline_results"].get(model_name, {})
            baseline_time_ms = baseline.get("avg_inference_time_ms", 0)
            
            # Assuming a reference CPU frequency for cycle conversion
            # This is approximate - real deployment would need actual CPU frequency
            ref_cpu_freq_mhz = 1000  # 1 GHz reference
            baseline_estimated_cycles = int(baseline_time_ms * ref_cpu_freq_mhz * 1000) if baseline_time_ms > 0 else 0
            
            report["model_metrics"][model_name]["baseline_estimated_cycles"] = baseline_estimated_cycles
            
            # Process each delegate
            for delegate_name, delegate_result in all_results["delegate_results"][model_name].items():
                obj_metrics = self.extract_objective_function_metrics(delegate_result)
                report["model_metrics"][model_name][delegate_name] = obj_metrics
                
                # Calculate speedup vs baseline (cycle-based)
                if baseline_estimated_cycles > 0 and obj_metrics.get("total_execution_cycles", 0) > 0:
                    speedup = baseline_estimated_cycles / obj_metrics["total_execution_cycles"]
                    obj_metrics["speedup_vs_baseline"] = speedup
                    
        return report

def main():
    """Main function to run comprehensive profiling focused on SystemC cycles"""
    print("SECDA-Lite Comprehensive DNN Profiling Tool")
    print("Focusing on SystemC Clock Cycles for Objective Function Optimization")
    print("=" * 70)
    
    profiler = SECDAProfiler()
    
    # Run comprehensive profiling
    results = profiler.run_comprehensive_profiling()
    
    # Generate objective function metrics report
    print("\nGenerating objective function metrics...")
    obj_function_report = profiler.generate_objective_function_report(results)
    
    # Save objective function report
    obj_report_file = f"{profiler.results_dir}/objective_function_metrics.json"
    with open(obj_report_file, 'w') as f:
        json.dump(obj_function_report, f, indent=2)
    
    print(f"\nProfileing completed successfully!")
    print(f"Tested {len(results['models_tested'])} models")
    print(f"Objective function metrics saved to: {obj_report_file}")
    print("\nKey metrics available for optimization:")
    print("- total_execution_cycles (primary performance metric)")
    print("- compute_efficiency (compute_cycles/total_cycles)")
    print("- memory_efficiency (memory utilization)")
    print("- operations_per_cycle (throughput)")
    print("- cache_hit_rate (memory hierarchy efficiency)")
    print("- utilization (hardware resource usage)")
    print("- speedup_vs_baseline (improvement over CPU baseline)")
    print("\nCheck the results directory for detailed output files.")

def test_single_model_cycles():
    """Quick test function to validate SystemC cycle profiling on a single model"""
    print("Testing SystemC cycle profiling on MobileNetV1...")
    
    profiler = SECDAProfiler()
    
    # Test only SA sim delegate with MobileNetV1
    model_name = "mobilenetv1.tflite"
    delegate_name = "sa_sim"
    
    if not os.path.exists(f"{profiler.models_dir}/{model_name}"):
        print(f"Model {model_name} not found!")
        return
        
    if not os.path.exists(profiler.delegates[delegate_name]):
        print(f"Delegate binary {profiler.delegates[delegate_name]} not found!")
        print("Please run the build command first:")
        print("bazel build -c opt tensorflow/lite/delegates/utils/sa_sim_delegate:label_image_plus_sa_sim_delegate -c dbg --cxxopt='-DSYSC'")
        return
    
    # Run single test
    result = profiler.run_delegate_profiling(model_name, delegate_name, runs=1)
    
    # Extract objective function metrics
    obj_metrics = profiler.extract_objective_function_metrics(result)
    
    print(f"\nSystemC Cycle Profiling Results for {model_name}:")
    print("=" * 50)
    
    if obj_metrics.get("total_execution_cycles", 0) > 0:
        print(f"Total execution cycles: {obj_metrics['total_execution_cycles']:,.0f}")
        print(f"Compute cycles: {obj_metrics['compute_cycles']:,.0f}")
        print(f"Memory cycles: {obj_metrics['memory_cycles']:,.0f}")
        print(f"Stall cycles: {obj_metrics['stall_cycles']:,.0f}")
        print(f"Compute efficiency: {obj_metrics['compute_efficiency']:.3f}")
        print(f"Operations per cycle: {obj_metrics['operations_per_cycle']:.3f}")
        print(f"Cache hit rate: {obj_metrics['cache_hit_rate']:.3f}")
        print(f"Average utilization: {obj_metrics['utilization']:.3f}")
        
        # Save test results
        test_result_file = f"{profiler.results_dir}/single_model_cycle_test.json"
        with open(test_result_file, 'w') as f:
            json.dump({
                "model": model_name,
                "delegate": delegate_name,
                "objective_metrics": obj_metrics,
                "full_results": result
            }, f, indent=2)
        print(f"\nTest results saved to: {test_result_file}")
    else:
        print("No SystemC cycle data captured. Check if:")
        print("1. The delegate is properly enabled")
        print("2. The profiling CSV files are being generated")
        print("3. The SystemC simulation is running")
        
        # Show what we captured
        print(f"\nRaw profiling data structure:")
        for run in result["runs"]:
            print(f"Run {run['run']}: {run['systemc_cycles']}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_single_model_cycles()
    else:
        main()
    # Uncomment the line below to run the single model cycle test
    # test_single_model_cycles()
