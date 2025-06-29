#!/usr/bin/env python3
"""
Batch BPSO Layer Partitioning Optimization

This script runs BPSO optimization for multiple models at once and generates
partition configurations for all supported models.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_bpso_for_model(model_name):
    """Run BPSO optimization for a specific model"""
    print(f"\n{'='*60}")
    print(f"RUNNING BPSO OPTIMIZATION FOR: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Run BPSO optimization
        cmd = [
            "python3", "scripts/bpso_layer_partitioning.py",
            "--model", model_name,
            "--output_config", f"outputs/{model_name.replace('.tflite', '')}_bpso_partition_config.csv"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: BPSO optimization completed for {model_name}")
            print("STDOUT:", result.stdout)
            return True
        else:
            print(f"❌ ERROR: BPSO optimization failed for {model_name}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: Failed to run BPSO for {model_name}: {e}")
        return False

def main():
    """Main function to run batch BPSO optimization"""
    
    # List of models to optimize
    models = [
        "mobilenetv1.tflite",
        "mobilenetv2.tflite", 
        "resnet18v1.tflite",
        "resnet50v2.tflite"
    ]
    
    print("=== BATCH BPSO LAYER PARTITIONING OPTIMIZATION ===")
    print(f"Models to optimize: {models}")
    
    # Ensure output directory exists
    Path("outputs").mkdir(exist_ok=True)
    
    # Track results
    results = {}
    successful_models = []
    failed_models = []
    
    # Run BPSO for each model
    for model in models:
        print(f"\n>>> Processing {model}...")
        
        # Check if profiling data exists
        model_name = model.replace('.tflite', '')
        profiling_data_path = f"results/{model_name}_baseline_partitioning_metrics.csv"
        
        if not os.path.exists(profiling_data_path):
            print(f"⚠️  WARNING: Profiling data not found for {model}: {profiling_data_path}")
            print(f"   Skipping {model}. Run comprehensive profiling first.")
            failed_models.append(model)
            results[model] = "missing_profiling_data"
            continue
        
        # Run BPSO optimization
        success = run_bpso_for_model(model)
        
        if success:
            successful_models.append(model)
            results[model] = "success"
        else:
            failed_models.append(model)
            results[model] = "failed"
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH BPSO OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total models: {len(models)}")
    print(f"Successful: {len(successful_models)}")
    print(f"Failed: {len(failed_models)}")
    
    if successful_models:
        print(f"\n✅ Successfully optimized models:")
        for model in successful_models:
            model_name = model.replace('.tflite', '')
            print(f"  - {model}")
            print(f"    Config: outputs/{model_name}_bpso_partition_config.csv")
            print(f"    Binary: outputs/{model_name}_bpso_binary_partition.txt")
    
    if failed_models:
        print(f"\n❌ Failed models:")
        for model in failed_models:
            print(f"  - {model}: {results[model]}")
    
    # Show next steps
    print(f"\n=== NEXT STEPS ===")
    if successful_models:
        print("1. Use the generated partition configs with SA sim delegate:")
        for model in successful_models:
            model_name = model.replace('.tflite', '')
            print(f"   {model}: --bpso_partition_config=outputs/{model_name}_bpso_partition_config.csv")
        
        print("\n2. Run comprehensive profiling with BPSO configs:")
        print("   python3 scripts/comprehensive_bpso_profile.py --models " + " ".join(successful_models))
    
    if failed_models:
        print("\n3. Generate missing profiling data:")
        for model in failed_models:
            if results[model] == "missing_profiling_data":
                print(f"   python3 scripts/per_layer_profiling.py --model {model}")
    
    return len(successful_models), len(failed_models)

if __name__ == "__main__":
    successful, failed = main()
    
    # Exit with appropriate code
    if failed > 0:
        print(f"\nExiting with code 1 due to {failed} failed optimizations")
        sys.exit(1)
    else:
        print(f"\nAll {successful} optimizations completed successfully!")
        sys.exit(0)
