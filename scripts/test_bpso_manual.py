#!/usr/bin/env python3
"""
Manual BPSO test script to generate proper BPSO profiling data
"""

import subprocess
import json
import os
import sys

def run_bpso_profiling_manual():
    """Run BPSO profiling manually and generate proper JSON"""
    
    workspace_dir = "/root/Workspace/tensorflow"
    
    print("=== Manual BPSO Profiling Test ===")
    
    # Step 1: Ensure we have BPSO config
    bpso_config_path = f"{workspace_dir}/outputs/bpso_partition_config.csv"
    if not os.path.exists(bpso_config_path):
        print("ERROR: BPSO config not found. Running BPSO optimization first...")
        result = subprocess.run(["python3", "bpso_layer_partitioning.py"], 
                               cwd=f"{workspace_dir}/scripts")
        if result.returncode != 0:
            print("ERROR: BPSO optimization failed")
            return
    
    # Step 2: Run inference with BPSO config to generate SystemC CSV
    print("Step 1: Running inference with BPSO config...")
    
    # Clear previous CSV
    csv_output = f"{workspace_dir}/outputs/sa_sim.csv"
    if os.path.exists(csv_output):
        os.remove(csv_output)
    
    # Run with BPSO config
    cmd = [
        f"{workspace_dir}/bazel-bin/tensorflow/lite/delegates/utils/sa_sim_delegate/label_image_plus_sa_sim_delegate",
        f"--tflite_model={workspace_dir}/models/mobilenetv1.tflite",
        f"--image={workspace_dir}/test_images/grace_hopper.bmp",
        f"--labels={workspace_dir}/tensorflow/lite/examples/ios/camera/data/labels.txt",
        "--use_sa_sim_delegate=true",
        f"--bpso_partition_config={bpso_config_path}",
        "-p", "1",
        "--verbose", "1"
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                           universal_newlines=True, cwd=workspace_dir)
    
    # Extract delegation info
    output_text = result.stdout + result.stderr
    num_delegated_layers = 0
    for line in output_text.split('\n'):
        if "nodes delegated out of" in line:
            parts = line.split("nodes delegated out of")
            if len(parts) > 0:
                try:
                    num_delegated_layers = int(parts[0].split()[-1])
                    print(f"Found {num_delegated_layers} delegated layers with BPSO")
                    break
                except ValueError:
                    pass
    
    if num_delegated_layers == 0:
        print("ERROR: No delegated layers found")
        return
    
    # Step 3: Import and use per_layer_profiling components
    print("Step 2: Processing profiling data...")
    
    sys.path.append(f"{workspace_dir}/scripts")
    from per_layer_profiling import LayerProfiler
    
    profiler = LayerProfiler()
    
    # Get layer info
    model_path = f"{workspace_dir}/models/mobilenetv1.tflite"
    binary_path = f"{workspace_dir}/bazel-bin/tensorflow/lite/delegates/utils/sa_sim_delegate/label_image_plus_sa_sim_delegate"
    
    layers = profiler.get_layer_info_from_tflite_verbose(model_path, binary_path)
    
    # Parse SystemC data with BPSO delegation count
    layer_metrics = profiler.parse_systemc_profiling_per_layer(csv_output, num_delegated_layers)
    
    print(f"Extracted {len(layers)} layers, {len(layer_metrics)} SystemC metrics")
    
    # Step 4: Load BPSO config to map layer decisions
    print("Step 3: Loading BPSO decisions...")
    
    bpso_decisions = {}
    with open(bpso_config_path, 'r') as f:
        import csv
        reader = csv.DictReader(f)
        for row in reader:
            layer_id = int(row.get('layer_id', 0))
            layer_type = row.get('layer_type', '')
            partition_decision = row.get('partition_decision', '0') == '1'
            bpso_decisions[layer_id] = {
                'layer_type': layer_type,
                'delegate': partition_decision
            }
    
    print(f"Loaded BPSO decisions for {len(bpso_decisions)} layers")
    
    # Step 5: Create profile data with BPSO decisions
    print("Step 4: Creating BPSO profile data...")
    
    profile_data = {
        'model_name': 'mobilenetv1_bpso_manual',
        'delegate_type': 'bpso_optimized',
        'total_layers': len(layers),
        'delegated_layers': num_delegated_layers,
        'layers': {}
    }
    
    delegated_layer_idx = 0
    
    for i, layer in enumerate(layers):
        layer_name = f"layer_{i}"
        
        # Use BPSO decision for this layer
        bpso_decision = bpso_decisions.get(i, {})
        is_delegated = bpso_decision.get('delegate', False)
        
        if is_delegated and delegated_layer_idx < len(layer_metrics):
            # Use SystemC profiling data for delegated layers
            delegated_name = f"delegated_layer_{delegated_layer_idx}"
            layer_metrics_data = layer_metrics.get(delegated_name, {})
            layer['is_delegated'] = True
            delegated_layer_idx += 1
        else:
            # Use CPU baseline for non-delegated layers
            layer_metrics_data = profiler.estimate_cpu_baseline(layer)
            layer['is_delegated'] = False
        
        profile_data['layers'][layer_name] = {
            'layer_id': i,
            'layer_info': layer,
            'is_delegated': is_delegated,
            'performance_metrics': layer_metrics_data,
            'partitioning_metrics': profiler.calculate_partitioning_metrics(layer, layer_metrics_data),
            'bpso_decision': bpso_decision
        }
    
    # Step 6: Save the profile
    print("Step 5: Saving BPSO profile...")
    
    output_file = f"{workspace_dir}/results/mobilenetv1_bpso_manual_partitioning_profile.json"
    with open(output_file, 'w') as f:
        json.dump(profile_data, f, indent=2)
    
    print(f"BPSO profile saved to: {output_file}")
    
    # Step 7: Print summary
    print("\n=== BPSO Profile Summary ===")
    print(f"Total layers: {profile_data['total_layers']}")
    print(f"Delegated layers: {profile_data['delegated_layers']}")
    
    delegated_count = sum(1 for layer_data in profile_data['layers'].values() 
                         if layer_data['is_delegated'])
    print(f"Layers marked as delegated in profile: {delegated_count}")
    
    # Calculate total metrics
    total_energy = sum(layer_data['partitioning_metrics'].get('energy_cost', 0) 
                      for layer_data in profile_data['layers'].values())
    total_cycles = sum(layer_data['performance_metrics'].get('total_cycles', 0) 
                      for layer_data in profile_data['layers'].values())
    
    print(f"Total energy cost: {total_energy}")
    print(f"Total cycles: {total_cycles}")
    
    return output_file

if __name__ == "__main__":
    run_bpso_profiling_manual()
