import json
import csv

def aggregate_json_metrics(json_data):
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
        'memory_efficiency_avg': 0
    }
    
    # Aggregate per-layer metrics
    layer_count = 0
    for layer_key, layer_data in json_data['layers'].items():
        if not isinstance(layer_data, dict):
            continue
            
        layer_count += 1
        perf_metrics = layer_data.get('performance_metrics', {})
        part_metrics = layer_data.get('partitioning_metrics', {})
        
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
    
    # Average efficiency metrics
    if layer_count > 0:
        total_metrics['compute_efficiency_avg'] /= layer_count
        total_metrics['memory_efficiency_avg'] /= layer_count
    
    return total_metrics

# Test it
with open('results/mobilenetv1.tflite_partitioning_profile.json', 'r') as f:
    data = json.load(f)

metrics = aggregate_json_metrics(data)
print('=== Aggregated Metrics ===')
for key, value in metrics.items():
    print(f'{key}: {value}')

# Write to CSV
with open('test_metrics.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
    writer.writeheader()
    writer.writerow(metrics)

print('\n✓ CSV written to test_metrics.csv')
