#include "tensorflow/lite/delegates/utils/sa_sim_delegate/bpso_partition_config.h"
#include <sstream>
#include <algorithm>

namespace tflite {
namespace delegates {
namespace sa_sim {

// Global instance
BPSOPartitionConfig* g_bpso_partition_config = nullptr;

bool BPSOPartitionConfig::LoadPartitionConfig(const std::string& config_file) {
  config_file_path_ = config_file;
  
  // Check file extension to determine format
  if (config_file.find(".csv") != std::string::npos) {
    return ParseCSVConfig(config_file);
  } else if (config_file.find(".json") != std::string::npos) {
    return ParseJSONConfig(config_file);
  }
  
  std::cerr << "Error: Unsupported config file format: " << config_file << std::endl;
  return false;
}

bool BPSOPartitionConfig::LoadBinaryPartition(const std::vector<int>& binary_partition) {
  layer_partitions_.clear();
  
  for (size_t i = 0; i < binary_partition.size(); ++i) {
    LayerPartition partition;
    partition.layer_id = static_cast<int>(i);
    partition.layer_name = "layer_" + std::to_string(i);
    partition.operation_type = "UNKNOWN";  // Will be updated when delegate checks
    partition.assigned_unit = (binary_partition[i] == 1) ? 
                              ProcessingUnit::SA_ACCELERATOR : 
                              ProcessingUnit::CPU;
    partition.execution_cost = 0.0;  // Can be updated later
    partition.communication_cost = 0.0;
    partition.force_delegation = true;  // BPSO decision overrides defaults
    
    layer_partitions_[static_cast<int>(i)] = partition;
  }
  
  bpso_control_enabled_ = true;
  std::cout << "BPSO: Loaded binary partition with " << binary_partition.size() 
            << " layer decisions" << std::endl;
  return true;
}

bool BPSOPartitionConfig::ShouldDelegateLayer(int layer_id, const std::string& op_type) const {
  if (!bpso_control_enabled_) {
    // Fallback to default delegate logic
    return (op_type == "CONV_2D" || op_type == "DEPTHWISE_CONV_2D" || op_type == "FULLY_CONNECTED");
  }
  
  auto it = layer_partitions_.find(layer_id);
  if (it != layer_partitions_.end()) {
    bool should_delegate = (it->second.assigned_unit == ProcessingUnit::SA_ACCELERATOR);
    std::cout << "BPSO: Layer " << layer_id << " (" << op_type << ") -> " 
              << (should_delegate ? "SA_ACCELERATOR" : "CPU") << std::endl;
    return should_delegate;
  }
  
  // Default: delegate CONV_2D layers if no BPSO decision available
  return (op_type == "CONV_2D" || op_type == "DEPTHWISE_CONV_2D");
}

ProcessingUnit BPSOPartitionConfig::GetAssignedUnit(int layer_id) const {
  auto it = layer_partitions_.find(layer_id);
  if (it != layer_partitions_.end()) {
    return it->second.assigned_unit;
  }
  return ProcessingUnit::CPU;  // Default
}

int BPSOPartitionConfig::GetTotalDelegatedLayers() const {
  int count = 0;
  for (const auto& pair : layer_partitions_) {
    if (pair.second.assigned_unit == ProcessingUnit::SA_ACCELERATOR) {
      count++;
    }
  }
  return count;
}

int BPSOPartitionConfig::GetTotalCPULayers() const {
  int count = 0;
  for (const auto& pair : layer_partitions_) {
    if (pair.second.assigned_unit == ProcessingUnit::CPU) {
      count++;
    }
  }
  return count;
}

void BPSOPartitionConfig::PrintPartitionConfig() const {
  std::cout << "=== BPSO Partition Configuration ===" << std::endl;
  std::cout << "BPSO Control Enabled: " << (bpso_control_enabled_ ? "YES" : "NO") << std::endl;
  std::cout << "Total Layers: " << layer_partitions_.size() << std::endl;
  std::cout << "SA Accelerator Layers: " << GetTotalDelegatedLayers() << std::endl;
  std::cout << "CPU Layers: " << GetTotalCPULayers() << std::endl;
  
  if (bpso_control_enabled_) {
    std::cout << "\nLayer Assignments:" << std::endl;
    for (const auto& pair : layer_partitions_) {
      const LayerPartition& partition = pair.second;
      std::cout << "  Layer " << partition.layer_id 
                << " (" << partition.operation_type << ") -> "
                << (partition.assigned_unit == ProcessingUnit::SA_ACCELERATOR ? "SA_ACCELERATOR" : "CPU")
                << std::endl;
    }
  }
  std::cout << "=================================" << std::endl;
}

bool BPSOPartitionConfig::ParseCSVConfig(const std::string& file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open CSV config file: " << file_path << std::endl;
    return false;
  }
  
  std::string line;
  bool is_header = true;
  
  while (std::getline(file, line)) {
    if (is_header) {
      is_header = false;
      continue;  // Skip header line
    }
    
    std::istringstream ss(line);
    std::string layer_id_str, layer_type, partition_str;
    
    if (std::getline(ss, layer_id_str, ',') &&
        std::getline(ss, layer_type, ',') &&
        std::getline(ss, partition_str)) {
      
      int layer_id = std::stoi(layer_id_str);
      int partition_decision = std::stoi(partition_str);
      
      LayerPartition partition;
      partition.layer_id = layer_id;
      partition.layer_name = "layer_" + std::to_string(layer_id);
      partition.operation_type = layer_type;
      partition.assigned_unit = (partition_decision == 1) ? 
                                ProcessingUnit::SA_ACCELERATOR : 
                                ProcessingUnit::CPU;
      partition.execution_cost = 0.0;
      partition.communication_cost = 0.0;
      partition.force_delegation = true;
      
      layer_partitions_[layer_id] = partition;
    }
  }
  
  bpso_control_enabled_ = true;
  std::cout << "BPSO: Loaded " << layer_partitions_.size() 
            << " layer partitions from CSV: " << file_path << std::endl;
  return true;
}

bool BPSOPartitionConfig::ParseJSONConfig(const std::string& file_path) {
  // JSON parsing would go here
  // For now, just return false as JSON parsing requires additional dependencies
  std::cerr << "JSON config parsing not yet implemented" << std::endl;
  return false;
}

void InitializeBPSOPartitionConfig() {
  if (g_bpso_partition_config == nullptr) {
    g_bpso_partition_config = new BPSOPartitionConfig();
  }
}

void CleanupBPSOPartitionConfig() {
  if (g_bpso_partition_config != nullptr) {
    delete g_bpso_partition_config;
    g_bpso_partition_config = nullptr;
  }
}

}  // namespace sa_sim
}  // namespace delegates
}  // namespace tflite
