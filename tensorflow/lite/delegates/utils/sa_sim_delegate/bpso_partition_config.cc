#include "bpso_partition_config.h"
#include <sstream>

namespace tflite {
namespace delegates {
namespace sa_sim {

// Global instance
BPSOPartitionConfig* g_bpso_partition_config = nullptr;

bool BPSOPartitionConfig::LoadPartitionConfig(const std::string& config_file) {
  config_file_path_ = config_file;
  
  // Determine file format based on extension
  if (config_file.find(".csv") != std::string::npos) {
    return ParseCSVConfig(config_file);
  } else if (config_file.find(".json") != std::string::npos) {
    return ParseJSONConfig(config_file);
  }
  
  std::cerr << "Unsupported partition config file format: " << config_file << std::endl;
  return false;
}

bool BPSOPartitionConfig::ParseCSVConfig(const std::string& file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open BPSO partition config: " << file_path << std::endl;
    return false;
  }
  
  std::string line;
  bool is_header = true;
  
  while (std::getline(file, line)) {
    if (is_header) {
      is_header = false;
      continue;  // Skip header row
    }
    
    std::stringstream ss(line);
    std::string item;
    std::vector<std::string> tokens;
    
    // Parse CSV line
    while (std::getline(ss, item, ',')) {
      tokens.push_back(item);
    }
    
    if (tokens.size() < 3) continue;  // Skip malformed lines (need at least layer_id, layer_type, partition_decision)
    
    LayerPartition partition;
    try {
      partition.layer_id = std::stoi(tokens[0]);
      partition.operation_type = tokens[1];  // layer_type column
      
      // Parse processing unit assignment from partition_decision column
      std::string unit_str = tokens[2];  // partition_decision column
      if (unit_str == "SA" || unit_str == "1" || unit_str == "SA_ACCELERATOR") {
        partition.assigned_unit = ProcessingUnit::SA_ACCELERATOR;
        partition.force_delegation = true;
      } else {
        partition.assigned_unit = ProcessingUnit::CPU;
        partition.force_delegation = false;
      }
      
      // Optional: execution cost (if available)
      if (tokens.size() > 3) {
        partition.execution_cost = std::stod(tokens[3]);  // cpu_cycles column
      }
      
      // Optional: communication cost (if available) 
      if (tokens.size() > 4) {
        partition.communication_cost = std::stod(tokens[5]);  // communication_cost column
      }
      
      layer_partitions_[partition.layer_id] = partition;
      
    } catch (const std::exception& e) {
      std::cerr << "Error parsing BPSO config line: " << line << std::endl;
      continue;
    }
  }
  
  file.close();
  std::cout << "Loaded BPSO partition config with " << layer_partitions_.size() 
            << " layer assignments from " << file_path << std::endl;
  
  return !layer_partitions_.empty();
}

bool BPSOPartitionConfig::ParseJSONConfig(const std::string& file_path) {
  // TODO: Implement JSON parsing if needed
  std::cerr << "JSON partition config parsing not implemented yet" << std::endl;
  return false;
}

bool BPSOPartitionConfig::ShouldDelegateLayer(int layer_id, const std::string& op_type) const {
  if (!bpso_control_enabled_) {
    // Fall back to default delegate logic (CONV2D only)
    return op_type == "CONV_2D";
  }
  
  auto it = layer_partitions_.find(layer_id);
  if (it != layer_partitions_.end()) {
    const LayerPartition& partition = it->second;
    
    // BPSO decision: delegate if assigned to SA accelerator
    return partition.assigned_unit == ProcessingUnit::SA_ACCELERATOR;
  }
  
  // If layer not found in BPSO config, use default logic
  return op_type == "CONV_2D";
}

ProcessingUnit BPSOPartitionConfig::GetAssignedUnit(int layer_id) const {
  auto it = layer_partitions_.find(layer_id);
  if (it != layer_partitions_.end()) {
    return it->second.assigned_unit;
  }
  return ProcessingUnit::CPU;  // Default to CPU
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
  std::cout << "Total layers: " << layer_partitions_.size() << std::endl;
  std::cout << "SA Accelerator layers: " << GetTotalDelegatedLayers() << std::endl;
  std::cout << "CPU layers: " << GetTotalCPULayers() << std::endl;
  std::cout << "BPSO control enabled: " << (bpso_control_enabled_ ? "YES" : "NO") << std::endl;
  
  for (const auto& pair : layer_partitions_) {
    const LayerPartition& p = pair.second;
    std::cout << "Layer " << p.layer_id << " (" << p.operation_type << ") -> " 
              << (p.assigned_unit == ProcessingUnit::SA_ACCELERATOR ? "SA" : "CPU") << std::endl;
  }
}

// Global functions
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
