#include "pbso_partition_config.h"
#include <sstream>

namespace tflite {
namespace delegates {
namespace sa_sim {

// Global instance
PBSOPartitionConfig* g_pbso_partition_config = nullptr;

bool PBSOPartitionConfig::LoadPartitionConfig(const std::string& config_file) {
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

bool PBSOPartitionConfig::ParseCSVConfig(const std::string& file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    std::cerr << "Failed to open PBSO partition config: " << file_path << std::endl;
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
    
    if (tokens.size() < 4) continue;  // Skip malformed lines
    
    LayerPartition partition;
    try {
      partition.layer_id = std::stoi(tokens[0]);
      partition.layer_name = tokens[1];
      partition.operation_type = tokens[2];
      
      // Parse processing unit assignment
      std::string unit_str = tokens[3];
      if (unit_str == "SA" || unit_str == "1" || unit_str == "SA_ACCELERATOR") {
        partition.assigned_unit = ProcessingUnit::SA_ACCELERATOR;
        partition.force_delegation = true;
      } else {
        partition.assigned_unit = ProcessingUnit::CPU;
        partition.force_delegation = false;
      }
      
      // Optional: execution cost (if available)
      if (tokens.size() > 4) {
        partition.execution_cost = std::stod(tokens[4]);
      }
      
      // Optional: communication cost (if available)
      if (tokens.size() > 5) {
        partition.communication_cost = std::stod(tokens[5]);
      }
      
      layer_partitions_[partition.layer_id] = partition;
      
    } catch (const std::exception& e) {
      std::cerr << "Error parsing PBSO config line: " << line << std::endl;
      continue;
    }
  }
  
  file.close();
  std::cout << "Loaded PBSO partition config with " << layer_partitions_.size() 
            << " layer assignments from " << file_path << std::endl;
  
  return !layer_partitions_.empty();
}

bool PBSOPartitionConfig::ParseJSONConfig(const std::string& file_path) {
  // TODO: Implement JSON parsing if needed
  std::cerr << "JSON partition config parsing not implemented yet" << std::endl;
  return false;
}

bool PBSOPartitionConfig::ShouldDelegateLayer(int layer_id, const std::string& op_type) const {
  if (!pbso_control_enabled_) {
    // Fall back to default delegate logic (CONV2D only)
    return op_type == "CONV_2D";
  }
  
  auto it = layer_partitions_.find(layer_id);
  if (it != layer_partitions_.end()) {
    const LayerPartition& partition = it->second;
    
    // PBSO decision: delegate if assigned to SA accelerator
    return partition.assigned_unit == ProcessingUnit::SA_ACCELERATOR;
  }
  
  // If layer not found in PBSO config, use default logic
  return op_type == "CONV_2D";
}

ProcessingUnit PBSOPartitionConfig::GetAssignedUnit(int layer_id) const {
  auto it = layer_partitions_.find(layer_id);
  if (it != layer_partitions_.end()) {
    return it->second.assigned_unit;
  }
  return ProcessingUnit::CPU;  // Default to CPU
}

int PBSOPartitionConfig::GetTotalDelegatedLayers() const {
  int count = 0;
  for (const auto& pair : layer_partitions_) {
    if (pair.second.assigned_unit == ProcessingUnit::SA_ACCELERATOR) {
      count++;
    }
  }
  return count;
}

int PBSOPartitionConfig::GetTotalCPULayers() const {
  int count = 0;
  for (const auto& pair : layer_partitions_) {
    if (pair.second.assigned_unit == ProcessingUnit::CPU) {
      count++;
    }
  }
  return count;
}

void PBSOPartitionConfig::PrintPartitionConfig() const {
  std::cout << "=== PBSO Partition Configuration ===" << std::endl;
  std::cout << "Total layers: " << layer_partitions_.size() << std::endl;
  std::cout << "SA Accelerator layers: " << GetTotalDelegatedLayers() << std::endl;
  std::cout << "CPU layers: " << GetTotalCPULayers() << std::endl;
  std::cout << "PBSO control enabled: " << (pbso_control_enabled_ ? "YES" : "NO") << std::endl;
  
  for (const auto& pair : layer_partitions_) {
    const LayerPartition& p = pair.second;
    std::cout << "Layer " << p.layer_id << " (" << p.operation_type << ") -> " 
              << (p.assigned_unit == ProcessingUnit::SA_ACCELERATOR ? "SA" : "CPU") << std::endl;
  }
}

// Global functions
void InitializePBSOPartitionConfig() {
  if (g_pbso_partition_config == nullptr) {
    g_pbso_partition_config = new PBSOPartitionConfig();
  }
}

void CleanupPBSOPartitionConfig() {
  if (g_pbso_partition_config != nullptr) {
    delete g_pbso_partition_config;
    g_pbso_partition_config = nullptr;
  }
}

}  // namespace sa_sim
}  // namespace delegates
}  // namespace tflite
