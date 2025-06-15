#ifndef TENSORFLOW_LITE_DELEGATES_SA_SIM_BPSO_PARTITION_CONFIG_H_
#define TENSORFLOW_LITE_DELEGATES_SA_SIM_BPSO_PARTITION_CONFIG_H_

#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

namespace tflite {
namespace delegates {
namespace sa_sim {

enum class ProcessingUnit {
  CPU = 0,
  SA_ACCELERATOR = 1
};

struct LayerPartition {
  int layer_id;
  std::string layer_name;
  std::string operation_type;
  ProcessingUnit assigned_unit;
  double execution_cost;
  double communication_cost;
  bool force_delegation;  // Override delegate selection
};

class BPSOPartitionConfig {
 public:
  BPSOPartitionConfig() = default;
  
  // Load BPSO optimization results from CSV/JSON
  bool LoadPartitionConfig(const std::string& config_file);
  
  // Check if a specific layer should be delegated
  bool ShouldDelegateLayer(int layer_id, const std::string& op_type) const;
  
  // Get processing unit assignment for a layer
  ProcessingUnit GetAssignedUnit(int layer_id) const;
  
  // Enable/disable BPSO-controlled partitioning
  void SetBPSOControlEnabled(bool enabled) { bpso_control_enabled_ = enabled; }
  bool IsBPSOControlEnabled() const { return bpso_control_enabled_; }
  
  // Get partition statistics
  int GetTotalDelegatedLayers() const;
  int GetTotalCPULayers() const;
  
  // Debug: Print current partition configuration
  void PrintPartitionConfig() const;
  
 private:
  std::map<int, LayerPartition> layer_partitions_;
  bool bpso_control_enabled_ = false;
  std::string config_file_path_;
  
  // Parse CSV format partition file
  bool ParseCSVConfig(const std::string& file_path);
  
  // Parse JSON format partition file  
  bool ParseJSONConfig(const std::string& file_path);
};

// Global partition config instance
extern BPSOPartitionConfig* g_bpso_partition_config;

// Helper functions
void InitializeBPSOPartitionConfig();
void CleanupBPSOPartitionConfig();

}  // namespace sa_sim
}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_SA_SIM_BPSO_PARTITION_CONFIG_H_
