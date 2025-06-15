#!/bin/bash

# Simple test to validate SECDA profiling setup
echo "=== SECDA-Lite Profiling Validation ==="

# Check if binaries exist
echo "Checking delegate binaries..."
BINARIES=(
    "bazel-bin/tensorflow/lite/delegates/utils/sa_sim_delegate/label_image_plus_sa_sim_delegate"
    "bazel-bin/tensorflow/lite/delegates/utils/vm_sim_delegate/label_image_plus_vm_sim_delegate"
    "bazel-bin/tensorflow/lite/delegates/utils/bert_sim_delegate/label_image_plus_bert_sim_delegate"
)

for binary in "${BINARIES[@]}"; do
    if [ -f "$binary" ]; then
        echo "✓ Found: $binary"
    else
        echo "✗ Missing: $binary"
    fi
done

# Check test files
echo -e "\nChecking test files..."
TEST_FILES=(
    "test_images/grace_hopper.bmp"
    "models/mobilenetv1.tflite"
    "models/mobilenetv2.tflite"
)

for file in "${TEST_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ Found: $file"
    else
        echo "✗ Missing: $file"
    fi
done

# Test SA sim delegate with profiling
echo -e "\n=== Testing SA Sim Delegate Profiling ==="
if [ -f "bazel-bin/tensorflow/lite/delegates/utils/sa_sim_delegate/label_image_plus_sa_sim_delegate" ] && [ -f "models/mobilenetv1.tflite" ]; then
    
    # Clean previous outputs
    rm -f outputs/sa_sim*.csv
    
    echo "Running SA sim delegate with MobileNetV1..."
    ./bazel-bin/tensorflow/lite/delegates/utils/sa_sim_delegate/label_image_plus_sa_sim_delegate \
        --image=test_images/grace_hopper.bmp \
        --model=models/mobilenetv1.tflite \
        --labels=imagenet_classification_eval_plus_sa_sim_delegate \
        --use_sa_sim_delegate=true \
        --verbose=1
    
    echo -e "\nChecking profiling outputs..."
    if [ -f "outputs/sa_sim.csv" ]; then
        echo "✓ SystemC profiling CSV generated"
        echo "Preview of sa_sim.csv:"
        head -n 5 outputs/sa_sim.csv
    else
        echo "✗ No sa_sim.csv found"
    fi
    
    if [ -f "outputs/sa_sim_model.csv" ]; then
        echo "✓ Model profiling CSV generated"
        echo "Preview of sa_sim_model.csv:"
        head -n 5 outputs/sa_sim_model.csv
    else
        echo "✗ No sa_sim_model.csv found"
    fi
    
else
    echo "Missing binary or model files for testing"
fi

echo -e "\n=== Profiling Validation Complete ==="
