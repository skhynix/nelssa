#!/bin/bash
# Build script for C++ Clustering Extension
# This compiles the multi-threaded CPU reorganization kernel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Building C++ Clustering Extension"
echo "=========================================="

# Check if gcc with OpenMP is available
if ! gcc -fopenmp -E -x c - < /dev/null 2>/dev/null; then
    echo "WARNING: OpenMP not available, building without multi-threading"
    EXTRA_FLAGS=""
else
    echo "OpenMP detected, enabling multi-threaded reorganization"
    EXTRA_FLAGS="-fopenmp"
fi

# Build the extension
echo "Compiling clustering_cpu.cpp..."
python3 setup_clustering_ext.py build_ext --inplace

echo ""
echo "=========================================="
echo "Build complete!"
echo "Extension should be at: vllm_clustering_ext.so"
echo "=========================================="
echo ""
echo "To use the extension, restart your Python process."
echo "The extension will be automatically loaded by clustering.py"