#!/bin/bash
set -euo pipefail

# --- Configuration ---
PROTOBUF_VERSION="v26.1"
GRPC_VERSION="v1.64.0"
INSTALL_DIR="/usr/local"
BUILD_DIR="/tmp/protobuf_grpc_build"

echo "=== 🚀 Starting Setup: Protobuf ($PROTOBUF_VERSION) and gRPC ($GRPC_VERSION) ==="
sudo apt update
sudo apt install -y protobuf-compiler libprotobuf-dev libgrpc++-dev

# --- 1. System Update and Dependencies ---
echo "--- 1. Installing Prerequisites ---"
sudo apt update
sudo apt upgrade -y

echo "Installing build essentials..."
sudo apt install -y build-essential cmake git pkg-config curl \
    autoconf libtool

# Cleanup and Prepare Build Directory
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# -----------------------------------------------------------------------------

# --- 2. Installing Protocol Buffers (Protobuf) via CMake ---
echo "--- 2. Building Protobuf (CMake) ---"

# Clone Protobuf
if [ ! -d "protobuf" ]; then
    git clone https://github.com/protocolbuffers/protobuf.git
fi
cd protobuf
git checkout "$PROTOBUF_VERSION"
git submodule update --init --recursive

# Safe directory config
git config --global --add safe.directory "$PWD"

# Build using CMake (Fixes the missing ./configure issue)
mkdir -p cmake_build
cd cmake_build
cmake .. \
    -DCMAKE_CXX_STANDARD=17 \
    -Dprotobuf_BUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

make -j$(nproc)
sudo make install
sudo ldconfig

cd "$BUILD_DIR" # Return to base

# Verify Protobuf
echo "Verifying Protoc..."
protoc --version

# -----------------------------------------------------------------------------

# --- 3. Installing gRPC via CMake ---
echo "--- 3. Building gRPC (CMake) ---"

# Clone gRPC
if [ ! -d "grpc" ]; then
    git clone -b "$GRPC_VERSION" https://github.com/grpc/grpc
fi
cd grpc
git checkout "$GRPC_VERSION"
git submodule update --init --recursive

# Build using CMake
mkdir -p cmake_build
cd cmake_build
cmake .. \
    -DCMAKE_CXX_STANDARD=17 \
    -DgRPC_INSTALL=ON \
    -DgRPC_BUILD_TESTS=OFF \
    -DgRPC_ABSL_PROVIDER=module \
    -DgRPC_PROTOBUF_PROVIDER=package \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

make -j$(nproc)
sudo make install
sudo ldconfig

cd "$BUILD_DIR"

# -----------------------------------------------------------------------------

# --- 4. Final Verification ---
echo "--- 4. Verification ---"

if command -v grpc_cpp_plugin &> /dev/null; then
    echo "✅ SUCCESS: grpc_cpp_plugin found at $(which grpc_cpp_plugin)"
else
    echo "❌ ERROR: grpc_cpp_plugin not found."
    exit 1
fi
echo "=== ✨ cpp Setup Complete! ==="

# --- 5. Installing Go Toolchain ---
echo "--- 5. Installing Go Toolchain and Environment ---"

# Set the specified Go version (1.24.5 as requested)
GO_VERSION="1.24.5" 
GO_TARBALL="go$GO_VERSION.linux-amd64.tar.gz"
GO_URL="https://golang.org/dl/$GO_TARBALL"

# Download and Install Go
curl -L -o /tmp/$GO_TARBALL "$GO_URL"
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf /tmp/$GO_TARBALL

# Add Go binaries to the PATH for the current session and for all users
echo 'export PATH=$PATH:/usr/local/go/bin' | sudo tee /etc/profile.d/go-path.sh
source /etc/profile.d/go-path.sh

# Verify Go installation
echo "Verifying Go version..."
go version

# -----------------------------------------------------------------------------

# --- 6. Installing Go gRPC Plugins ---
echo "--- 6. Installing Go gRPC Protobuf Plugins ---"

# These tools are essential for generating Go code from your .proto files.

# Set GOPATH environment variable and add $GOPATH/bin to PATH
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin

# 6a. Install the Protobuf compiler plugin for Go
echo "Installing protoc-gen-go..."
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest

# 6b. Install the gRPC service code generator plugin for Go
echo "Installing protoc-gen-go-grpc..."
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Verify Go plugins
if command -v protoc-gen-go &> /dev/null; then
    echo "✅ SUCCESS: protoc-gen-go found at $(which protoc-gen-go)"
else
    echo "❌ ERROR: protoc-gen-go not found."
    exit 1
fi

if command -v protoc-gen-go-grpc &> /dev/null; then
    echo "✅ SUCCESS: protoc-gen-go-grpc found at $(which protoc-gen-go-grpc)"
else
    echo "❌ ERROR: protoc-gen-go-grpc not found."
    exit 1
fi

# --- 7. Final Clean-up ---
echo "--- 7. Final Clean-up ---"
# Note: Assuming $BUILD_DIR was set earlier in the script (e.g., /tmp/protobuf_grpc_build)
rm -rf "$BUILD_DIR" 
rm -f /tmp/$GO_TARBALL

echo "=== ✨ Go Toolchain Setup Complete! ==="
