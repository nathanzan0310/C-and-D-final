#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# --------------------------------------------------
# Step 1: Clone the Junction repository
# --------------------------------------------------
echo "Cloning the Junction repository..."
git clone https://github.com/JunctionOS/junction.git  
cd junction

# --------------------------------------------------
# Step 2: Install required packages and build dependencies
# --------------------------------------------------
echo "Installing dependencies and building..."
scripts/install.sh

# --------------------------------------------------
# Step 3: Install Rust (nightly toolchain)
# --------------------------------------------------
echo "Installing Rust nightly toolchain..."
curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain=nightly
export PATH="$HOME/.cargo/bin:$PATH"  # Ensure Rust is in PATH

# --------------------------------------------------
# Step 4: Compile Junction
# --------------------------------------------------
echo "Building Junction..."
scripts/build.sh

# --------------------------------------------------
# Step 5: Start the core scheduler in a separate terminal
# --------------------------------------------------
echo "Starting core scheduler..."
gnome-terminal -- bash -c "sudo lib/caladan/scripts/setup_machine.sh; sudo lib/caladan/iokerneld ias; exec bash"

# --------------------------------------------------
# Step 6: Run a sample Junction container
# --------------------------------------------------
echo "Running sample Junction container..."
cd build/junction
./junction_run caladan_test.config -- /usr/bin/openssl speed

# host_addr 192.168.127.7
# host_netmask 255.255.255.0
# host_gateway 192.168.127.1
# runtime_kthreads 10
# runtime_spinning_kthreads 0
# runtime_guaranteed_kthreads 0
# runtime_priority lc
# runtime_quantum_us 0
echo "Junction setup and sample run complete!"
