#!/bin/bash
set -e

# 1. Install containerd & git
apt-get update && apt-get install -y git containerd curl

# 2. Load Bridge Module (FIXES YOUR ERROR)
modprobe br_netfilter

# 3. Enable Kernel Forwarding
sysctl -w net.ipv4.ip_forward=1
sysctl -w net.bridge.bridge-nf-call-iptables=1

# 4. Install faasd using the official hack script
# This script automatically downloads the binary, CNI plugins, and sets up systemd
cd /tmp
git clone https://github.com/openfaas/faasd --depth=1
cd faasd
./hack/install.sh

# 5. Install CLI
curl -sL https://cli.openfaas.com | sh

# 6. Wait & Print Password
echo "Waiting for faasd to start..."
sleep 15
cat /var/lib/faasd/secrets/basic-auth-password
echo ""