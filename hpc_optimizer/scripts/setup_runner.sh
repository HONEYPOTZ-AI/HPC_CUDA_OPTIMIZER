#!/bin/bash
# =============================================================================
# setup_runner.sh — Configure a GitHub Actions self-hosted runner on a SLURM node
#
# Run this script ONCE on the SLURM login/runner node as the CI service user.
# It installs the GitHub Actions runner agent, registers it with your repo,
# and installs a systemd unit + SLURM job-forwarding wrapper.
#
# Prerequisites:
#   - A GitHub Personal Access Token (PAT) or Fine-Grained Token with
#     repo scope (Settings → Developer Settings → Personal Access Tokens)
#   - CUDA toolkit, NCCL, OpenMPI already installed on compute nodes
#   - The runner node can submit SLURM jobs (squeue, sbatch, etc.)
#   - sudo / systemd access on the runner host
#
# Usage:
#   export GH_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
#   export GH_REPO="your-org/hpc-optimizer"       # <owner>/<repo>
#   export RUNNER_NAME="slurm-gpu-runner-01"       # Unique runner name
#   export RUNNER_LABELS="self-hosted,slurm,gpu"   # Tags visible in ci.yml
#   bash scripts/setup_runner.sh
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
GH_TOKEN="${GH_TOKEN:?Set GH_TOKEN to your GitHub PAT}"
GH_REPO="${GH_REPO:?Set GH_REPO to owner/repo}"
RUNNER_NAME="${RUNNER_NAME:-slurm-gpu-runner-$(hostname -s)}"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted,slurm,gpu,linux,x64}"
RUNNER_VERSION="${RUNNER_VERSION:-2.317.0}"
RUNNER_INSTALL_DIR="${RUNNER_INSTALL_DIR:-${HOME}/actions-runner}"
RUNNER_WORK_DIR="${RUNNER_WORK_DIR:-${HOME}/actions-runner/_work}"
RUNNER_USER="${RUNNER_USER:-$(whoami)}"
SERVICE_NAME="github-runner-$(echo "${RUNNER_NAME}" | tr '/' '-')"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     GitHub Actions Self-Hosted Runner Setup (SLURM)         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Repo        : https://github.com/${GH_REPO}"
echo "  Runner name : ${RUNNER_NAME}"
echo "  Labels      : ${RUNNER_LABELS}"
echo "  Install dir : ${RUNNER_INSTALL_DIR}"
echo "  Service     : ${SERVICE_NAME}"
echo ""

# ── 1. Install system dependencies ───────────────────────────────────────────
echo "[1/7] Checking system dependencies..."

MISSING=()
for cmd in curl tar jq cmake nvcc mpirun; do
    if ! command -v "$cmd" &>/dev/null; then
        MISSING+=("$cmd")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "  WARNING: Missing commands: ${MISSING[*]}"
    echo "  Attempting to install via apt..."
    sudo apt-get update -qq
    sudo apt-get install -y curl tar jq cmake 2>/dev/null || true
fi

echo "  CUDA    : $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',')"
echo "  CMake   : $(cmake --version | head -1)"
echo "  MPI     : $(mpirun --version 2>/dev/null | head -1 || echo 'not found')"
echo ""

# ── 2. Download GitHub Actions runner ────────────────────────────────────────
echo "[2/7] Downloading GitHub Actions runner v${RUNNER_VERSION}..."
mkdir -p "${RUNNER_INSTALL_DIR}"
cd "${RUNNER_INSTALL_DIR}"

ARCH=$(uname -m)
case "${ARCH}" in
    x86_64)  RUNNER_ARCH="x64" ;;
    aarch64) RUNNER_ARCH="arm64" ;;
    *)        echo "Unsupported arch: ${ARCH}"; exit 1 ;;
esac

RUNNER_TARBALL="actions-runner-linux-${RUNNER_ARCH}-${RUNNER_VERSION}.tar.gz"
RUNNER_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_TARBALL}"

if [ ! -f "${RUNNER_TARBALL}" ]; then
    curl -sSL -o "${RUNNER_TARBALL}" "${RUNNER_URL}"
    tar xzf "${RUNNER_TARBALL}"
    echo "  Downloaded and extracted runner v${RUNNER_VERSION} (${RUNNER_ARCH})"
else
    echo "  Runner tarball already present — skipping download"
fi

# ── 3. Get registration token ─────────────────────────────────────────────────
echo ""
echo "[3/7] Fetching registration token from GitHub API..."
REG_TOKEN=$(curl -sSL \
    -X POST \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${GH_TOKEN}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/${GH_REPO}/actions/runners/registration-token" \
    | jq -r '.token')

if [ -z "${REG_TOKEN}" ] || [ "${REG_TOKEN}" = "null" ]; then
    echo "  ERROR: Could not obtain registration token. Check your GH_TOKEN and repo name."
    exit 1
fi
echo "  Registration token obtained ✓"

# ── 4. Configure runner ───────────────────────────────────────────────────────
echo ""
echo "[4/7] Configuring runner..."
cd "${RUNNER_INSTALL_DIR}"
mkdir -p "${RUNNER_WORK_DIR}"

./config.sh \
    --url "https://github.com/${GH_REPO}" \
    --token "${REG_TOKEN}" \
    --name "${RUNNER_NAME}" \
    --labels "${RUNNER_LABELS}" \
    --work "${RUNNER_WORK_DIR}" \
    --runnergroup "Default" \
    --unattended \
    --replace

echo "  Runner configured ✓"

# ── 5. Install environment shim ───────────────────────────────────────────────
# The runner executes jobs in a shell without module system. We source
# a common environment script so nvcc, mpirun, etc. are always on PATH.

echo ""
echo "[5/7] Writing environment shim..."
cat > "${RUNNER_INSTALL_DIR}/.env" <<'ENVEOF'
# GitHub Actions runner environment — sourced before each job
# Adjust module names to match your cluster's module system

# CUDA
CUDA_ROOT=/usr/local/cuda
export PATH="${CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${CUDA_ROOT}"

# NCCL (adjust to actual path)
NCCL_ROOT="${NCCL_ROOT:-/usr/local/nccl}"
export LD_LIBRARY_PATH="${NCCL_ROOT}/lib:${LD_LIBRARY_PATH}"
export NCCL_HOME="${NCCL_ROOT}"

# OpenMPI (adjust to actual path)
MPI_ROOT="${MPI_ROOT:-/usr/local/openmpi}"
export PATH="${MPI_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${MPI_ROOT}/lib:${LD_LIBRARY_PATH}"

# CMake (if installed outside system path)
# export PATH="/opt/cmake/bin:${PATH}"

# SLURM environment (already set by host, but ensure visible)
export SLURM_CONF="${SLURM_CONF:-/etc/slurm/slurm.conf}"
ENVEOF

echo "  Environment shim written to ${RUNNER_INSTALL_DIR}/.env"

# ── 6. Install systemd service ────────────────────────────────────────────────
echo ""
echo "[6/7] Installing systemd service..."

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

sudo tee "${SERVICE_FILE}" > /dev/null <<SVCEOF
[Unit]
Description=GitHub Actions Runner (${RUNNER_NAME})
After=network-online.target slurmctld.service
Wants=network-online.target

[Service]
Type=simple
User=${RUNNER_USER}
WorkingDirectory=${RUNNER_INSTALL_DIR}
EnvironmentFile=${RUNNER_INSTALL_DIR}/.env
ExecStart=${RUNNER_INSTALL_DIR}/run.sh
Restart=on-failure
RestartSec=10
KillMode=process
KillSignal=SIGTERM
TimeoutStopSec=5min

# Resource limits — runner process itself is lightweight; GPU jobs run via SLURM
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
SVCEOF

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl start "${SERVICE_NAME}"

sleep 2
STATUS=$(sudo systemctl is-active "${SERVICE_NAME}" 2>/dev/null || echo "unknown")
echo "  Service ${SERVICE_NAME}: ${STATUS}"

# ── 7. Validate ───────────────────────────────────────────────────────────────
echo ""
echo "[7/7] Validating runner registration..."

RUNNER_STATUS=$(curl -sSL \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${GH_TOKEN}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/${GH_REPO}/actions/runners" \
    | jq -r --arg name "${RUNNER_NAME}" '.runners[] | select(.name==$name) | .status')

if [ "${RUNNER_STATUS}" = "online" ]; then
    echo "  Runner '${RUNNER_NAME}' is ONLINE ✓"
else
    echo "  Runner '${RUNNER_NAME}' status: ${RUNNER_STATUS:-not found}"
    echo "  Check logs: sudo journalctl -u ${SERVICE_NAME} -f"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Useful commands:"
echo "    sudo systemctl status ${SERVICE_NAME}"
echo "    sudo journalctl -u ${SERVICE_NAME} -f"
echo "    sudo systemctl restart ${SERVICE_NAME}"
echo ""
echo "  To add more GPU labels (e.g. multi-gpu), re-run with:"
echo "    RUNNER_LABELS='self-hosted,slurm,gpu,multi-gpu' bash $0"
echo ""
echo "  Runner work dir: ${RUNNER_WORK_DIR}"
echo "  Test logs dir  : ${RUNNER_WORK_DIR}/../logs"
