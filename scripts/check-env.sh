#!/usr/bin/env bash
# wormuse — environment sanity check.
# Run from the repo root.

set -e

echo "=== wormuse env check ==="

# 1) Docker
if ! command -v docker >/dev/null 2>&1; then
  echo "FAIL: docker not in PATH"; exit 1
else
  echo "ok docker $(docker --version)"
fi

# 2) AMSC MK image
if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q '^quay.io/pjbaioni/amsc_mk:2025$'; then
  echo "ok MK image present"
else
  echo "WARN: quay.io/pjbaioni/amsc_mk:2025 not pulled. Run: docker pull quay.io/pjbaioni/amsc_mk:2025"
fi

# 3) OpenWorm image
if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q '^openworm/openworm:latest$'; then
  echo "ok OpenWorm image present"
else
  echo "WARN: openworm/openworm:latest not pulled. Run: docker pull openworm/openworm:latest"
fi

# 4) docker compose
if docker compose version >/dev/null 2>&1; then
  echo "ok $(docker compose version)"
else
  echo "FAIL: docker compose plugin missing"; exit 1
fi

# 5) Python (for PyANNOW + wormuse-analytics)
if command -v python3 >/dev/null 2>&1; then
  PYVER=$(python3 --version)
  echo "ok $PYVER"
  python3 -c "import sys; assert sys.version_info >= (3,10), 'need >=3.10'" 2>/dev/null \
    || echo "WARN: Python $PYVER < 3.10, PyANNOW requires 3.10+"
else
  echo "WARN: python3 not in PATH (only required for PyANNOW + wormuse-analytics)"
fi

# 6) Quick MK toolchain smoke
echo
echo "=== MK toolchain smoke test (in container) ==="
docker run --rm quay.io/pjbaioni/amsc_mk:2025 bash -c '
  source /u/sw/etc/profile.d/mk.sh 2>/dev/null
  module load gcc-glibc/11.2.0 eigen dealii 2>&1 | tail -1
  module list 2>&1 | tail -3
' 2>/dev/null || echo "WARN: could not exercise MK image"

# 7) OpenWorm container smoke
echo
echo "=== OpenWorm smoke ==="
docker run --rm openworm/openworm:latest bash -c '
  ls -d $C302_HOME $SIBERNETIC_HOME 2>&1
  python3 --version 2>&1
' 2>/dev/null || echo "WARN: could not exercise OpenWorm image"

echo
echo "Environment check complete."
