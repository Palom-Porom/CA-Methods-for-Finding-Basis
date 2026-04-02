#!/usr/bin/env bash
# build.sh — configure, build, and run gmres_test
#
# Usage:
#   ./build.sh              — STL backend (default)
#   ./build.sh --blas       — BLAS backend
#   ./build.sh --clean      — wipe build dir first
#   ./build.sh --clean --blas
#   ./build.sh --no-run     — build only, don't execute

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="$ROOT/build"
SRC="$ROOT/src/gmres"

USE_BLAS=OFF
CLEAN=0
RUN=1

for arg in "$@"; do
    case "$arg" in
        --blas)   USE_BLAS=ON ;;
        --clean)  CLEAN=1 ;;
        --no-run) RUN=0 ;;
        -h|--help)
            echo "Usage: $0 [--blas] [--clean] [--no-run]"
            exit 0 ;;
        *)
            echo "Unknown option: $arg"; exit 1 ;;
    esac
done

# ── Clean ────────────────────────────────────────────────────────────────────
if [[ $CLEAN -eq 1 && -d "$BUILD" ]]; then
    echo "» Removing $BUILD"
    rm -rf "$BUILD"
fi

# ── Configure (only when CMakeCache is absent or USE_BLAS changed) ───────────
CACHE="$BUILD/CMakeCache.txt"
NEEDS_CONFIGURE=1

if [[ -f "$CACHE" ]]; then
    cached=$(grep -m1 'USE_BLAS:BOOL=' "$CACHE" 2>/dev/null | cut -d= -f2 || true)
    [[ "$cached" == "$USE_BLAS" ]] && NEEDS_CONFIGURE=0
fi

if [[ $NEEDS_CONFIGURE -eq 1 ]]; then
    echo "» Configuring  (USE_BLAS=$USE_BLAS)"
    cmake -S "$SRC" -B "$BUILD" -G Ninja \
          -DCMAKE_BUILD_TYPE=Release \
          -DUSE_BLAS="$USE_BLAS"
fi

# ── Build ────────────────────────────────────────────────────────────────────
echo "» Building"
cmake --build "$BUILD" --parallel

# ── Run ──────────────────────────────────────────────────────────────────────
if [[ $RUN -eq 1 ]]; then
    echo "» Running gmres_test"
    echo ""
    "$BUILD/gmres_test.exe"
fi
