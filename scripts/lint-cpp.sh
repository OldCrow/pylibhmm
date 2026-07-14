#!/usr/bin/env bash
# scripts/lint-cpp.sh
# Static analysis for the C++ nanobind binding layer (src/pylibhmm/_core.cpp,
# src/pylibhmm/_common.h): cppcheck warning/style/performance checks.
#
# This is its own invocation, not a copy of libhmm's scripts/-equivalent
# cppcheck command, for two concrete reasons found while setting this up:
#   1. libhmm's invocation relies on file-extension inference for C++ (it
#      only scans .cpp files under src/). _common.h is a header; without
#      --language=c++ explicit, cppcheck misparses it as C and raises a
#      false `namespace libhmm {` syntaxError.
#   2. Without a suppression scoped to libhmm's own include path, cppcheck
#      attributes findings to libhmm's headers (reached via -I) that are
#      libhmm's concern, not pylibhmm's.
#
# The suppression list otherwise matches libhmm's for consistency.
#
# Usage:
#   ./scripts/lint-cpp.sh
#
# Prerequisites:
#   - cppcheck on PATH
#   - a local ../libhmm checkout (for its headers; matches the same
#     dependency-resolution preference used by the CMake build)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIBHMM_INCLUDE="$REPO_ROOT/../libhmm/include"

if [[ ! -d "$LIBHMM_INCLUDE" ]]; then
    echo "ERROR: $LIBHMM_INCLUDE not found — this script expects a local ../libhmm checkout."
    exit 1
fi

echo ""
echo "==> Running cppcheck (src/pylibhmm/_core.cpp, src/pylibhmm/_common.h)..."

cppcheck --enable=warning,style,performance --error-exitcode=1 \
    --suppress=missingIncludeSystem --suppress=useStlAlgorithm \
    --suppress=shadowFunction --suppress=virtualCallInConstructor \
    --suppress=constParameterReference --suppress=noExplicitConstructor \
    --suppress=toomanyconfigs --suppress=functionStatic \
    --suppress="*:*/libhmm/include/*" \
    --std=c++20 --language=c++ \
    -I "$LIBHMM_INCLUDE" -I "$REPO_ROOT/src/pylibhmm" \
    "$REPO_ROOT/src/pylibhmm/_core.cpp" "$REPO_ROOT/src/pylibhmm/_common.h"

echo ""
echo "==> cppcheck clean."
