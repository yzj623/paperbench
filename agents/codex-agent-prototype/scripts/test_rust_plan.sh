#!/usr/bin/env bash
# Tests the real generic prefix helper with lightweight stubs, no inference.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -x "$PROJECT_ROOT/.build-tools/cargo/bin/cargo" ]]; then
  export RUSTUP_HOME="$PROJECT_ROOT/.build-tools/rustup"
  export CARGO_HOME="$PROJECT_ROOT/.build-tools/cargo"
  export PATH="$CARGO_HOME/bin:$PATH"
fi
cd "$PROJECT_ROOT"
export CARGO_TARGET_DIR="$PROJECT_ROOT/build/plan-tests"
cargo +1.95.0 test --locked --manifest-path tests/rust-plan-harness/Cargo.toml
