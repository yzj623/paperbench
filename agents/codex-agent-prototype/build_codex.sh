#!/usr/bin/env bash
# Compile with an existing Rust 1.95.0 toolchain, or inside Docker. Never install host Rust.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_JOBS="${BUILD_JOBS:-1}"
MODE=host
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --docker) MODE=docker; shift ;;
    --jobs) BUILD_JOBS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Usage: $0 [--docker] [--jobs N] [--dry-run]" >&2; exit 2 ;;
  esac
done
[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo 'jobs must be a positive integer' >&2; exit 2; }
if [[ "$MODE" == docker ]]; then
  # Reuse the exact multistage image recipe; export only the standalone binary.
  command=(bash "$PROJECT_ROOT/build_codex_image.sh" --jobs "$BUILD_JOBS")
  if [[ "$DRY_RUN" == 1 ]]; then "${command[@]}" --dry-run; exit; fi
  "${command[@]}"
  mkdir -p "$PROJECT_ROOT/build/bin"
  container_id="$(docker create "${CODEX_IMAGE:-codex-paperbench:latest}")"
  trap 'docker rm "$container_id" >/dev/null' EXIT
  docker cp "$container_id:/opt/codex/bin/codex-app-server" "$PROJECT_ROOT/build/bin/codex-app-server"
else
  if [[ -x "$PROJECT_ROOT/.build-tools/cargo/bin/rustup" ]]; then
    export CARGO_HOME="$PROJECT_ROOT/.build-tools/cargo"
    export RUSTUP_HOME="$PROJECT_ROOT/.build-tools/rustup"
    export PATH="$CARGO_HOME/bin:$PATH"
  fi
  command=(cargo +1.95.0 build --locked -p codex-app-server --bin codex-app-server)
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'CARGO_BUILD_JOBS=%q CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 ' "$BUILD_JOBS"
    printf '%q ' "${command[@]}"; printf '\n'; exit
  fi
  command -v rustup >/dev/null || { echo 'Existing Rust 1.95.0 required; use --docker to avoid installing host Rust.' >&2; exit 1; }
  rustup run 1.95.0 rustc --version >/dev/null || { echo 'Rust 1.95.0 is not installed; use --docker.' >&2; exit 1; }
  export CARGO_BUILD_JOBS="$BUILD_JOBS" CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0
  # Optional isolated headers/libs prepared within this project; never install
  # development packages globally or change the system linker configuration.
  if [[ -f "$PROJECT_ROOT/.build-tools/sysroot/usr/include/openssl/ssl.h" ]]; then
    export OPENSSL_INCLUDE_DIR="$PROJECT_ROOT/.build-tools/sysroot/usr/include"
    export OPENSSL_LIB_DIR="$PROJECT_ROOT/.build-tools/sysroot/usr/lib/x86_64-linux-gnu"
  fi
  export CARGO_TARGET_DIR="$PROJECT_ROOT/build/cargo"
  cd "$PROJECT_ROOT/codex/codex-rs"
  "${command[@]}"
  mkdir -p "$PROJECT_ROOT/build/bin"
  cp "$CARGO_TARGET_DIR/debug/codex-app-server" "$PROJECT_ROOT/build/bin/codex-app-server"
fi
ln -sf codex-app-server "$PROJECT_ROOT/build/bin/codex-linux-sandbox"
ln -sf codex-app-server "$PROJECT_ROOT/build/bin/apply_patch"
