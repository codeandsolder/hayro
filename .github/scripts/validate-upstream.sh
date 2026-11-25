#!/usr/bin/env bash
set -euo pipefail
cargo fmt --all -- --check
cargo check --locked --workspace --all-targets
cargo test --locked -p hayro-syntax --all-features
cargo test --locked -p hayro-interpret --all-features
cargo check --locked -p hayro-syntax --no-default-features
cargo check --locked -p hayro-syntax --no-default-features --features images
