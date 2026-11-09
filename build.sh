#!/usr/bin/env sh
# Build the Rust core to WebAssembly and put it next to the web host.
set -e
rustup target add wasm32-unknown-unknown
cargo build --release --lib --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/swarm.wasm web/swarm.wasm
echo "built web/swarm.wasm ($(wc -c < web/swarm.wasm) bytes)"
echo "now run:  python3 -m http.server -d web 8080   and open http://localhost:8080"
