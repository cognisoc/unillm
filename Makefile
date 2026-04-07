.PHONY: check test build fmt clippy clean release

check:
	cargo check --workspace

test:
	cargo test --workspace

build:
	cargo build --workspace

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --workspace -- -D warnings

clean:
	cargo clean

release:
	cargo build --release --workspace
