# TODO — Tensor Algebra in Rust

### P0 — Polish and correctness
- [X] README examples list matches actual files:
  - Replace with: `examples/arrays_basic.rs`, `examples/matrices_basic.rs`, `examples/matmul_basic.rs`.
- [X] Fix label in `examples/matrices_basic.rs` to reflect the actual vector:
  - Current prints `"a * [1, 0, -1]^T"`, but `v = vector![1, -1]`.
  - Change to: `println!("a * [1, -1]^T = {:?}", mv);`
- [X] Add crate metadata in `Cargo.toml`: `description`, `license = "MIT"`, `repository`, `readme = "README.md"`, `keywords`, `categories`.
- [X] `rustfmt` + `clippy -D warnings` clean.
- [X] Implement `Display` for `Vector/Matrix/Tensor`.
- [X] Add `Index<(usize, usize)>` for `Matrix` and helpers `get(row, col)` / `get_mut(row, col)`.
- [X] Add `prelude` module re-exports (types + macros).
- [X] Module docs + doctests for `tensor`, `arithmetic`, `macros`.

### P0 — Core API additions
- [X] `Matrix::transpose()` (allocating; view-based later).
- [X] Iterators and views:
  - [X] `row(i)`, `col(j)` accessors; row/col iterators.
  - [X] `map`, `zip_map` on `Vector/Matrix`.
- [X] Reductions:
  - [X] `sum`, `mean`, `max`, `min`, `argmax` for `Vector/Matrix`.
- [X] Matrix Hadamard product.
- [ ] Vector `outer` product; norms (`l1`, `l2`, `linf`) and normalization.
- [ ] `trace()` for square `Matrix`.

### P1 — Tensor features
- [ ] `Tensor::reshape`, `permute_axes`, `expand_dims`, `squeeze`.
- [ ] Slicing/views with ranges and steps; non-owning sub-tensors.
- [ ] Axis-wise reductions: `sum(axis)`, `mean(axis)`, `max(axis)` with `keepdims`.

### P1 — Performance
- [ ] Blocked/tiled matmul (cache-friendly).
- [ ] Optional parallelism with Rayon (feature flag).
- [ ] SIMD inner loops using `std::simd` (feature flag).
- [ ] Optional BLAS backend for `f32/f64` paths (feature flag, pure-Rust default).

### P1 — Data layout
- [ ] Add contiguous storage + strides representation for `Matrix/Tensor` to enable slicing and views.
- [ ] Conversions between current row-of-vectors layout and contiguous (for migration).

### P1 — Interop and traits
- [ ] Integrate `num-traits` (`Zero`, `One`, `Num`, `Float`); keep `AllowedNumericTypes` as thin wrapper if needed.
- [ ] `approx` for epsilon comparisons; use in tests.
- [ ] `serde` (feature-gated) for `Serialize/Deserialize`.
- [ ] Conversions: `From<Vec<Vec<T>>>`, `TryFrom<&[&[T]]>`, and interop with `ndarray`/`nalgebra` behind features.
- [ ] Trait impls: `AddAssign/SubAssign/MulAssign`, `IntoIterator`/`FromIterator`, `AsRef<[T]>` for `Vector`, reference-friendly ops (`&A + &B`).

### P2 — Linear algebra
- [ ] Determinant and LU decomposition (square matrices).
- [ ] Inverse (via LU) with clear error semantics.
- [ ] Linear solves (`Ax=b`) with LU/QR; start with small/moderate sizes.

### Tests
- [ ] Unit tests for all new APIs (transpose, indexing, reductions, Hadamard, outer, norms, trace).
- [ ] Error/pathology tests:
  - [ ] Dimension mismatches in add/mul/matvec and new ops.
  - [ ] Out-of-bounds indexing for new indexers.
  - [ ] Division-by-zero coverage for scalar and elementwise.
- [ ] Property tests (`proptest`):
  - [ ] Additive identities/associativity where valid, distributivity, dot properties (bilinear, symmetry).
  - [ ] Shape laws for matmul and broadcasting.
- [ ] Randomized numeric tests comparing:
  - [ ] Matmul vs a naive reference (small sizes).
  - [ ] Reductions vs iterators over raw data.
- [ ] Stability tests for `f32/f64` (catastrophic cancellation cases).
- [ ] Fuzzing (`cargo-fuzz`) for indexing/slicing and matmul shape handling.
- [ ] Run Miri in CI to catch UB.

### Benchmarks
- [ ] Set up Criterion:
  - [ ] Vector elementwise ops.
  - [ ] Matvec and matmul for sizes: 32/64/128/256; square and rectangular.
  - [ ] Compare naive vs blocked vs rayon vs BLAS (when enabled).
- [ ] Document default tile sizes and their impact.

### CI and tooling
- [ ] GitHub Actions:
  - [ ] fmt, clippy, test (stable + beta).
  - [ ] miri (nightly job).
  - [ ] doc build (`cargo doc`).
  - [ ] Optional benches job (upload artifacts).
- [ ] Coverage (`tarpaulin` or grcov) badge.
- [ ] `cargo-deny` for dependency audits.

### Feature flags (suggested)
- [ ] `rayon`: parallel loops for heavy kernels.
- [ ] `simd`: `std::simd` accelerated inner loops.
- [ ] `blas`: cblas/OpenBLAS backend for matmul.
- [ ] `serde`: serialization.
- [ ] `approx`: epsilon-based comparisons.
- [ ] `no_std`: support `alloc` when feasible.

Example `Cargo.toml` additions (sketch):
```toml
[features]
default = []
rayon = ["dep:rayon"]
simd = []
blas = ["dep:cblas-sys", "dep:openblas-src"]
serde = ["dep:serde"]
approx = ["dep:approx"]

[dependencies]
num-traits = "0.2"
serde = { version = "1", features = ["derive"], optional = true }
approx = { version = "0.5", optional = true }
rayon = { version = "1.8", optional = true }
cblas-sys = { version = "0.1", optional = true }
openblas-src = { version = "0.10", optional = true, default-features = false, features = ["system"] }
```

### Milestones
- **v0.1.0**: P0 polish, transpose/indexing, reductions, Criterion benches, CI.
- **v0.2.0**: Views/slices, broadcasting, hadamard/outer/norms/trace.
- **v0.3.0**: Blocked matmul, Rayon/SIMD features, BLAS optional.
- **v1.0.0**: LA basics (LU/det/inv/solve), contiguous layout + views stabilized, docs site.
