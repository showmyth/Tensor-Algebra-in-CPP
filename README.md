# Tensor Algebra in Rust

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=flat&logo=rust&logoColor=white)](https://www.rust-lang.org/)

> **Note:** This is a personal project collaboration between [@showmyth](https://github.com/showmyth) and [@IrregularPersona](https://github.com/IrregularPersona)

A foundational library for performing basic algebraic operations on N-dimensional numerical data structures (Arrays, Matrices, and Tensors) in Rust.

## Overview

Tensor Algebra provides efficient implementations for working with:
- **Vectors** (1D arrays)
- **Matrices** (2D arrays) 
- **Tensors** (N-dimensional arrays)

All operations are aimed to be fast and safe whilst using Rust's powerful type system.

## Features

### Current
- **Generic Numeric Types**: Full support for `i32/64`, `u32/64`, `f32/64`
- **Type Safety**
- **Zero-Copy Operations**: Efficient memory usage through references and views
- **Seamless integration with Rust's Iterator ecosystem**

### Upcoming
- **Basic Arithmetic**: Element-wise addition, subtraction, multiplication, and division
- **Scalar Operations**: Broadcasting scalar values across tensors
- **Linear Algebra**: Matrix multiplication, transpose, determinant, and inverse operations
- **Advanced Operations**: Dot products, cross products, and norm calculations

##  Quick Start

### Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
tensor-algebra = "0.1.0"
```

##  Performance

This library is designed with performance in mind:
- Zero-cost abstractions
- Minimal allocations
- Cache-friendly memory layouts

## Development Status

This project is in active development. Current focus areas:

- Core tensor data structures
- Basic arithmetic operations
- Linear algebra foundations
- Comprehensive test suite
- Documentation and examples

## Contributing

**Status**: Pull requests are currently closed while we establish the core architecture.

We welcome contributions! Once we open contributions, please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure all tests pass (`cargo test`)
5. Update documentation as needed
6. Submit a pull request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/showmyth/Tensor-Algebra-in-Rust.git
cd Tensor-Algebra-in-Rust

# Run tests
cargo test

# Run benchmarks
cargo bench

# Generate documentation
cargo doc --open
```

## Architecture

The library is structured around three core abstractions:

- **`Tensor<T, const N: usize>`**: N-dimensional arrays with compile-time dimension checking
- **`Matrix<T>`**: Specialized 2D operations with linear algebra support
- **`Vector<T>`**: 1D operations with vector space operations

## Examples

Check out the `examples/` directory for comprehensive usage examples:

- `arrays_basics.rs` 
- `matmul_basics.rs` 
- `matrices_basics.rs` 


## Documentation

Coming soon!

## Roadmap

### Version 0.0.0 (Current)
- Basic data structures
- Memory layout optimization
- Type system foundation

### Version 0.1.0
- Arithmetic operations
- Basic linear algebra
- Comprehensive benchmarks


## License(s)

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## Acknowledgments

- Inspired by NumPy and similar tensor libraries
- Built using Rust's powerful type system
- Thanks to the Rust community for excellent documentation and tools

---

**Stay tuned for upcoming announcements!** 👀

For questions or suggestions, feel free to open an issue or reach out to the maintainers.
