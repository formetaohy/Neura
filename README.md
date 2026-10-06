<p align="center">
  <img src="docs/assets/neura-readme-banner-animated.svg" width="784" alt="Neura">
</p>

<p align="center">
  <a href="https://crates.io/crates/neura"><img src="https://img.shields.io/crates/v/neura.svg?color=blue" alt="Crate version"></a>
  <a href="https://github.com/formetaohy/Neura/actions/workflows/ci.yml"><img src="https://github.com/formetaohy/Neura/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI status"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/language-Rust-orange?logo=rust" alt="Rust"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT license"></a>
</p>

## What is Neura?

Neura is a **deep learning framework designed for games**, making it easy to build **AI-native games**. It uses your game's native graphics API, **without CUDA**.

## Features

- **Game-loop Integration**: Run training and inference directly from your frame loop, without async/await.
- **Cross-platform Deployment**: Use your game's native graphics API directly — Neura runs right alongside your game.
- **Kernel Compiler**: Write custom kernels in Rust; Neura generates SPIR-V, HLSL, and MSL, so you do not maintain separate shader implementations.

## Document

**[Examples](examples/)**: Runnable examples for learning Neura and exploring specific concepts.

## Contact

Email: formetaohy@gmail.com

## License

[MIT](LICENSE)
