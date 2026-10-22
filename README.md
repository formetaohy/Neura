<p align="center">
  <img src="docs/assets/neura-readme-banner-animated.svg" width="784" alt="Neura">
</p>

<p align="center">
  <strong>Train and run AI in your game.</strong>
</p>

<p align="center">
  <a href="https://crates.io/crates/neura"><img src="https://img.shields.io/crates/v/neura.svg?color=blue" alt="Crate version"></a>
  <a href="https://github.com/formetaohy/Neura/actions/workflows/ci.yml"><img src="https://github.com/formetaohy/Neura/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI status"></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/language-Rust-orange?logo=rust" alt="Rust"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT license"></a>
</p>

## What is Neura?

Neura is an **AI framework designed for games**. It lets you **train and run models such as Qwen locally inside your game**, using the game's native graphics API—**no CUDA required**.

## Features

- **High Performance**: Designed for local devices with limited VRAM and compute.
- **Game-loop Integration**: Run training and inference directly from your frame loop, without async/await.
- **Cross-platform Deployment**: Neura runs wherever your game runs, using the same native graphics API.
- **Kernel Compiler**: Custom kernels written in Rust are compiled by Neura's built-in AOT compiler into SPIR-V, HLSL, and MSL.

## Documents

- **[Examples](examples)**: Runnable examples for learning Neura and exploring specific concepts.

## Integrations

- **[bevy-neura](https://github.com/formetaohy/bevy-neura)**: An official integration for building AI-native games in [Bevy](https://bevy.org).

## Contact

Email: formetaohy@gmail.com

## License

[MIT](LICENSE)
