# MNIST

Trains a convolutional network on handwritten digits and classifies held out ones, printing six of them as ASCII art beside the guessed number. The first run downloads about 11 MB of archives into `target/mnist`; point `NEURA_MNIST_DIR` at a directory holding those four archives to skip it.

Run from the repository root:

```sh
cargo run --release --example mnist
```
