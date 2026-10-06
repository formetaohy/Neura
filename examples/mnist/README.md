# MNIST

Train a convolutional network on handwritten digits and classify held out ones with the very same weights.

Run from the repository root:

```sh
cargo run --release --example mnist
```

The program fetches MNIST (60,000 training and 10,000 held out digits of 28 by 28) on its first run, trains a convolution of 8 filters into a convolution of 16 filters with max pooling and two linear layers for four epochs of 128 digit batches with AdamW and cross entropy, and then reads back six held out digits as ASCII art beside the number the network guessed.

## The dataset

The four archives come from the mirror the PyTorch examples use:

```sh
curl -fsSL https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz -o train-images-idx3-ubyte.gz
curl -fsSL https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz -o train-labels-idx1-ubyte.gz
curl -fsSL https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz -o t10k-images-idx3-ubyte.gz
curl -fsSL https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz -o t10k-labels-idx1-ubyte.gz
```

They land in `target/mnist`, which the example fills by itself through `curl`, `wget`, or PowerShell. Point `NEURA_MNIST_DIR` at a directory that already holds those four archives to skip the download, and expect the first run to take a download of about 11 MB on top of the training below.

The archives stay compressed on disk and are parsed in process, so no separate extraction step exists.

## What the run shows

```
device: NVIDIA GeForce RTX 2060 (Numeric { vendor: 4318, device: 7957 }, Discrete, Dx12)
dataset: 60000 training and 10000 held out digits of 28 by 28, cached in target/mnist
training graph: 5632 tasks in 23 waves, 35.8 MB of tensors beside 0.6 MB of weights on a 268 MB heap
inference graph: 732 tasks in 8 waves, 4.4 MB of tensors beside 0.6 MB of weights on a 268 MB heap
device: 2 device programs for 2 plans
epoch 1/4: loss 0.5172, held out loss 0.1193, accuracy 96.20% over 10000 digits in 5.2 s (1916 digits a second)
...
```

Accuracy reaches about 98% on the held out digits in about twenty seconds.

## The two graphs

The batch axis of both graphs is a free extent bounded at 128, so one compiled program serves every batch size:

```rust
let batch = graph.free(128);
let images = graph.input(Shape::of([128, 1, 28, 28]).freed(&[(0, batch)]), Element::Single);
```

Training binds 128 digits a step, and the last step of an epoch binds the 96 digits that remain. Evaluation binds 128 and walks the held out set, while the ASCII gallery binds 6. A binding only rewrites the value and task tables of a program that was compiled once, and `Program` reports what it carries.

The training graph adds automatic differentiation and the AdamW update to the forward pass, and the inference graph is the same model without either of them. Both are compiled against one weight store:

```rust
let weights = runtime.weights(&training);
let learning = runtime.compile(&training, &weights);
runtime.rebind(&weights, &inference);
let inferring = runtime.compile(&inference, &weights);
```

`Runtime::rebind` proves that the second graph declares the very same parameters in the very same order, so inference reads the weights the last training step wrote and never updates them.
