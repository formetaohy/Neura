use crate::CLASSES;
use crate::dataset::Dataset;
use neura::{AdapterInfo, Program};
use std::path::Path;

const RAMP: [char; 10] = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];

pub(crate) fn device(info: &AdapterInfo) {
    println!(
        "device: {} ({:?}, {:?}, {:?})",
        info.name, info.id, info.device_type, info.backend,
    );
}

pub(crate) fn dataset(train: &Dataset, test: &Dataset, directory: &Path) {
    println!(
        "dataset: {} training and {} held out digits of {} by {}, cached in {}",
        train.len(),
        test.len(),
        train.rows(),
        train.columns(),
        directory.display(),
    );
}

pub(crate) fn program(label: &str, program: &Program<'_>) {
    println!(
        "{label}: {} tasks in {} waves, {:.1} MB of tensors beside {:.1} MB of weights on a {:.0} MB heap",
        program.task_count(),
        program.wave_count(),
        program.tensor_bytes() as f64 / 1e6,
        program.weights().bytes() as f64 / 1e6,
        program.heap_bytes() as f64 / 1e6,
    );
}

pub(crate) fn step(index: u32, steps: u32, loss: f32) {
    println!("  step {index:>4}/{steps}: loss {loss:.4}");
}

pub(crate) fn epoch(
    epoch: u32,
    epochs: u32,
    loss: f32,
    held_out_loss: f32,
    accuracy: f64,
    digits: u32,
    seconds: f64,
) {
    println!(
        "epoch {epoch}/{epochs}: loss {loss:.4}, held out loss {held_out_loss:.4}, accuracy {:.2}% over {digits} digits in {seconds:.1} s ({:.0} digits a second)",
        accuracy * 100.0,
        f64::from(digits) / seconds,
    );
}

pub(crate) fn confusion(matrix: &[[u32; CLASSES as usize]; CLASSES as usize]) {
    println!(
        "  truth \\ guess {}",
        (0..CLASSES)
            .map(|class| format!("{class:>6}"))
            .collect::<String>(),
    );
    for (truth, counts) in matrix.iter().enumerate() {
        println!(
            "  {truth:>13} {}",
            counts
                .iter()
                .map(|count| format!("{count:>6}"))
                .collect::<String>(),
        );
    }
}

pub(crate) fn gallery(
    pixels: &[u8],
    labels: &[u8],
    guesses: &[u8],
    rows: u32,
    columns: u32,
    per_row: usize,
) {
    println!("inference on held out digits:");
    for group in (0..labels.len()).collect::<Vec<_>>().chunks(per_row) {
        println!(
            "{}",
            group
                .iter()
                .map(|at| format!(
                    "{:<width$}",
                    format!(
                        "truth {}  guessed {}  {}",
                        labels[*at],
                        guesses[*at],
                        verdict(labels[*at], guesses[*at]),
                    ),
                    width = columns as usize,
                ))
                .collect::<Vec<_>>()
                .join("   ")
                .trim_end(),
        );
        let digits = group
            .iter()
            .map(|at| {
                let image = &pixels[*at * (rows * columns) as usize..];
                (0..rows)
                    .map(|row| {
                        image[(row * columns) as usize..((row + 1) * columns) as usize]
                            .iter()
                            .map(|ink| RAMP[usize::from(*ink) * 9 / 255])
                            .collect::<String>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for row in 0..rows as usize {
            println!(
                "{}",
                digits
                    .iter()
                    .map(|digit| digit[row].clone())
                    .collect::<Vec<_>>()
                    .join("   ")
                    .trim_end(),
            );
        }
        println!();
    }
}

fn verdict(label: u8, guess: u8) -> &'static str {
    if label == guess { "ok" } else { "wrong" }
}

pub(crate) fn timing(device: f64, host: f64, steps: u32, digits: u64) {
    println!(
        "training: {:.2} ms of device time and {:.2} ms of host time per step over {steps} steps, {:.0} digits a second",
        device * 1e3 / f64::from(steps),
        host * 1e3 / f64::from(steps),
        digits as f64 / device,
    );
}
