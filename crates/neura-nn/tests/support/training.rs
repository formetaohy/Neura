use neura_graph::Value;
use neura_runtime::{Program, Runtime};

pub fn assert_converges(
    runtime: &Runtime,
    program: &Program,
    loss: Value,
    steps: u32,
    fraction: f32,
) {
    assert!(steps > 1 && fraction > 0.0 && fraction < 1.0);
    runtime.run(program);
    let first = runtime.read(program, loss)[0];
    assert!(
        first.is_finite() && first > 0.0,
        "the loss started at {first}"
    );
    let mut last = first;
    for step in 1..steps {
        runtime.run(program);
        if step % 20 != 0 && step + 1 != steps {
            continue;
        }
        last = runtime.read(program, loss)[0];
        assert!(
            last.is_finite() && last >= 0.0,
            "the loss became {last} on step {step}",
        );
        if last < first * fraction {
            return;
        }
    }
    panic!(
        "the loss moved from {first} to {last} over {steps} steps without reaching {fraction} of its initial value"
    );
}
