mod attention;
mod decision;
mod element;
mod measure;
mod movement;
mod product;
mod ragged;
mod scheduler;
mod training;
mod vision;

use neura::{MemoryRequest, Runtime, RuntimeRequest};

type Section = fn(&Runtime);

fn main() {
    let runtime = Runtime::open(RuntimeRequest {
        memory: MemoryRequest {
            heap_bytes: 2 << 30,
            readback_bytes: 8 << 20,
            ..Default::default()
        },
        ..Default::default()
    })
    .expect("a device to measure");
    let info = runtime.context().adapter_info();
    println!(
        "device: {} ({:?}, {:?})",
        info.name, info.backend, info.device_type
    );
    let sections: [(&str, Section); 10] = [
        ("fixed", scheduler::section),
        ("element", element::section),
        ("decision", decision::section),
        ("movement", movement::section),
        ("ragged", ragged::section),
        ("product", product::section),
        ("wide", product::wide_section),
        ("attention", attention::section),
        ("vision", vision::section),
        ("training", training::section),
    ];
    let wanted = std::env::args().nth(1);
    let mut held = 0;
    for (name, section) in sections {
        if wanted.as_deref().is_some_and(|filter| filter != name) {
            continue;
        }
        section(&runtime);
        held += 1;
    }
    assert!(
        wanted.is_none() || held > 0,
        "a benchmark section names itself, and {} names no section of {:?}",
        wanted.unwrap_or_default(),
        sections.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
    );
    println!(
        "\n{} device programs served every section above",
        runtime.declared_kernels()
    );
}
