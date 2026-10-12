use neura::decoder::Decoder;
use neura::generate::Generator;
use neura::gguf::GgufFile;
use neura::template::{Message, Template, Value};
use neura::{Element, Graph, MemoryRequest, Runtime, RuntimeRequest, Shape};
use neura_model::qwen35::{Config, Model, Tokenizer};
use std::path::Path;

const CONTEXT: u32 = 1024;
const READBACK: u64 = 8 << 20;
const HEAP: u64 = 2 << 30;

fn main() {
    let mut arguments = std::env::args().skip(1);
    let path = arguments.next().unwrap_or_else(|| {
        panic!("usage: qwen <model.gguf> [prompt] [tokens], holding one Qwen3.5 gguf");
    });
    let prompt = arguments
        .next()
        .unwrap_or_else(|| "What is the capital of France?".to_owned());
    let wanted = arguments
        .next()
        .map_or(48, |held| held.parse::<usize>().expect("a token count"));

    let source = GgufFile::open(Path::new(&path));
    let tokenizer = Tokenizer::read(&source);
    let config = Config::read(source.metadata(), tokenizer.vocabulary());
    let context = CONTEXT.min(config.context);
    println!(
        "{}: {} layers, {} hidden, {} heads of {}, {} context",
        path, config.layers, config.hidden, config.heads, config.head_width, context,
    );

    let rendered = match prompt.strip_prefix("raw:") {
        Some(held) => held.to_owned(),
        None => Template::read(&source).render(
            &[Message::new("user", prompt.as_str())],
            &[
                ("add_generation_prompt", Value::Flag(true)),
                ("enable_thinking", Value::Flag(false)),
            ],
        ),
    };
    let tokens = tokenizer.encode(&rendered);
    println!("prompt: {rendered:?}");
    println!("tokens: {}", tokens.len());

    let runtime = Runtime::open(RuntimeRequest {
        memory: MemoryRequest {
            heap_bytes: HEAP,
            readback_bytes: READBACK,
            ..Default::default()
        },
        ..Default::default()
    })
    .expect("a device that runs the model");
    let graph = Graph::new();
    let model = Model::new(&graph, &source, &config, context);
    let token = graph.input(Shape::matrix(1, 1), Element::Single);
    let position = graph.input(Shape::matrix(1, 1), Element::Single);
    let logits = model.step(&graph, token, position);
    let decoder = Decoder::greedy(
        &graph,
        token,
        position,
        logits,
        model.states(),
        tokenizer.end_of_turn(),
    );
    let weights = runtime.load(&graph, &source);
    let program = runtime.compile(&graph, &weights);
    println!(
        "plan: {:.1}MB of tensors, {:.1}MB of device memory",
        program.tensor_bytes() as f64 / (1 << 20) as f64,
        program.device_bytes() as f64 / (1 << 20) as f64,
    );

    let mut generator = Generator::new(runtime, program, decoder, tokenizer.engine(), context);
    let report = generator.run(&tokens, wanted, |held| print!("{held}"));
    println!(
        "\n{} tokens in {:.2}s ({:.1} tokens/s)",
        report.produced(),
        report.seconds(),
        report.produced() as f64 / report.seconds(),
    );
    if report.reached_context() {
        println!("(the model reached its {context} token context)");
    }
}
