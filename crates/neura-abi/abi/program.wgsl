const MAX_RANK: u32 = 4u;
const MAX_WAVES: u32 = 4096u;
const CURSOR_WORDS: u32 = 4097u;
const BOUNDS_WORDS: u32 = 3u;

const CURSOR_REFUSED: u32 = 0u;
const CURSOR_WAVE_BASE: u32 = 1u;

const NO_VALUE: u32 = 0xffffffffu;

struct Value {
    base: u32,
    dims: vec4<u32>,
    strides: vec4<u32>,
}

struct Task {
    kind: u32,
    op: u32,
    geometry: u32,
    first: u32,
    count: u32,
    slot: u32,
    out: u32,
    a: u32,
    b: u32,
    c: u32,
    param: f32,
    chain: u32,
    steps: u32,
}

struct Step {
    op: u32,
    operand: u32,
    swapped: u32,
}

struct Bounds {
    first_task: u32,
    task_count: u32,
    wave: u32,
}

@group(0) @binding(0) var<storage, read> tasks: array<Task>;
@group(0) @binding(1) var<storage, read> values: array<Value>;
@group(0) @binding(2) var<storage, read_write> heap: array<f32>;
@group(0) @binding(3) var<storage, read_write> cursor: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> bounds: Bounds;
@group(0) @binding(5) var<storage, read> steps: array<Step>;
