use neura_shader_ir::{
    Access, Address, Argument, AtomicOp, Barrier, BinaryOp, Binding, BuiltIn, Constant, Function,
    Global, Instruction, MathFun, Member, Module, Scalar, Space,
};

fn example() -> Module {
    let mut module = Module::new(64);
    let u32_ty = module.scalar(Scalar::U32);
    let f32_ty = module.scalar(Scalar::F32);
    let bool_ty = module.scalar(Scalar::Bool);
    let uvec4_ty = module.vector(Scalar::U32, 4);
    let atomic_ty = module.atomic(Scalar::U32);
    let heap_ty = module.array(f32_ty, None);
    let counts_ty = module.array(atomic_ty, None);
    let record_ty = module.structure(
        "Placement",
        vec![
            Member {
                name: "tensors".to_owned(),
                ty: u32_ty,
                offset: 0,
            },
            Member {
                name: "weights".to_owned(),
                ty: u32_ty,
                offset: 4,
            },
        ],
        8,
    );
    let records_ty = module.array(record_ty, None);
    let scratch_ty = module.array(f32_ty, Some(16));
    let claim_ty = module.array(u32_ty, Some(2));
    let heap = module.add_global(Global {
        name: "heap".to_owned(),
        ty: heap_ty,
        space: Space::Storage,
        binding: Some(Binding {
            group: 0,
            binding: 0,
        }),
        access: Access::ReadWrite,
        coherent: true,
    });
    let counts = module.add_global(Global {
        name: "counts".to_owned(),
        ty: counts_ty,
        space: Space::Storage,
        binding: Some(Binding {
            group: 0,
            binding: 1,
        }),
        access: Access::ReadWrite,
        coherent: true,
    });
    let records = module.add_global(Global {
        name: "records".to_owned(),
        ty: records_ty,
        space: Space::Storage,
        binding: Some(Binding {
            group: 0,
            binding: 2,
        }),
        access: Access::Read,
        coherent: false,
    });
    let scratch = module.add_global(Global {
        name: "scratch".to_owned(),
        ty: scratch_ty,
        space: Space::WorkGroup,
        binding: None,
        access: Access::ReadWrite,
        coherent: false,
    });
    let claim = module.add_global(Global {
        name: "claim".to_owned(),
        ty: claim_ty,
        space: Space::WorkGroup,
        binding: None,
        access: Access::ReadWrite,
        coherent: false,
    });

    let helper = {
        let mut body = Vec::new();
        let value = module.define(f32_ty);
        body.push(Instruction::Argument {
            index: 0,
            result: value,
        });
        let doubled = module.define(f32_ty);
        let two = module.constant(Constant::F32(2.0));
        body.push(Instruction::Binary {
            op: BinaryOp::Multiply,
            left: value,
            right: two,
            result: doubled,
        });
        body.push(Instruction::Return {
            value: Some(doubled),
        });
        module.declare(Function {
            name: "doubled".to_owned(),
            arguments: vec![Argument {
                name: "value".to_owned(),
                ty: f32_ty,
                builtin: None,
            }],
            result: Some(f32_ty),
            locals: Vec::new(),
            body,
        })
    };

    let mut body = Vec::new();
    let lid = module.define(u32_ty);
    body.push(Instruction::Argument {
        index: 0,
        result: lid,
    });
    let pointer = module.pointer(Space::Storage, heap_ty);
    let heap_ptr = module.define(pointer);
    body.push(Instruction::Address {
        address: Address::Global(heap),
        result: heap_ptr,
    });
    let pointer = module.pointer(Space::Storage, f32_ty);
    let cell = module.define(pointer);
    body.push(Instruction::Access {
        base: heap_ptr,
        index: lid,
        result: cell,
    });
    let value = module.define(f32_ty);
    body.push(Instruction::Load {
        pointer: cell,
        result: value,
    });
    let doubled = module.define(f32_ty);
    body.push(Instruction::Call {
        function: helper,
        arguments: vec![value],
        result: Some(doubled),
    });
    let negated = module.define(f32_ty);
    body.push(Instruction::Unary {
        op: neura_shader_ir::UnaryOp::Negate,
        value: doubled,
        result: negated,
    });
    let index_value = module.define(f32_ty);
    body.push(Instruction::Convert {
        value: lid,
        result: index_value,
    });
    let root = module.define(f32_ty);
    body.push(Instruction::Math {
        fun: MathFun::Sqrt,
        arguments: vec![value],
        result: root,
    });
    let pointer = module.pointer(Space::WorkGroup, scratch_ty);
    let scratch_ptr = module.define(pointer);
    body.push(Instruction::Address {
        address: Address::Global(scratch),
        result: scratch_ptr,
    });
    let pointer = module.pointer(Space::WorkGroup, f32_ty);
    let slot = module.define(pointer);
    body.push(Instruction::Access {
        base: scratch_ptr,
        index: lid,
        result: slot,
    });
    body.push(Instruction::Store {
        pointer: slot,
        value: root,
    });
    body.push(Instruction::Barrier(Barrier::WorkGroup));
    let mirrored = module.define(f32_ty);
    body.push(Instruction::Load {
        pointer: slot,
        result: mirrored,
    });
    let pointer = module.pointer(Space::WorkGroup, claim_ty);
    let claim_ptr = module.define(pointer);
    body.push(Instruction::Address {
        address: Address::Global(claim),
        result: claim_ptr,
    });
    let zero = module.constant(Constant::U32(0));
    let pointer = module.pointer(Space::WorkGroup, u32_ty);
    let claim_slot = module.define(pointer);
    body.push(Instruction::Access {
        base: claim_ptr,
        index: zero,
        result: claim_slot,
    });
    let broadcast = module.define(u32_ty);
    body.push(Instruction::WorkGroupUniformLoad {
        pointer: claim_slot,
        result: broadcast,
    });
    let pointer = module.pointer(Space::Storage, counts_ty);
    let counts_ptr = module.define(pointer);
    body.push(Instruction::Address {
        address: Address::Global(counts),
        result: counts_ptr,
    });
    let pointer = module.pointer(Space::Storage, atomic_ty);
    let counter = module.define(pointer);
    body.push(Instruction::Access {
        base: counts_ptr,
        index: lid,
        result: counter,
    });
    let one = module.constant(Constant::U32(1));
    let previous = module.define(u32_ty);
    body.push(Instruction::Atomic {
        op: AtomicOp::Add,
        pointer: counter,
        value: one,
        result: previous,
    });
    let guard = module.define(bool_ty);
    let limit = module.constant(Constant::U32(8));
    body.push(Instruction::Binary {
        op: BinaryOp::GreaterEqual,
        left: previous,
        right: limit,
        result: guard,
    });
    let accepted = module.define(f32_ty);
    body.push(Instruction::Select {
        condition: guard,
        accept: negated,
        reject: mirrored,
        result: accepted,
    });
    let rejected = module.define(u32_ty);
    body.push(Instruction::Unary {
        op: neura_shader_ir::UnaryOp::BitwiseNot,
        value: previous,
        result: rejected,
    });
    let tuple = module.define(uvec4_ty);
    body.push(Instruction::Compose {
        constituents: vec![lid, previous, broadcast, zero],
        result: tuple,
    });
    let component = module.define(u32_ty);
    body.push(Instruction::AccessIndex {
        base: tuple,
        index: 2,
        result: component,
    });
    let pointer = module.pointer(Space::Storage, records_ty);
    let records_ptr = module.define(pointer);
    body.push(Instruction::Address {
        address: Address::Global(records),
        result: records_ptr,
    });
    let pointer = module.pointer(Space::Storage, record_ty);
    let record = module.define(pointer);
    body.push(Instruction::Access {
        base: records_ptr,
        index: component,
        result: record,
    });
    let pointer = module.pointer(Space::Storage, u32_ty);
    let tensors = module.define(pointer);
    body.push(Instruction::AccessIndex {
        base: record,
        index: 1,
        result: tensors,
    });
    let stored = module.define(u32_ty);
    body.push(Instruction::Load {
        pointer: tensors,
        result: stored,
    });
    body.push(Instruction::Store {
        pointer: cell,
        value: index_value,
    });
    let accept = vec![Instruction::Store {
        pointer: slot,
        value: accepted,
    }];
    let reject = {
        let mut reject = Vec::new();
        let alternate = module.define(u32_ty);
        body.push(Instruction::Binary {
            op: BinaryOp::Add,
            left: stored,
            right: previous,
            result: alternate,
        });
        reject.push(Instruction::Store {
            pointer: claim_slot,
            value: alternate,
        });
        reject
    };
    body.push(Instruction::If {
        condition: guard,
        accept,
        reject,
    });
    let header = module.define(bool_ty);
    let bound = module.constant(Constant::U32(4));
    body.push(Instruction::Binary {
        op: BinaryOp::Less,
        left: lid,
        right: bound,
        result: header,
    });
    let body_loop = {
        let mut inner = Vec::new();
        let loaded = module.define(f32_ty);
        inner.push(Instruction::Load {
            pointer: slot,
            result: loaded,
        });
        inner.push(Instruction::Store {
            pointer: slot,
            value: loaded,
        });
        inner.push(Instruction::Barrier(Barrier::Storage));
        inner.push(Instruction::Break);
        inner
    };
    body.push(Instruction::Loop {
        body: body_loop,
        continuing: Vec::new(),
    });
    let chosen = module.define(u32_ty);
    body.push(Instruction::Select {
        condition: header,
        accept: component,
        reject: stored,
        result: chosen,
    });
    body.push(Instruction::Store {
        pointer: claim_slot,
        value: chosen,
    });
    body.push(Instruction::Return { value: None });
    let entry = module.declare(Function {
        name: "main".to_owned(),
        arguments: vec![Argument {
            name: "lid".to_owned(),
            ty: u32_ty,
            builtin: Some(BuiltIn::LocalInvocationIndex),
        }],
        result: None,
        locals: Vec::new(),
        body,
    });
    module.set_entry(entry);
    module
}

#[test]
fn writes_valid_spirv() {
    let module = example();
    module.verify();
    let words = neura_spirv::write(&module);
    assert_eq!(words[0], 0x0723_0203);
    let bytes = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<u8>>();
    std::fs::create_dir_all("E:/tmp/neura-reference").unwrap();
    std::fs::write("E:/tmp/neura-reference/example.spv", bytes).unwrap();
}

#[test]
fn writes_valid_cooperative_matrix_spirv() {
    let mut module = Module::new(64);
    let u32_ty = module.scalar(Scalar::U32);
    let f16_ty = module.scalar(Scalar::F16);
    let f32_ty = module.scalar(Scalar::F32);
    let left_ty = module.cooperative_matrix(Scalar::F16, 16, 16, neura_shader_ir::MatrixUse::A);
    let right_ty = module.cooperative_matrix(Scalar::F16, 16, 16, neura_shader_ir::MatrixUse::B);
    let accumulate_ty =
        module.cooperative_matrix(Scalar::F16, 16, 16, neura_shader_ir::MatrixUse::Accumulator);
    let half_panel_ty = module.array(f16_ty, Some(512));
    let panel = module.add_global(Global {
        name: "panel".to_owned(),
        ty: half_panel_ty,
        space: Space::WorkGroup,
        binding: None,
        access: Access::ReadWrite,
        coherent: false,
    });
    let heap_ty = module.array(f32_ty, None);
    let heap = module.add_global(Global {
        name: "heap".to_owned(),
        ty: heap_ty,
        space: Space::Storage,
        binding: Some(Binding {
            group: 0,
            binding: 0,
        }),
        access: Access::ReadWrite,
        coherent: true,
    });
    let mut body = Vec::new();
    let lid = module.define(u32_ty);
    body.push(Instruction::Argument {
        index: 0,
        result: lid,
    });
    let panel_pointer = module.pointer(Space::WorkGroup, half_panel_ty);
    let panel_base = module.define(panel_pointer);
    body.push(Instruction::Address {
        address: Address::Global(panel),
        result: panel_base,
    });
    let heap_pointer = module.pointer(Space::Storage, heap_ty);
    let heap_base = module.define(heap_pointer);
    body.push(Instruction::Address {
        address: Address::Global(heap),
        result: heap_base,
    });
    let cell = module.pointer(Space::Storage, f32_ty);
    let heap_cell = module.define(cell);
    body.push(Instruction::Access {
        base: heap_base,
        index: lid,
        result: heap_cell,
    });
    let source = module.define(f32_ty);
    body.push(Instruction::Load {
        pointer: heap_cell,
        result: source,
    });
    let half = module.define(f16_ty);
    body.push(Instruction::Convert {
        value: source,
        result: half,
    });
    let slot = module.pointer(Space::WorkGroup, f16_ty);
    let panel_slot = module.define(slot);
    body.push(Instruction::Access {
        base: panel_base,
        index: lid,
        result: panel_slot,
    });
    body.push(Instruction::Store {
        pointer: panel_slot,
        value: half,
    });
    body.push(Instruction::Barrier(Barrier::WorkGroup));
    let stride = module.constant(Constant::U32(32));
    let zero = module.constant(Constant::U32(0));
    let head = module.define(slot);
    body.push(Instruction::Access {
        base: panel_base,
        index: zero,
        result: head,
    });
    let left = module.define(left_ty);
    body.push(Instruction::MatrixLoad {
        pointer: head,
        stride,
        layout: neura_shader_ir::MatrixLayout::RowMajor,
        result: left,
    });
    let right = module.define(right_ty);
    body.push(Instruction::MatrixLoad {
        pointer: head,
        stride,
        layout: neura_shader_ir::MatrixLayout::ColumnMajor,
        result: right,
    });
    let zero_half = module.constant(Constant::Zero(accumulate_ty));
    let empty = module.define(f16_ty);
    body.push(Instruction::MatrixExtract {
        value: zero_half,
        index: zero,
        result: empty,
    });
    let accumulate = module.define(accumulate_ty);
    body.push(Instruction::MatrixInsert {
        value: zero_half,
        index: zero,
        component: empty,
        result: accumulate,
    });
    let product = module.define(accumulate_ty);
    body.push(Instruction::MatrixMulAdd {
        left,
        right,
        accumulate,
        result: product,
    });
    let length = module.define(u32_ty);
    body.push(Instruction::MatrixLength {
        ty: left_ty,
        result: length,
    });
    let accumulator_cell = module.pointer(Space::WorkGroup, f16_ty);
    let product_slot = module.define(accumulator_cell);
    body.push(Instruction::Access {
        base: panel_base,
        index: length,
        result: product_slot,
    });
    body.push(Instruction::MatrixStore {
        pointer: product_slot,
        value: product,
        stride,
        layout: neura_shader_ir::MatrixLayout::RowMajor,
    });
    let entry = module.declare(Function {
        name: "main".to_owned(),
        arguments: vec![Argument {
            name: "lid".to_owned(),
            ty: u32_ty,
            builtin: Some(BuiltIn::LocalInvocationIndex),
        }],
        result: None,
        locals: Vec::new(),
        body,
    });
    module.set_entry(entry);
    module.verify();
    let words = neura_spirv::write(&module);
    assert_eq!(
        words[1], 0x0001_0600,
        "a cooperative program asks SPIR-V 1.6"
    );
    let bytes = words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<u8>>();
    std::fs::write("E:/tmp/neura-reference/cooperative.spv", bytes).unwrap();
}
