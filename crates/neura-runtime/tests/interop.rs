use neura_abi::WORD_BYTES;
use neura_gpu::wgpu;
use neura_program::{Graph, Shape};
use neura_runtime::RuntimeRequest;

const VERTEX: &str = "
@vertex
fn vertex_main(@location(0) vertex: vec4<f32>) -> VertexOut {
    var out: VertexOut;
    out.position = vec4<f32>(vertex.x, vertex.y, 0.0, 1.0);
    out.payload = vertex.w;
    return out;
}

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) payload: f32,
}

@fragment
fn fragment_main(@location(0) payload: f32) -> @location(0) f32 {
    return payload;
}
";

#[test]
fn a_game_pipeline_draws_a_tensor_straight_out_of_the_arena() {
    let runtime = pollster::block_on(neura_runtime::Runtime::open(RuntimeRequest {
        arena_bytes: 1 << 20,
        readback_bytes: 1 << 12,
        ..Default::default()
    }))
    .expect("device");
    let graph = Graph::new();
    let vertices = graph.input(Shape::matrix(4, 4));
    let drawn = graph.mul(vertices, graph.fill(Shape::matrix(4, 4), 1.0));
    let program = runtime.compile(&graph);
    let quad = [
        -1.0, -1.0, 0.0, 7.0, //
        1.0, -1.0, 0.0, 7.0, //
        -1.0, 1.0, 0.0, 7.0, //
        1.0, 1.0, 0.0, 7.0,
    ];
    runtime.write(&program, vertices, &quad);
    runtime.run(&program);

    let device = runtime.context().device();
    let queue = runtime.context().queue();
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("neura interop target"),
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("neura interop shader"),
        source: wgpu::ShaderSource::Wgsl(VERTEX.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("neura interop pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vertex_main"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                }],
            })],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fragment_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::R32Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    let span = program.span(drawn);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("neura interop readback"),
        size: 4 * 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("neura interop"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("neura interop pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, runtime.arena().buffer().slice(span.offset..));
        pass.draw(0..4, 0..1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(4),
            },
        },
        wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
    );
    let index = queue.submit([encoder.finish()]);
    let (sender, completion) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(index),
            timeout: Some(std::time::Duration::from_secs(30)),
        })
        .expect("the render submission finished");
    completion
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the readback callback fired")
        .expect("the readback mapped");
    let bytes = staging.slice(..).get_mapped_range().expect("mapped");
    let mut pixels: Vec<f32> = Vec::new();
    for row in 0..4usize {
        pixels.extend(bytemuck::cast_slice::<u8, f32>(
            &bytes[row * 256..row * 256 + 16],
        ));
    }
    assert_eq!(span.elements, 16);
    assert_eq!(span.offset % WORD_BYTES, 0);
    assert!(
        pixels.iter().all(|pixel| (*pixel - 7.0).abs() < 1e-6),
        "the drawn pixels came back as {pixels:?}",
    );
    drop(bytes);
    staging.unmap();
}
