use neura_gpu::{Backend, Device, GpuRequest, GpuUnavailable, PREFERENCE};

#[test]
fn a_request_without_a_backend_opens_the_one_a_build_prefers_first() {
    let device = Device::open(&GpuRequest::default()).expect("a native compute device");
    assert_eq!(device.adapter_info().backend, PREFERENCE[0]);
}

#[test]
fn a_request_opens_the_backend_it_names() {
    for &backend in PREFERENCE {
        let device = Device::open(&GpuRequest {
            backend: Some(backend),
            ..Default::default()
        })
        .unwrap_or_else(|error| panic!("{backend:?} could not open a device: {error}"));
        assert_eq!(device.adapter_info().backend, backend);
    }
}

#[test]
fn a_backend_a_build_lacks_is_refused() {
    let absent = [Backend::Dx12, Backend::Metal, Backend::Vulkan]
        .into_iter()
        .find(|backend| !PREFERENCE.contains(backend))
        .expect("a build offers fewer than three backends");
    match Device::open(&GpuRequest {
        backend: Some(absent),
        ..Default::default()
    }) {
        Err(GpuUnavailable::BackendMissing { wanted, offered }) => {
            assert_eq!(wanted, absent);
            assert_eq!(offered, PREFERENCE);
        }
        Err(error) => panic!("a build without {absent:?} refused it for another reason: {error}"),
        Ok(device) => panic!(
            "a build without {absent:?} opened {}",
            device.adapter_info().name
        ),
    }
}
