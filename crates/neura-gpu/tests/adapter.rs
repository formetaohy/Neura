use neura_gpu::{AdapterId, AdapterPolicy, Device, GpuRequest, GpuUnavailable, PowerPreference};

fn open(policy: AdapterPolicy) -> Device {
    Device::open(&GpuRequest {
        adapter: policy,
        ..Default::default()
    })
    .unwrap_or_else(|error| panic!("a native compute device: {error}"))
}

#[test]
fn both_power_classes_open_a_device() {
    open(AdapterPolicy::Power(PowerPreference::HighPerformance));
    open(AdapterPolicy::Power(PowerPreference::LowPower));
}

#[test]
fn an_adapter_identity_selects_that_adapter() {
    let device = open(AdapterPolicy::Power(PowerPreference::HighPerformance));
    let wanted = device.adapter_info().clone();
    drop(device);
    let exact = open(AdapterPolicy::Identity(wanted.id));
    assert_eq!(exact.adapter_info().id, wanted.id);
}

#[test]
fn an_absent_adapter_identity_is_refused() {
    let wanted = AdapterId::Numeric {
        vendor: u32::MAX,
        device: u32::MAX,
    };
    let refused = Device::open(&GpuRequest {
        adapter: AdapterPolicy::Identity(wanted),
        ..Default::default()
    });
    match refused {
        Err(GpuUnavailable::AdapterMissing {
            wanted: reported,
            offered,
        }) => {
            assert_eq!(reported, wanted);
            assert!(
                !offered.is_empty(),
                "a machine that runs this test offers a compute adapter"
            );
        }
        Err(error) => panic!("an absent adapter was refused for another reason: {error}"),
        Ok(device) => panic!("an absent adapter opened {}", device.adapter_info().name),
    }
}
