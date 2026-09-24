use neura_graph::Window;

fn refuses(action: impl FnOnce() + std::panic::UnwindSafe) -> bool {
    std::panic::catch_unwind(action).is_err()
}

#[test]
fn a_window_declares_how_it_walks_its_input() {
    let window = Window::new([3, 5], [2, 4], [1, 6]);
    assert_eq!(window.reach_rows(), 3);
    assert_eq!(window.reach_columns(), 5);
    assert_eq!(window.stride_rows(), 2);
    assert_eq!(window.stride_columns(), 4);
    assert_eq!(window.pad_rows(), 1);
    assert_eq!(window.pad_columns(), 6);
    let sliding = Window::sliding([3, 3]);
    assert_eq!(sliding.stride_rows(), 1);
    assert_eq!(sliding.stride_columns(), 1);
    assert_eq!(sliding.pad_rows(), 0);
    assert_eq!(sliding.pad_columns(), 0);
    assert_eq!(sliding.reach_rows(), 3);
    assert!(refuses(|| {
        let _ = Window::new([0, 3], [1, 1], [0, 0]);
    }));
    assert!(refuses(|| {
        let _ = Window::new([3, 3], [0, 1], [0, 0]);
    }));
    assert!(refuses(|| {
        let _ = Window::sliding([3, 0]);
    }));
}
