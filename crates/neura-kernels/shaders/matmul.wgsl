var<workgroup> matmul_left: array<f32, MATMUL_ROW_TILE * MATMUL_DEPTH_TILE>;
var<workgroup> matmul_right: array<f32, MATMUL_DEPTH_TILE * MATMUL_COL_TILE>;

fn run_matmul(task: Task, lid: u32) {
    let left = values[task.a];
    let right = values[task.b];
    let output = values[task.out];
    let rows = left.dims.z;
    let depth = left.dims.w;
    let columns = right.dims.w;
    let column_blocks = (columns + MATMUL_COL_TILE - 1u) / MATMUL_COL_TILE;
    let depth_blocks = (depth + MATMUL_DEPTH_TILE - 1u) / MATMUL_DEPTH_TILE;
    let thread_row = (lid / (MATMUL_COL_TILE / 2u)) * 2u;
    let thread_column = (lid % (MATMUL_COL_TILE / 2u)) * 2u;
    for (var tile = task.first; tile < task.first + task.count; tile = tile + 1u) {
        let base_row = (tile / column_blocks) * MATMUL_ROW_TILE;
        let base_column = (tile % column_blocks) * MATMUL_COL_TILE;
        var sums = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
        for (var block = 0u; block < depth_blocks; block = block + 1u) {
            let base_depth = block * MATMUL_DEPTH_TILE;
            for (var unit = lid; unit < MATMUL_ROW_TILE * MATMUL_DEPTH_TILE; unit = unit + WORKGROUP_SIZE) {
                let row = base_row + unit / MATMUL_DEPTH_TILE;
                let column = base_depth + unit % MATMUL_DEPTH_TILE;
                let inside = row < rows && column < depth;
                let address = select(0u, row * left.strides.z + column * left.strides.w, inside);
                matmul_left[unit] = select(0.0, arena[left.base + address], inside);
            }
            for (var unit = lid; unit < MATMUL_DEPTH_TILE * MATMUL_COL_TILE; unit = unit + WORKGROUP_SIZE) {
                let row = base_depth + unit / MATMUL_COL_TILE;
                let column = base_column + unit % MATMUL_COL_TILE;
                let inside = row < depth && column < columns;
                let address = select(0u, row * right.strides.z + column * right.strides.w, inside);
                matmul_right[unit] = select(0.0, arena[right.base + address], inside);
            }
            workgroupBarrier();
            for (var step = 0u; step < MATMUL_DEPTH_TILE; step = step + 1u) {
                let left0 = matmul_left[thread_row * MATMUL_DEPTH_TILE + step];
                let left1 = matmul_left[(thread_row + 1u) * MATMUL_DEPTH_TILE + step];
                let right0 = matmul_right[step * MATMUL_COL_TILE + thread_column];
                let right1 = matmul_right[step * MATMUL_COL_TILE + thread_column + 1u];
                sums[0] = sums[0] + left0 * right0;
                sums[1] = sums[1] + left0 * right1;
                sums[2] = sums[2] + left1 * right0;
                sums[3] = sums[3] + left1 * right1;
            }
            workgroupBarrier();
        }
        for (var unit = 0u; unit < 4u; unit = unit + 1u) {
            let row = base_row + thread_row + unit / 2u;
            let column = base_column + thread_column + unit % 2u;
            if (row < rows && column < columns) {
                arena[output.base + row * columns + column] = sums[unit];
            }
        }
    }
}
