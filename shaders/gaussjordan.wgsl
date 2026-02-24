// Gauss-Jordan elimination with row pivoting.
// Maps cuda_gaussjordan.cu from Gpufit.
//
// Solves H * delta = gradient by building the 4x5 augmented matrix [H|g]
// and reducing the left side to (permuted) identity, then extracting delta.
//
// Dispatch: (n_fits, 1, 1)  workgroup_size: (5, 4, 1)
//   One workgroup per fit; 20 threads cover the 4×5 augmented matrix.
//   lid.x = column (0..4),  lid.y = row (0..3)

struct Uniforms {
    n_fits:         u32,
    n_points:       u32,
    n_params:       u32,
    iteration:      u32,
    max_iterations: u32,
    tolerance:      f32,
    pad0:           u32,
    pad1:           u32,
}

@group(0) @binding(0) var<uniform>            uniforms:     Uniforms;
@group(0) @binding(1) var<storage, read>      hessians:     array<f32>;
@group(0) @binding(2) var<storage, read>      gradients:    array<f32>;
@group(0) @binding(3) var<storage, read>      finished:     array<i32>;
@group(0) @binding(4) var<storage, read_write> deltas:      array<f32>;
@group(0) @binding(5) var<storage, read_write> solution_info: array<i32>;

// 4 rows × 5 cols augmented matrix, shared within the workgroup.
var<workgroup> calc_mat:   array<f32, 20>;
var<workgroup> pivot_col:  u32;

@compute @workgroup_size(5, 4, 1)
fn main(
    @builtin(workgroup_id)        wg_id: vec3<u32>,
    @builtin(local_invocation_id)  lid:  vec3<u32>,
) {
    let fit_idx = wg_id.x;
    let col     = lid.x;   // 0..4
    let row     = lid.y;   // 0..3

    if fit_idx >= uniforms.n_fits { return; }

    // Initialise solution_info and augmented matrix.
    if row == 0u && col == 0u {
        solution_info[fit_idx] = 0;
    }

    if finished[fit_idx] != 0 {
        workgroupBarrier();
        return;
    }

    if col < 4u {
        // Left side: Hessian row
        calc_mat[row * 5u + col] = hessians[fit_idx * 16u + row * 4u + col];
    } else {
        // Right side: gradient vector
        calc_mat[row * 5u + 4u] = gradients[fit_idx * 4u + row];
    }
    workgroupBarrier();

    // Gauss-Jordan elimination over 4 pivot rows.
    for (var cr: u32 = 0u; cr < 4u; cr++) {

        // Thread (row=cr, col=0) finds the column of the largest absolute
        // value in the current row (row-pivoting strategy from cuda_gaussjordan).
        if row == cr && col == 0u {
            var max_abs: f32 = 0.0;
            var max_c:  u32  = 0u;
            for (var c: u32 = 0u; c < 4u; c++) {
                let v = abs(calc_mat[cr * 5u + c]);
                if v > max_abs {
                    max_abs = v;
                    max_c   = c;
                }
            }
            pivot_col = max_c;
            if max_abs == 0.0 {
                solution_info[fit_idx] = 1;   // singular
            }
        }
        workgroupBarrier();

        // Divide the current row by its pivot element.
        if row == cr {
            let pv = calc_mat[cr * 5u + pivot_col];
            if pv != 0.0 {
                calc_mat[row * 5u + col] /= pv;
            }
        }
        workgroupBarrier();

        // Eliminate the pivot column in all other rows.
        if row != cr {
            let factor = calc_mat[row * 5u + pivot_col];
            calc_mat[row * 5u + col] -= factor * calc_mat[cr * 5u + col];
        }
        workgroupBarrier();
    }

    // Extract solution: the reduced left side is a permuted identity matrix.
    // Each row has exactly one entry equal to 1; its column index is the variable index.
    if col < 4u && abs(calc_mat[row * 5u + col] - 1.0) < 1e-5 {
        deltas[fit_idx * 4u + col] = calc_mat[row * 5u + 4u];
    }
}
