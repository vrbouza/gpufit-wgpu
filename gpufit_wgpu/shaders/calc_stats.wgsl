// Merged: chi-square + gradient + Hessian + iteration_failed check.
// Maps: cuda_calculate_chi_squares + cuda_calculate_gradients +
//       cuda_calculate_hessians + cuda_check_fit_improvement (from Gpufit).
//
// LSE formulas (lse.cuh):
//   chi2      = sum_i (f_i - y_i)^2
//   grad[k]   = sum_i (y_i - f_i) * df/dp_k[i]
//   H[j][k]   = sum_i df/dp_j[i] * df/dp_k[i]
//
// iteration_failed = 1 if chi2 >= prev_chi2 AND prev_chi2 was initialised.
//
// Dispatch: (n_fits, 1, 1)  workgroup_size: (WG_SIZE, 1, 1)
//   Each workgroup handles one fit; threads cooperate via workgroup memory.
//   Threads 0..n_points-1 compute per-point contributions.
//   Thread 0 performs tree-reduction write-out and sequential Hessian.

override WG_SIZE: u32 = 64u;

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

@group(0) @binding(0) var<uniform>            uniforms:        Uniforms;
@group(0) @binding(1) var<storage, read>      data:            array<f32>;
@group(0) @binding(2) var<storage, read>      values:          array<f32>;
@group(0) @binding(3) var<storage, read>      derivatives:     array<f32>;
@group(0) @binding(4) var<storage, read>      finished:        array<i32>;
@group(0) @binding(5) var<storage, read_write> chi_squares:    array<f32>;
@group(0) @binding(6) var<storage, read_write> gradients:      array<f32>;
@group(0) @binding(7) var<storage, read_write> hessians:       array<f32>;
@group(0) @binding(8) var<storage, read_write> iteration_failed: array<i32>;
@group(0) @binding(9) var<storage, read>      prev_chi_squares: array<f32>;

// Workgroup-shared accumulation arrays (zero-initialised per dispatch).
var<workgroup> s_chi:   array<f32, WG_SIZE>;
var<workgroup> s_grad0: array<f32, WG_SIZE>;
var<workgroup> s_grad1: array<f32, WG_SIZE>;
var<workgroup> s_grad2: array<f32, WG_SIZE>;
var<workgroup> s_grad3: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id)        wg_id: vec3<u32>,
    @builtin(local_invocation_id)  lid:  vec3<u32>,
) {
    let fit_idx = wg_id.x;
    let li      = lid.x;

    // Zero-init shared arrays (covers threads beyond n_points).
    s_chi[li]   = 0.0;
    s_grad0[li] = 0.0;
    s_grad1[li] = 0.0;
    s_grad2[li] = 0.0;
    s_grad3[li] = 0.0;

    if fit_idx >= uniforms.n_fits { workgroupBarrier(); return; }
    if finished[fit_idx] != 0     { workgroupBarrier(); return; }

    if li < uniforms.n_points {
        let base   = fit_idx * uniforms.n_points;
        let val    = values[base + li];
        let dat    = data[base + li];
        let diff   = val - dat;

        s_chi[li]  = diff * diff;

        let d_base = fit_idx * uniforms.n_points * 4u;
        let d0     = derivatives[d_base + 0u * uniforms.n_points + li];
        let d1     = derivatives[d_base + 1u * uniforms.n_points + li];
        let d2     = derivatives[d_base + 2u * uniforms.n_points + li];
        let d3     = derivatives[d_base + 3u * uniforms.n_points + li];

        // LSE gradient: (data - value) * deriv
        s_grad0[li] = (dat - val) * d0;
        s_grad1[li] = (dat - val) * d1;
        s_grad2[li] = (dat - val) * d2;
        s_grad3[li] = (dat - val) * d3;
    }
    workgroupBarrier();

    // Parallel tree reduction (64 → 1).
    var stride = WG_SIZE >> 1u;
    while stride > 0u {
        if li < stride {
            s_chi[li]   += s_chi[li   + stride];
            s_grad0[li] += s_grad0[li + stride];
            s_grad1[li] += s_grad1[li + stride];
            s_grad2[li] += s_grad2[li + stride];
            s_grad3[li] += s_grad3[li + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    // Thread 0: write aggregated results and compute full Hessian.
    if li == 0u {
        let chi2       = s_chi[0u];
        chi_squares[fit_idx] = chi2;

        let g_base = fit_idx * 4u;
        gradients[g_base + 0u] = s_grad0[0u];
        gradients[g_base + 1u] = s_grad1[0u];
        gradients[g_base + 2u] = s_grad2[0u];
        gradients[g_base + 3u] = s_grad3[0u];

        // iteration_failed: chi2 increased relative to prev?
        let prev_chi2   = prev_chi_squares[fit_idx];
        let prev_init   = prev_chi2 != 0.0;
        let chi2_bad    = chi2 >= prev_chi2;
        if prev_init && chi2_bad {
            iteration_failed[fit_idx] = 1;
        } else {
            iteration_failed[fit_idx] = 0;
        }

        // Hessian: sequential loop over all data points (thread 0 only).
        // H[i][j] = sum_pt  d_i[pt] * d_j[pt]
        let h_base  = fit_idx * 16u;
        let np      = uniforms.n_points;
        let db      = fit_idx * np * 4u;

        for (var i = 0u; i < 16u; i++) {
            hessians[h_base + i] = 0.0;
        }

        for (var pt = 0u; pt < np; pt++) {
            let d0 = derivatives[db + 0u * np + pt];
            let d1 = derivatives[db + 1u * np + pt];
            let d2 = derivatives[db + 2u * np + pt];
            let d3 = derivatives[db + 3u * np + pt];
            let ds = array<f32, 4>(d0, d1, d2, d3);

            for (var pi = 0u; pi < 4u; pi++) {
                for (var pj = pi; pj < 4u; pj++) {
                    let h = ds[pi] * ds[pj];
                    hessians[h_base + pi * 4u + pj] += h;
                    if pi != pj {
                        hessians[h_base + pj * 4u + pi] += h;
                    }
                }
            }
        }
    }
}
