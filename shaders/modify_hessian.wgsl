// Scale Hessian diagonal by lambda (LM damping).
// Maps cuda_modify_step_widths from Gpufit/cuda_kernels.cu lines ~857-898.
//
// Algorithm (adaptive scaling):
//   1. If iteration_failed: undo previous lambda — H_diag -= sv * (lambda/10)
//   2. sv = max(sv, H_diag)            ← adaptive scaling vector update
//   3. H_diag += sv * lambda           ← apply current lambda
//
// Dispatch: (n_fits, 1, 1)  workgroup_size: (4, 1, 1)
//   wg_id.x = fit index,  lid.x = parameter index (0..3)

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

@group(0) @binding(0) var<uniform>            uniforms:         Uniforms;
@group(0) @binding(1) var<storage, read_write> hessians:        array<f32>;
@group(0) @binding(2) var<storage, read>       lambdas:         array<f32>;
@group(0) @binding(3) var<storage, read_write> scaling_vectors: array<f32>;
@group(0) @binding(4) var<storage, read>       iteration_failed:array<i32>;
@group(0) @binding(5) var<storage, read>       finished:        array<i32>;

@compute @workgroup_size(4, 1, 1)
fn main(
    @builtin(workgroup_id)        wg_id: vec3<u32>,
    @builtin(local_invocation_id)  lid:  vec3<u32>,
) {
    let fit_idx   = wg_id.x;
    let param_idx = lid.x;

    if fit_idx >= uniforms.n_fits { return; }
    if finished[fit_idx] != 0     { return; }

    let lambda    = lambdas[fit_idx];
    let sv_idx    = fit_idx * 4u + param_idx;
    // Diagonal element: H[p,p] at index p*4+p in the 4x4 matrix.
    let h_idx     = fit_idx * 16u + param_idx * 4u + param_idx;

    if iteration_failed[fit_idx] != 0 {
        // Undo previous lambda application: lambda was already multiplied by 10,
        // so the previous lambda was lambda/10.
        hessians[h_idx] -= scaling_vectors[sv_idx] * (lambda / 10.0);
    }

    // Adaptive scaling: sv = max(sv, current H diagonal).
    scaling_vectors[sv_idx] = max(scaling_vectors[sv_idx], hessians[h_idx]);

    // Apply current lambda to diagonal.
    hessians[h_idx] += scaling_vectors[sv_idx] * lambda;
}
