// Update fitting parameters after each LM step.
// Maps cuda_update_parameters from Gpufit/cuda_kernels.cu.
//
// Always: prev_parameters[fit, p] = parameters[fit, p]
// If not finished AND not singular: parameters[fit, p] += deltas[fit, p]
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

@group(0) @binding(0) var<uniform>            uniforms:        Uniforms;
@group(0) @binding(1) var<storage, read_write> parameters:     array<f32>;
@group(0) @binding(2) var<storage, read_write> prev_parameters:array<f32>;
@group(0) @binding(3) var<storage, read>       deltas:         array<f32>;
@group(0) @binding(4) var<storage, read>       finished:       array<i32>;
@group(0) @binding(5) var<storage, read>       solution_info:  array<i32>;

@compute @workgroup_size(4, 1, 1)
fn main(
    @builtin(workgroup_id)        wg_id: vec3<u32>,
    @builtin(local_invocation_id)  lid:  vec3<u32>,
) {
    let fit_idx   = wg_id.x;
    let param_idx = lid.x;

    if fit_idx >= uniforms.n_fits { return; }

    let idx = fit_idx * 4u + param_idx;

    // Always save current → prev.
    prev_parameters[idx] = parameters[idx];

    if finished[fit_idx] != 0      { return; }
    if solution_info[fit_idx] != 0 { return; }   // singular: skip delta

    parameters[idx] += deltas[idx];
}
