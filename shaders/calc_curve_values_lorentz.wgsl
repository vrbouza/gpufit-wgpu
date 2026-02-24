// Lorentzian model evaluation + all 4 partial derivatives.
//
// Parameters: [A, x0, gamma, offset]
//   value = A / (1 + ((x - x0) / gamma)^2) + offset
//
// Let u = (x - x0) / gamma,  denom = 1 + u^2
//   df/dA      = 1 / denom
//   df/dx0     = 2 * A * u      / (gamma * denom^2)
//   df/dgamma  = 2 * A * u^2   / (gamma * denom^2)
//   df/doffset = 1
//
// Derivatives layout in buf_derivatives:
//   derivatives[fit * n_points * 4 + param * n_points + point]
//
// Dispatch: (n_fits, 1, 1)  workgroup_size: (WG_SIZE, 1, 1)
//   wg_id.x  = fit index
//   lid.x    = point index within fit

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

@group(0) @binding(0) var<uniform>            uniforms:    Uniforms;
@group(0) @binding(1) var<storage, read>      parameters:  array<f32>;
@group(0) @binding(2) var<storage, read>      finished:    array<i32>;
@group(0) @binding(3) var<storage, read_write> values:     array<f32>;
@group(0) @binding(4) var<storage, read_write> derivatives:array<f32>;
@group(0) @binding(5) var<storage, read>      x_values:    array<f32>;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id)       wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>,
) {
    let fit_idx   = wg_id.x;
    let point_idx = lid.x;

    if fit_idx >= uniforms.n_fits     { return; }
    if finished[fit_idx] != 0         { return; }
    if point_idx >= uniforms.n_points { return; }

    let p_base = fit_idx * 4u;
    let A      = parameters[p_base + 0u];
    let x0     = parameters[p_base + 1u];
    let gamma  = parameters[p_base + 2u];
    let offset = parameters[p_base + 3u];

    let x      = x_values[point_idx];
    let u      = (x - x0) / gamma;
    let denom  = 1.0 + u * u;
    let denom2 = denom * denom;

    let val_idx = fit_idx * uniforms.n_points + point_idx;
    values[val_idx] = A / denom + offset;

    let d_base = fit_idx * uniforms.n_points * 4u;
    // df/dA
    derivatives[d_base + 0u * uniforms.n_points + point_idx] = 1.0 / denom;
    // df/dx0
    derivatives[d_base + 1u * uniforms.n_points + point_idx] = 2.0 * A * u / (gamma * denom2);
    // df/dgamma
    derivatives[d_base + 2u * uniforms.n_points + point_idx] = 2.0 * A * u * u / (gamma * denom2);
    // df/doffset
    derivatives[d_base + 3u * uniforms.n_points + point_idx] = 1.0;
}
