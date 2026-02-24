// Convergence check, state update, lambda update, and parameter restore.
// Merges cuda_check_for_convergence + cuda_evaluate_iteration +
//        cuda_prepare_next_iteration + cuda_update_state_after_solving (Gpufit).
//
// Per fit (one thread per fit):
//   1. Mark singular fits: states = SINGULAR_HESSIAN (2) if solution_info != 0
//   2. Convergence: |chi2 - prev_chi2| < tolerance * max(1, chi2) → finished
//   3. Max iterations: iteration == max_iters-1 → states = MAX_ITERATION (1) → finished
//   4. evaluate_iteration: any non-CONVERGED state → finished; record n_iterations
//   5. prepare_next_iteration:
//        chi2 improved → prev_chi2 = chi2; lambda *= 0.1
//        chi2 worsened → chi2 = prev_chi2; params = prev_params; lambda *= 10
//
// Dispatch: (ceil(n_fits/64), 1, 1)  workgroup_size: (64, 1, 1)

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

@group(0) @binding(0) var<uniform>            uniforms:       Uniforms;
@group(0) @binding(1) var<storage, read_write> finished:      array<i32>;
@group(0) @binding(2) var<storage, read_write> states:        array<i32>;
@group(0) @binding(3) var<storage, read_write> chi_squares:   array<f32>;
@group(0) @binding(4) var<storage, read_write> prev_chi_squares: array<f32>;
@group(0) @binding(5) var<storage, read_write> parameters:    array<f32>;
@group(0) @binding(6) var<storage, read>       prev_parameters: array<f32>;
@group(0) @binding(7) var<storage, read_write> lambdas:       array<f32>;
@group(0) @binding(8) var<storage, read_write> n_iterations:  array<i32>;
@group(0) @binding(9) var<storage, read>       solution_info: array<i32>;

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let fit_idx = gid.x;
    if fit_idx >= uniforms.n_fits { return; }
    if finished[fit_idx] != 0     { return; }

    // 1. Singular hessian → set state (but keep running to restore params).
    if solution_info[fit_idx] != 0 {
        states[fit_idx] = 2;   // SINGULAR_HESSIAN
    }

    let chi2      = chi_squares[fit_idx];
    let prev_chi2 = prev_chi_squares[fit_idx];

    // 2. Convergence check.
    let tol       = uniforms.tolerance;
    let converged = abs(chi2 - prev_chi2) < tol * max(1.0, chi2);

    if converged {
        finished[fit_idx] = 1;
    }

    // 3. Max iterations check.
    let max_iter_reached = uniforms.iteration == uniforms.max_iterations - 1u;
    if !converged && max_iter_reached {
        states[fit_idx] = 1;   // MAX_ITERATION
    }

    // 4. evaluate_iteration: any non-zero state → finished.
    if states[fit_idx] != 0 {
        finished[fit_idx] = 1;
    }

    // Record iteration count on first finish.
    if finished[fit_idx] != 0 && n_iterations[fit_idx] == 0 {
        n_iterations[fit_idx] = i32(uniforms.iteration) + 1;
    }

    // 5. prepare_next_iteration: lambda and parameter update.
    if chi2 < prev_chi2 {
        // Improvement: accept step, decrease lambda.
        lambdas[fit_idx]       *= 0.1;
        prev_chi_squares[fit_idx] = chi2;
    } else {
        // No improvement: reject step, increase lambda, restore parameters.
        lambdas[fit_idx]       *= 10.0;
        chi_squares[fit_idx]    = prev_chi2;
        let base = fit_idx * 4u;
        parameters[base + 0u] = prev_parameters[base + 0u];
        parameters[base + 1u] = prev_parameters[base + 1u];
        parameters[base + 2u] = prev_parameters[base + 2u];
        parameters[base + 3u] = prev_parameters[base + 3u];
    }
}
