"""
LM fitting loop orchestrator using wgpu (WebGPU).

Mirrors LMFitCUDA::run() from Gpufit/Gpufit/lm_fit_cuda.cpp.

All GPU pipelines and bind groups are created once in __init__.
Each call to fit() uploads data, runs the loop, and returns results.
"""
from __future__ import annotations

import importlib.resources
import math
from typing import NamedTuple

import numpy as np
import webgpu as wgpu
from webgpu.webgpu_api import ShaderStage, Device
from webgpu.utils import init_device

from .gpu_buffers import (
    allocate_buffers,
    copy_buffer,
    download,
    upload,
    upload_zeros,
    write_uniforms,
)

SHADER_DIR = importlib.resources.files("gpufit_wgpu") / "shaders"
N_PARAMS = 4


class FitResult(NamedTuple):
    parameters:   np.ndarray   # [n_fits, 4]  final fitted params
    states:       np.ndarray   # [n_fits]  0=converged 1=max_iter 2=singular
    chi_squares:  np.ndarray   # [n_fits]
    n_iterations: np.ndarray   # [n_fits]


# ---------------------------------------------------------------------------
# Helper: nearest power-of-2 >= n
# ---------------------------------------------------------------------------
def _pow2(n: int) -> int:
    return 1 << math.ceil(math.log2(max(n, 1)))


# ---------------------------------------------------------------------------
# Bind-group layout helpers
# ---------------------------------------------------------------------------
def _bgl_entry(binding: int, buffer_type: str) -> dict:
    """Return a BindGroupLayoutEntry dict for a COMPUTE-visible buffer binding."""
    return {
        "binding":    binding,
        "visibility": ShaderStage.COMPUTE,
        "buffer":     {"type": buffer_type},
    }


def _bg_entry(binding: int, buf: wgpu.GPUBuffer) -> dict:
    return {"binding": binding, "resource": {"buffer": buf}}


# ---------------------------------------------------------------------------
# LMFitter
# ---------------------------------------------------------------------------
_MODEL_SHADERS = {
    "gauss":   "calc_curve_values_gauss.wgsl",
    "lorentz": "calc_curve_values_lorentz.wgsl",
}


class LMFitter:
    """
    GPU Levenberg-Marquardt fitter using wgpu.

    Parameters
    ----------
    device      : wgpu GPU device (call once with gpu_device() helper).
    n_fits      : Number of parallel fits.
    n_points    : Data points per fit.
    max_iters   : Maximum LM iterations (default 100).
    tolerance   : Convergence tolerance (default 1e-4).
    init_lambda : Initial Marquardt damping parameter (default 1e-3).
    model       : Model function — "lorentz" (default) or "gauss".
    """

    def __init__(
        self,
        device: wgpu.GPUDevice,
        n_fits: int,
        n_points: int,
        max_iters: int = 100,
        tolerance: float = 1e-4,
        init_lambda: float = 1e-3,
        model: str = "lorentz",
    ):
        if model not in _MODEL_SHADERS:
            raise ValueError(f"model must be one of {list(_MODEL_SHADERS)}, got {model!r}")
        self.device      = device
        self.n_fits      = n_fits
        self.n_points    = n_points
        self.max_iters   = max_iters
        self.tolerance   = tolerance
        self.init_lambda = init_lambda
        self.model       = model
        self.wg_size     = _pow2(n_points)

        self.bufs = allocate_buffers(device, n_fits, n_points, N_PARAMS)
        self._create_pipelines()
        self._create_bind_groups()

    # ------------------------------------------------------------------
    # Pipeline creation
    # ------------------------------------------------------------------
    def _load_shader(self, name: str):
        src = (SHADER_DIR / name).read_text()
        return self.device.createShaderModule({"code": src})

    def _make_pipeline(self, shader_module, entry: str, bgl, constants: dict | None = None):
        pl = self.device.createPipelineLayout({"bindGroupLayouts": [bgl]})
        compute_stage: dict = {"module": shader_module, "entryPoint": entry}
        if constants:
            compute_stage["constants"] = constants
        return self.device.createComputePipeline({"layout": pl, "compute": compute_stage})

    def _create_pipelines(self):
        dev = self.device
        wgs = self.wg_size

        # ── calc_curve_values (model-specific) ────────────────────────
        sh_cv = self._load_shader(_MODEL_SHADERS[self.model])
        bgl_cv = dev.createBindGroupLayout({"entries": [
            _bgl_entry(0, "uniform"),
            _bgl_entry(1, "read-only-storage"),  # parameters
            _bgl_entry(2, "read-only-storage"),  # finished
            _bgl_entry(3, "storage"),             # values
            _bgl_entry(4, "storage"),             # derivatives
            _bgl_entry(5, "read-only-storage"),  # x_values
        ]})
        self.pipe_cv  = self._make_pipeline(sh_cv, "main", bgl_cv, {"WG_SIZE": wgs})
        self.bgl_cv   = bgl_cv

        # ── calc_stats ─────────────────────────────────────────────────
        sh_cs = self._load_shader("calc_stats.wgsl")
        bgl_cs = dev.createBindGroupLayout({"entries": [
            _bgl_entry(0, "uniform"),
            _bgl_entry(1, "read-only-storage"),  # data
            _bgl_entry(2, "read-only-storage"),  # values
            _bgl_entry(3, "read-only-storage"),  # derivatives
            _bgl_entry(4, "read-only-storage"),  # finished
            _bgl_entry(5, "storage"),             # chi_squares
            _bgl_entry(6, "storage"),             # gradients
            _bgl_entry(7, "storage"),             # hessians
            _bgl_entry(8, "storage"),             # iteration_failed
            _bgl_entry(9, "read-only-storage"),  # prev_chi_squares
        ]})
        self.pipe_cs  = self._make_pipeline(sh_cs, "main", bgl_cs, {"WG_SIZE": wgs})
        self.bgl_cs   = bgl_cs

        # ── modify_hessian ─────────────────────────────────────────────
        sh_mh = self._load_shader("modify_hessian.wgsl")
        bgl_mh = dev.createBindGroupLayout({"entries": [
            _bgl_entry(0, "uniform"),
            _bgl_entry(1, "storage"),             # hessians (rw)
            _bgl_entry(2, "read-only-storage"),  # lambdas
            _bgl_entry(3, "storage"),             # scaling_vectors (rw)
            _bgl_entry(4, "read-only-storage"),  # iteration_failed
            _bgl_entry(5, "read-only-storage"),  # finished
        ]})
        self.pipe_mh  = self._make_pipeline(sh_mh, "main", bgl_mh)
        self.bgl_mh   = bgl_mh

        # ── gaussjordan ────────────────────────────────────────────────
        sh_gj = self._load_shader("gaussjordan.wgsl")
        bgl_gj = dev.createBindGroupLayout({"entries": [
            _bgl_entry(0, "uniform"),
            _bgl_entry(1, "read-only-storage"),  # hessians
            _bgl_entry(2, "read-only-storage"),  # gradients
            _bgl_entry(3, "read-only-storage"),  # finished
            _bgl_entry(4, "storage"),             # deltas
            _bgl_entry(5, "storage"),             # solution_info
        ]})
        self.pipe_gj  = self._make_pipeline(sh_gj, "main", bgl_gj)
        self.bgl_gj   = bgl_gj

        # ── update_params ──────────────────────────────────────────────
        sh_up = self._load_shader("update_params.wgsl")
        bgl_up = dev.createBindGroupLayout({"entries": [
            _bgl_entry(0, "uniform"),
            _bgl_entry(1, "storage"),             # parameters (rw)
            _bgl_entry(2, "storage"),             # prev_parameters (rw)
            _bgl_entry(3, "read-only-storage"),  # deltas
            _bgl_entry(4, "read-only-storage"),  # finished
            _bgl_entry(5, "read-only-storage"),  # solution_info
        ]})
        self.pipe_up  = self._make_pipeline(sh_up, "main", bgl_up)
        self.bgl_up   = bgl_up

        # ── convergence ────────────────────────────────────────────────
        sh_co = self._load_shader("convergence.wgsl")
        bgl_co = dev.createBindGroupLayout({"entries": [
            _bgl_entry(0, "uniform"),
            _bgl_entry(1, "storage"),             # finished (rw)
            _bgl_entry(2, "storage"),             # states (rw)
            _bgl_entry(3, "storage"),             # chi_squares (rw)
            _bgl_entry(4, "storage"),             # prev_chi_squares (rw)
            _bgl_entry(5, "storage"),             # parameters (rw)
            _bgl_entry(6, "read-only-storage"),  # prev_parameters
            _bgl_entry(7, "storage"),             # lambdas (rw)
            _bgl_entry(8, "storage"),             # n_iterations (rw)
            _bgl_entry(9, "read-only-storage"),  # solution_info
        ]})
        self.pipe_co  = self._make_pipeline(sh_co, "main", bgl_co)
        self.bgl_co   = bgl_co

    # ------------------------------------------------------------------
    # Bind group creation (buffers are fixed after __init__)
    # ------------------------------------------------------------------
    def _create_bind_groups(self):
        dev  = self.device
        bufs = self.bufs

        self.bg_cv = dev.createBindGroup({"layout": self.bgl_cv, "entries": [
            _bg_entry(0, bufs["uniforms"]),
            _bg_entry(1, bufs["parameters"]),
            _bg_entry(2, bufs["finished"]),
            _bg_entry(3, bufs["values"]),
            _bg_entry(4, bufs["derivatives"]),
            _bg_entry(5, bufs["x_values"]),
        ]})

        self.bg_cs = dev.createBindGroup({"layout": self.bgl_cs, "entries": [
            _bg_entry(0, bufs["uniforms"]),
            _bg_entry(1, bufs["data"]),
            _bg_entry(2, bufs["values"]),
            _bg_entry(3, bufs["derivatives"]),
            _bg_entry(4, bufs["finished"]),
            _bg_entry(5, bufs["chi_squares"]),
            _bg_entry(6, bufs["gradients"]),
            _bg_entry(7, bufs["hessians"]),
            _bg_entry(8, bufs["iteration_failed"]),
            _bg_entry(9, bufs["prev_chi_squares"]),
        ]})

        self.bg_mh = dev.createBindGroup({"layout": self.bgl_mh, "entries": [
            _bg_entry(0, bufs["uniforms"]),
            _bg_entry(1, bufs["hessians"]),
            _bg_entry(2, bufs["lambdas"]),
            _bg_entry(3, bufs["scaling_vectors"]),
            _bg_entry(4, bufs["iteration_failed"]),
            _bg_entry(5, bufs["finished"]),
        ]})

        self.bg_gj = dev.createBindGroup({"layout": self.bgl_gj, "entries": [
            _bg_entry(0, bufs["uniforms"]),
            _bg_entry(1, bufs["hessians"]),
            _bg_entry(2, bufs["gradients"]),
            _bg_entry(3, bufs["finished"]),
            _bg_entry(4, bufs["deltas"]),
            _bg_entry(5, bufs["solution_info"]),
        ]})

        self.bg_up = dev.createBindGroup({"layout": self.bgl_up, "entries": [
            _bg_entry(0, bufs["uniforms"]),
            _bg_entry(1, bufs["parameters"]),
            _bg_entry(2, bufs["prev_parameters"]),
            _bg_entry(3, bufs["deltas"]),
            _bg_entry(4, bufs["finished"]),
            _bg_entry(5, bufs["solution_info"]),
        ]})

        self.bg_co = dev.createBindGroup({"layout": self.bgl_co, "entries": [
            _bg_entry(0, bufs["uniforms"]),
            _bg_entry(1, bufs["finished"]),
            _bg_entry(2, bufs["states"]),
            _bg_entry(3, bufs["chi_squares"]),
            _bg_entry(4, bufs["prev_chi_squares"]),
            _bg_entry(5, bufs["parameters"]),
            _bg_entry(6, bufs["prev_parameters"]),
            _bg_entry(7, bufs["lambdas"]),
            _bg_entry(8, bufs["n_iterations"]),
            _bg_entry(9, bufs["solution_info"]),
        ]})

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------
    def _dispatch(self, pipeline, bind_group, x: int, y: int = 1, z: int = 1):
        """Submit one compute pass."""
        enc = self.device.createCommandEncoder()
        cp  = enc.beginComputePass()
        cp.setPipeline(pipeline)
        cp.setBindGroup(0, bind_group)
        cp.dispatchWorkgroups(x, y, z)
        cp.end()
        self.device.queue.submit([enc.finish()])

    def _dispatch_cv(self):
        self._dispatch(self.pipe_cv, self.bg_cv, self.n_fits)

    def _dispatch_cs(self):
        self._dispatch(self.pipe_cs, self.bg_cs, self.n_fits)

    def _dispatch_mh(self):
        self._dispatch(self.pipe_mh, self.bg_mh, self.n_fits)

    def _dispatch_gj(self):
        self._dispatch(self.pipe_gj, self.bg_gj, self.n_fits)

    def _dispatch_up(self):
        self._dispatch(self.pipe_up, self.bg_up, self.n_fits)

    def _dispatch_co(self):
        wgs = math.ceil(self.n_fits / 64)
        self._dispatch(self.pipe_co, self.bg_co, wgs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self,
        data: np.ndarray,           # [n_fits, n_points] float32
        x_values: np.ndarray,       # [n_points]         float32
        initial_params: np.ndarray, # [n_fits, 4]        float32
    ) -> FitResult:
        """
        Run the LM fitting loop on the GPU.

        Returns a FitResult named-tuple with parameters, states,
        chi_squares, and n_iterations arrays.
        """
        assert data.dtype == np.float32
        assert x_values.dtype == np.float32
        assert initial_params.dtype == np.float32
        assert data.shape == (self.n_fits, self.n_points)
        assert x_values.shape == (self.n_points,)
        assert initial_params.shape == (self.n_fits, N_PARAMS)

        dev  = self.device
        bufs = self.bufs
        nf   = self.n_fits
        np_  = self.n_points

        # ── Upload inputs ────────────────────────────────────────────
        upload(dev, bufs["data"],        data.ravel())
        upload(dev, bufs["x_values"],    x_values)
        upload(dev, bufs["parameters"],  initial_params.ravel())

        # ── Initialise scalars and control arrays ────────────────────
        upload(dev, bufs["lambdas"],
               np.full(nf, self.init_lambda, dtype=np.float32))
        upload_zeros(dev, bufs["finished"],         nf, np.int32)
        upload_zeros(dev, bufs["states"],           nf, np.int32)
        upload_zeros(dev, bufs["n_iterations"],     nf, np.int32)
        upload_zeros(dev, bufs["scaling_vectors"],  nf * N_PARAMS, np.float32)
        upload_zeros(dev, bufs["prev_chi_squares"], nf, np.float32)
        upload_zeros(dev, bufs["iteration_failed"], nf, np.int32)
        upload_zeros(dev, bufs["solution_info"],    nf, np.int32)

        # ── Initial curve evaluation and statistics ──────────────────
        write_uniforms(dev, bufs["uniforms"],
                       nf, np_, N_PARAMS, 0, self.max_iters, self.tolerance)
        self._dispatch_cv()
        self._dispatch_cs()

        # Copy initial chi_squares → prev_chi_squares
        copy_buffer(dev, bufs["chi_squares"], bufs["prev_chi_squares"], nf * 4)

        # ── Main LM loop ─────────────────────────────────────────────
        for iteration in range(self.max_iters):
            write_uniforms(dev, bufs["uniforms"],
                           nf, np_, N_PARAMS,
                           iteration, self.max_iters, self.tolerance)

            self._dispatch_mh()   # scale hessian diagonal
            self._dispatch_gj()   # solve H·delta = gradient
            self._dispatch_up()   # apply deltas, save prev_params

            self._dispatch_cv()   # recompute curve values
            self._dispatch_cs()   # recompute chi2 / grad / hessian

            self._dispatch_co()   # convergence + lambda update

            # Read finished flags back to CPU to check early exit.
            finished_cpu = download(dev, bufs["finished"], np.int32, nf)
            if np.all(finished_cpu != 0):
                break

        # ── Read results ─────────────────────────────────────────────
        params_out = download(dev, bufs["parameters"],  np.float32, nf * N_PARAMS
                              ).reshape(nf, N_PARAMS)
        states_out = download(dev, bufs["states"],      np.int32,   nf)
        chi2_out   = download(dev, bufs["chi_squares"], np.float32, nf)
        niter_out  = download(dev, bufs["n_iterations"],np.int32,   nf)

        return FitResult(
            parameters=params_out,
            states=states_out,
            chi_squares=chi2_out,
            n_iterations=niter_out,
        )


# ---------------------------------------------------------------------------
# Convenience: request a GPU device
# ---------------------------------------------------------------------------
async def gpu_device(power_preference: str = "high-performance") -> Device:
    # In a Pyodide web-worker, webgpu.platform.js starts as None because
    # the Jupyter-specific init_pyodide() is never called.  Set it here so
    # that init_device() can reach navigator.gpu.
    from webgpu import platform as _platform
    if _platform.is_pyodide and _platform.js is None:
        import js as _js
        _platform.js = _js
    return await init_device()
