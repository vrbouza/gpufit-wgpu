"""
Buffer allocation, upload, and download helpers.

Each buffer maps to a cuda Device_Array from Gpufit.
All storage buffers have STORAGE | COPY_SRC | COPY_DST usage.
"""
import struct
import numpy as np
import wgpu


def _storage(device: wgpu.GPUDevice, size_bytes: int) -> wgpu.GPUBuffer:
    return device.create_buffer(
        size=size_bytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
    )


def _uniform(device: wgpu.GPUDevice, size_bytes: int) -> wgpu.GPUBuffer:
    return device.create_buffer(
        size=size_bytes,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )


def allocate_buffers(device: wgpu.GPUDevice, n_fits: int, n_points: int, n_params: int = 4):
    """Allocate all GPU buffers and return them as a dict."""
    f32 = 4  # bytes per float32
    i32 = 4  # bytes per int32

    bufs = {
        # Data
        "data":            _storage(device, n_fits * n_points * f32),
        "parameters":      _storage(device, n_fits * n_params * f32),
        "prev_parameters": _storage(device, n_fits * n_params * f32),
        "values":          _storage(device, n_fits * n_points * f32),
        "derivatives":     _storage(device, n_fits * n_points * n_params * f32),
        # Statistics
        "chi_squares":     _storage(device, n_fits * f32),
        "prev_chi_squares":_storage(device, n_fits * f32),
        "gradients":       _storage(device, n_fits * n_params * f32),
        "hessians":        _storage(device, n_fits * n_params * n_params * f32),
        "deltas":          _storage(device, n_fits * n_params * f32),
        # LM control
        "lambdas":         _storage(device, n_fits * f32),
        "scaling_vectors": _storage(device, n_fits * n_params * f32),
        "finished":        _storage(device, n_fits * i32),
        "states":          _storage(device, n_fits * i32),
        "n_iterations":    _storage(device, n_fits * i32),
        "iteration_failed":_storage(device, n_fits * i32),
        "solution_info":   _storage(device, n_fits * i32),
        # Shared x-axis (one set of x-values used for all fits)
        "x_values":        _storage(device, n_points * f32),
        # Uniforms (32 bytes: 8 x u32/f32)
        "uniforms":        _uniform(device, 32),
    }
    return bufs


def upload(device: wgpu.GPUDevice, buf: wgpu.GPUBuffer, arr: np.ndarray):
    """Write a numpy array into a GPU buffer (host→device)."""
    device.queue.write_buffer(buf, 0, arr.tobytes())


def upload_zeros(device: wgpu.GPUDevice, buf: wgpu.GPUBuffer, n_elements: int, dtype=np.float32):
    """Upload a zero-filled array of n_elements into buf."""
    upload(device, buf, np.zeros(n_elements, dtype=dtype))


def download(device: wgpu.GPUDevice, buf: wgpu.GPUBuffer, dtype, count: int) -> np.ndarray:
    """Read count elements of dtype from a GPU buffer back to CPU."""
    raw = device.queue.read_buffer(buf)
    return np.frombuffer(raw, dtype=dtype)[:count].copy()


def write_uniforms(device: wgpu.GPUDevice, buf: wgpu.GPUBuffer,
                   n_fits: int, n_points: int, n_params: int,
                   iteration: int, max_iterations: int, tolerance: float):
    """Pack and upload the uniforms struct (32 bytes)."""
    data = struct.pack(
        "5If2I",           # 5 unsigned ints, 1 float, 2 unsigned ints (padding)
        n_fits,
        n_points,
        n_params,
        iteration,
        max_iterations,
        tolerance,
        0,                 # pad0
        0,                 # pad1
    )
    device.queue.write_buffer(buf, 0, data)


def copy_buffer(device: wgpu.GPUDevice,
                src: wgpu.GPUBuffer, dst: wgpu.GPUBuffer, size_bytes: int):
    """GPU-side buffer copy (COPY_SRC → COPY_DST)."""
    encoder = device.create_command_encoder()
    encoder.copy_buffer_to_buffer(src, 0, dst, 0, size_bytes)
    device.queue.submit([encoder.finish()])
