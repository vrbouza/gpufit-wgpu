"""
Entry point: load Brillouin spectroscopy data via brimfile, then demonstrate
both single-pixel and full-volume GPU LM Lorentzian fitting.

Each spectrum is split at frequency = 0 and two independent fits are run:
one for the negative-frequency (Stokes) peak and one for the positive-frequency
(anti-Stokes) peak.

Run with:
    uv run python main.py
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import numpy as np

import brimfile as bf

from lm_fit import LMFitter, FitResult, gpu_device

TESTS_DIR = pathlib.Path(__file__).parent / "tests"
TESTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
ZARR_URL = "https://s3.embl.de/brim-example-files/zebrafish_eye_confocal.brim.zarr"


def load_brillouin_data():
    """Download and return (PSD, frequency) arrays from the example zarr file."""
    print(f"  Loading data from {ZARR_URL} …")
    f = bf.File(ZARR_URL)
    d = f.get_data()
    PSD, frequency, PSD_units, frequency_units = d.get_PSD_as_spatial_map()
    print(f"  PSD shape      : {PSD.shape}  units: {PSD_units}")
    print(f"  frequency shape: {frequency.shape}  units: {frequency_units}")
    return PSD.astype(np.float32), frequency.astype(np.float32)


# ---------------------------------------------------------------------------
# Lorentzian model (CPU side, for plotting)
# ---------------------------------------------------------------------------
def _lorentzian(x: np.ndarray, A: float, x0: float, gamma: float, offset: float) -> np.ndarray:
    u = (x - x0) / gamma
    return A / (1.0 + u ** 2) + offset


# ---------------------------------------------------------------------------
# Initial-parameter estimation (vectorised over all fits)
# ---------------------------------------------------------------------------
def estimate_initial_params(data: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    """
    Estimate [A, x0, gamma, offset] for every spectrum row in data.

    data     : [n_fits, n_points]  float32
    x_values : [n_points]          float32
    returns  : [n_fits, 4]         float32
    """
    offset    = np.min(data,  axis=1)
    amplitude = np.max(data,  axis=1) - offset
    peak_idx  = np.argmax(data, axis=1)
    center    = x_values[peak_idx]
    # Initial gamma: half the range of x_values (rough but safe starting point)
    gamma     = np.full(len(data),
                        abs(float(x_values[-1] - x_values[0])) / 4.0,
                        dtype=np.float32)
    params = np.stack([amplitude, center, gamma, offset], axis=1).astype(np.float32)
    return params


# ---------------------------------------------------------------------------
# Helpers: split a spectrum at freq = 0
# ---------------------------------------------------------------------------
def _split_spectrum(data: np.ndarray, x_values: np.ndarray):
    """
    Split data columns into negative-frequency and positive-frequency halves.

    data     : [n_fits, n_points]  float32
    x_values : [n_points]          float32

    Returns (neg_data, neg_x, pos_data, pos_x) all float32.
    """
    neg_mask = x_values < 0
    pos_mask = x_values > 0
    return (
        data[:, neg_mask], x_values[neg_mask],
        data[:, pos_mask], x_values[pos_mask],
    )


# ---------------------------------------------------------------------------
# Single-pixel fit
# ---------------------------------------------------------------------------
def fit_single_pixel(device, PSD: np.ndarray, frequency: np.ndarray):
    """Fit two Lorentzians to PSD[0,0,0,:] (one per frequency side) and plot."""
    print("\n── Single-pixel fit ────────────────────────────────────")

    freq      = frequency[0, 0, 0, :].astype(np.float32)
    spectrum  = PSD[0, 0, 0, :].astype(np.float32)
    data_full = spectrum.reshape(1, -1)

    neg_data, neg_freq, pos_data, pos_freq = _split_spectrum(data_full, freq)

    print(f"  n_points total   : {len(freq)}  "
          f"(neg={neg_data.shape[1]}, pos={pos_data.shape[1]})")

    neg_init = estimate_initial_params(neg_data, neg_freq)
    pos_init = estimate_initial_params(pos_data, pos_freq)

    neg_fitter = LMFitter(device, n_fits=1, n_points=neg_data.shape[1])
    pos_fitter = LMFitter(device, n_fits=1, n_points=pos_data.shape[1])

    neg_result = neg_fitter.fit(neg_data, neg_freq, neg_init)
    pos_result = pos_fitter.fit(pos_data, pos_freq, pos_init)

    state_names = {0: "CONVERGED", 1: "MAX_ITERATION", 2: "SINGULAR_HESSIAN"}
    for label, result in (("Stokes (neg)", neg_result), ("anti-Stokes (pos)", pos_result)):
        p = result.parameters[0]
        print(f"  [{label}]")
        print(f"    State      : {state_names.get(int(result.states[0]), '?')}")
        print(f"    Iterations : {result.n_iterations[0]}")
        print(f"    Chi²       : {result.chi_squares[0]:.6g}")
        print(f"    A={p[0]:.4g}  x0={p[1]:.6g} GHz  gamma={p[2]:.4g} GHz  offset={p[3]:.4g}")

    # ── Plot ──────────────────────────────────────────────────────────
    neg_p = neg_result.parameters[0]
    pos_p = pos_result.parameters[0]

    x_neg_dense = np.linspace(neg_freq[0],  neg_freq[-1],  400)
    x_pos_dense = np.linspace(pos_freq[0],  pos_freq[-1],  400)
    y_neg_fit   = _lorentzian(x_neg_dense, *neg_p)
    y_pos_fit   = _lorentzian(x_pos_dense, *pos_p)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(freq, spectrum, "o", ms=4, color="steelblue", label="Data", zorder=3)
    ax.plot(x_neg_dense, y_neg_fit, "-", lw=2, color="tomato",
            label=f"Stokes fit  x₀={neg_p[1]:.4f} GHz, γ={neg_p[2]:.4f} GHz")
    ax.plot(x_pos_dense, y_pos_fit, "-", lw=2, color="seagreen",
            label=f"anti-Stokes fit  x₀={pos_p[1]:.4f} GHz, γ={pos_p[2]:.4f} GHz")

    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("PSD")
    ax.set_title("Single-pixel Brillouin spectrum — Lorentzian LM fits (WebGPU)")
    ax.legend(framealpha=0.9, fontsize=9)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out_path = TESTS_DIR / f"single_pixel_fit.{ext}"
        fig.savefig(out_path, dpi=150)
        print(f"  Plot saved       : {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full-volume fit
# ---------------------------------------------------------------------------
def fit_all_pixels(device, PSD: np.ndarray, frequency: np.ndarray):
    """Fit two Lorentzians per pixel (one per frequency side) for the full volume."""
    print("\n── Full-volume fit ─────────────────────────────────────")

    nx, ny, nz, n_pts = PSD.shape
    n_fits    = nx * ny * nz
    freq      = frequency[0, 0, 0, :].astype(np.float32)
    data_all  = PSD.reshape(n_fits, n_pts).astype(np.float32)

    neg_data, neg_freq, pos_data, pos_freq = _split_spectrum(data_all, freq)

    print(f"  Spatial dims     : {nx} × {ny} × {nz}  →  {n_fits} fits")
    print(f"  n_points total   : {n_pts}  "
          f"(neg={neg_data.shape[1]}, pos={pos_data.shape[1]})")

    neg_init = estimate_initial_params(neg_data, neg_freq)
    pos_init = estimate_initial_params(pos_data, pos_freq)

    neg_fitter = LMFitter(device, n_fits=n_fits, n_points=neg_data.shape[1])
    pos_fitter = LMFitter(device, n_fits=n_fits, n_points=pos_data.shape[1])

    print("  Fitting Stokes (negative) peaks …")
    neg_result = neg_fitter.fit(neg_data, neg_freq, neg_init)

    print("  Fitting anti-Stokes (positive) peaks …")
    pos_result = pos_fitter.fit(pos_data, pos_freq, pos_init)

    for label, result in (("Stokes (neg)", neg_result), ("anti-Stokes (pos)", pos_result)):
        n_conv    = int(np.sum(result.states == 0))
        conv_frac = n_conv / n_fits * 100.0
        peak_freq = result.parameters[:, 1]
        gamma     = result.parameters[:, 2]
        print(f"  [{label}]")
        print(f"    Converged    : {n_conv}/{n_fits}  ({conv_frac:.1f} %)")
        print(f"    Mean chi²    : {np.mean(result.chi_squares):.6g}")
        print(f"    Mean iters   : {np.mean(result.n_iterations):.1f}")
        print(f"    Mean x0      : {np.mean(peak_freq):.6g} GHz  "
              f"(std: {np.std(peak_freq):.4g})")
        print(f"    Mean gamma   : {np.mean(gamma):.6g} GHz  "
              f"(std: {np.std(gamma):.4g})")

    return neg_result, pos_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== WebGPU Levenberg-Marquardt Lorentzian Fitting ===\n")

    print("Step 1: Initialising GPU …")
    device = gpu_device()
    print(f"  Adapter : {device.adapter.info}")

    print("\nStep 2: Loading Brillouin data …")
    PSD, frequency = load_brillouin_data()

    fit_single_pixel(device, PSD, frequency)
    fit_all_pixels(device, PSD, frequency)

    print("\nDone.")


if __name__ == "__main__":
    main()
