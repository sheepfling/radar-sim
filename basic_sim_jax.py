import logging
from argparse import Namespace
from dataclasses import dataclass
from logging import getLogger
from typing import Annotated, Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

# types
Vec3 = Annotated[jnp.ndarray, (3,)]
Array1 = Annotated[jnp.ndarray, (...,)]
Array3 = Annotated[jnp.ndarray, (..., 3)]

log = getLogger(__name__)
logging.basicConfig(level=logging.INFO)

speed_of_light = 299792458.0

def next_fast_len(x: int) -> int:
    """Return a power-of-two length >= x for efficient FFT usage."""
    return 1 if x <= 1 else 1 << (x - 1).bit_length()
####

def fftconvolve(a: jnp.ndarray, b: jnp.ndarray, mode: Literal['full', 'same', 'valid'] = 'full') -> jnp.ndarray:
    """FFT-based convolution using jax.numpy.fft. Works with complex arrays."""
    s1 = a.shape[0]
    s2 = b.shape[0]
    n = next_fast_len(s1 + s2 - 1)
    A = jnp.fft.fft(a, n)
    B = jnp.fft.fft(b, n)
    c = jnp.fft.ifft(A * B)
    if mode == 'full':
        return c[: s1 + s2 - 1]
    elif mode == 'same':
        start = (s2 - 1) // 2
        return c[start: start + s1]
    elif mode == 'valid':
        start = s2 - 1
        return c[start: start + s1 - s2 + 1]
    else:
        raise ValueError("mode must be 'full', 'same', or 'valid'")
    ####
####

def time_from_range_2way(distance: jnp.ndarray | float) -> jnp.ndarray | float:
    return distance * (2.0 / speed_of_light)
####

def range_from_time_2way(time: jnp.ndarray | float) -> jnp.ndarray | float:
    return time * (speed_of_light / 2.0)
####

@dataclass
class Scatterer:
    position: Vec3
    velocity: Vec3
    rcs_amplitude: complex
####

@dataclass
class GroupedScatterers:
    positions: Array3
    velocities: Array3
    rcs_amplitudes: Array1
####

def create_group_from_individual_scatterers(scatterers: list[Scatterer]) -> GroupedScatterers:
    """Pack python Scatterer list into JAX arrays on default device."""
    positions = jnp.vstack([jnp.asarray(s.position) for s in scatterers])
    velocities = jnp.vstack([jnp.asarray(s.velocity) for s in scatterers])
    rcs_amplitudes = jnp.asarray([s.rcs_amplitude for s in scatterers])
    return GroupedScatterers(positions=positions, velocities=velocities, rcs_amplitudes=rcs_amplitudes)
####

def make_lfm_chirp(bw: float, T: float) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Return a baseband LFM chirp function s(t).
    Pulse centered at 0 with width T and bandwidth bw.
    """
    k = 1j * jnp.pi * bw / T
    lb = -T / 2.0
    rb = T / 2.0

    def w(t: jnp.ndarray) -> jnp.ndarray:
        # vectorized within JAX
        out = jnp.zeros_like(t, dtype=jnp.complex128)
        mask = (t >= lb) & (t <= rb)
        tt = jnp.where(mask, t, 0.0)
        phi = k * (tt ** 2)
        out = jnp.where(mask, jnp.exp(phi), out)
        return out
    ####
    return w
####

def make_rx_baseband(waveform: Callable[[jnp.ndarray], jnp.ndarray], fc: float) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Return a JAX-compatible rx(t, tau) = s(t - tau) * exp(2j*pi*fc*tau)."""
    k = 2j * jnp.pi * fc
    def rx(t: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
        return waveform(t - tau) * jnp.exp(k * tau)
    ####
    return rx
####

@jax.jit
def _compute_pos_t(positions: jnp.ndarray, velocities: jnp.ndarray, t_rx: jnp.ndarray, radar_pos: jnp.ndarray) -> jnp.ndarray:
    """
    Compute instantaneous positions for each time and scatterer:
    shape -> (N_time, N_scat, 3)
    """
    # velocities: (N_scat, 3), t_rx: (N_time,)
    pos_t = velocities[None, :, :] * t_rx[:, None, None] + (positions - radar_pos)[None, :, :]
    return pos_t
####

@jax.jit
def _compute_tau_from_pos_t(pos_t: jnp.ndarray) -> jnp.ndarray:
    """Compute two-way delay tau(t, scat, view) assuming pos_t shape (N_time, N_scat, 3) and single view rhat provided later."""
    R_t = jnp.linalg.norm(pos_t, axis=-1)
    return time_from_range_2way(R_t)
####

@jax.jit
def simulate_received_baseband_jax(
        grouped_scatterers: GroupedScatterers,
        rx_baseband: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        t_rx: jnp.ndarray,
        tx_power: float = 1.0,
        radar_pos: jnp.ndarray | None = None,
        rhat: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Vectorized simulation of received baseband.
    - grouped_scatterers.positions: (N_scat, 3)
    - grouped_scatterers.velocities: (N_scat, 3)
    - grouped_scatterers.rcs_amplitudes: (N_scat,)
    - t_rx: (N_time,)
    - rhat: optional (3,) or (N_views, 3). If None, assumes rhat = [0,0,1].
    Returns: rx shape (N_time, N_views) or (N_time,) if single view.
    """
    if radar_pos is None:
        radar_pos = jnp.zeros((3,), dtype=jnp.float64)
    if rhat is None:
        rhat = jnp.array([[0.0, 0.0, 1.0]], dtype=jnp.float64)
    # normalize rhat
    rhat = jnp.atleast_2d(rhat)
    rhat = rhat / jnp.linalg.norm(rhat, axis=-1, keepdims=True)

    K = (tx_power ** 0.5) / (4.0 * jnp.pi)
    positions = grouped_scatterers.positions
    velocities = grouped_scatterers.velocities
    amps = grouped_scatterers.rcs_amplitudes

    # pos_t shape (N_time, N_scat, 3)
    pos_t = _compute_pos_t(positions, velocities, t_rx, radar_pos)

    # For multiple views, compute projection along each rhat:
    # pos_proj shape (N_time, N_scat, N_views)
    # use einsum to avoid huge temporaries
    # vel_proj + pos_proj = rhat dot (pos + vel*t)
    # pos_t: (N_time, N_scat, 3), rhat: (N_views, 3)
    pos_proj = jnp.einsum('tnj,vj->tnv', pos_t, rhat)  # (N_time, N_scat, N_views)
    tau = time_from_range_2way(pos_proj)
    # s_rx: (N_time, N_scat, N_views)
    # rx_baseband expects JAX arrays and is JAX-traceable
    s_rx = rx_baseband(t_rx[:, None, None], tau)  # broadcast t over scatterers and views
    # amplitude: (N_time, N_scat, N_views) broadcasting amps over time and views
    amp = (K / (pos_proj ** 2)) * amps[None, :, None]
    rx = jnp.sum(s_rx * amp, axis=1)  # sum over scatterers -> (N_time, N_views)
    # if single view, squeeze
    if rx.shape[-1] == 1:
        rx = rx[:, 0]
    return rx
####

def apply_matched_filter(rx: jnp.ndarray, rx_expected: jnp.ndarray) -> jnp.ndarray:
    """Matched filter using FFT convolution. Inputs can be complex."""
    h = jnp.conjugate(rx_expected[::-1])
    return fftconvolve(rx, h, mode="same")
####

def plot_complex_signal(
        time: np.ndarray,
        signal: np.ndarray,
        time_label: str | None,
        title: str | None = None,
) -> list[plt.Figure]:
    figs = []
    fig = plt.figure()
    show_legend = False
    if np.iscomplexobj(signal):
        plt.plot(time, signal.real, label=r'Re(s(t))')
        plt.plot(time, signal.imag, label=r'Im(s(t))')
        plt.plot(time, np.abs(signal), 'r--', label=r'|s(t)|')
        show_legend = True
    else:
        plt.plot(time, signal)
    if time_label:
        plt.xlabel(time_label)
    plt.ylabel('Signal')
    if title:
        plt.title(title)
    if show_legend:
        plt.legend()
    plt.grid(True)
    figs.append(fig)
    return figs
####

def plot_baseband(baseband: Callable[[jnp.ndarray], jnp.ndarray], pulse_width: float, fs: float) -> list[plt.Figure]:
    Ts = 1.0 / fs
    t = np.arange(-pulse_width / 2 * 1.1, pulse_width / 2 * 1.1, Ts)
    # evaluate on host arrays
    tb = jnp.asarray(t)
    sig = baseband(tb)
    sig_h = np.asarray(jax.device_get(sig))
    return plot_complex_signal(time=t * 1e6, signal=sig_h, time_label='Time [us]', title='Baseband Signal')
####

def main():
    opts = Namespace()
    opts.plot_baseband = True
    opts.plot_received_baseband = True
    opts.plot_matched_filter = True

    figs = []

    tx_power = 1.0
    pulse_width = 5e-6
    bandwidth = 10e6
    f0 = 10e9
    fs = 5e9

    s_baseband = make_lfm_chirp(bw=bandwidth, T=pulse_width)
    rx_baseband = make_rx_baseband(waveform=s_baseband, fc=f0)

    min_range = range_from_time_2way(pulse_width)
    print(f'{pulse_width=}; {range_from_time_2way(pulse_width)=}')

    if opts.plot_baseband:
        figs.extend(plot_baseband(baseband=s_baseband, pulse_width=pulse_width, fs=fs))
    ####

    radar_pos = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64)

    s1 = Scatterer(position=jnp.array([2 * min_range - 5.0, 0.0, 0.0]),
                   velocity=jnp.array([-5.0, 0.0, 0.0]),
                   rcs_amplitude=1.0 + 0j)
    s2 = Scatterer(position=jnp.array([2 * min_range + 10.0, 2.0, 0.0]),
                   velocity=jnp.array([0.0, 0.0, 0.0]),
                   rcs_amplitude=0.8 * jnp.exp(1j * jnp.pi / 4))
    s3 = Scatterer(position=jnp.array([1300.0, 0.0, 0.0]),
                   velocity=jnp.array([0.0, 0.0, 5.0]),
                   rcs_amplitude=0.8 * jnp.exp(1j * jnp.pi / 4))

    # build a modest cloud of scatterers
    rng = np.random.default_rng(0)
    scatterers = [s1, s2, s3]
    for i in range(8000):
        p = jnp.asarray(np.array([1300.0, 0.0, 0.0]) + rng.normal(0, 10, size=3))
        v = jnp.asarray(np.array([0.0, 0.0, 5.0]) + rng.normal(0, 0.1, size=3))
        amp = 0.8 * jnp.exp(1j * (rng.random() * 2 * np.pi)) * (rng.exponential(scale=0.1))
        scatterers.append(Scatterer(position=p, velocity=v, rcs_amplitude=amp))
    ####

    grouped = create_group_from_individual_scatterers(scatterers)

    center_range = 2 * min_range
    window_m = min_range * 1.1

    min_range_w = center_range - window_m / 2.0
    max_range_w = center_range + window_m / 2.0
    min_time = 2.0 * min_range_w / speed_of_light
    max_time = 2.0 * max_range_w / speed_of_light
    dt = 1.0 / fs
    t_rx = jnp.arange(min_time, max_time, dt, dtype=jnp.float64)

    # simulate (this will JIT-compile on first call)
    rx = simulate_received_baseband_jax(
        positions=grouped.positions,
        velocities=grouped.velocities,
        amplitudes=grouped.rcs_amplitudes,
        rx_baseband=rx_baseband,
        t_rx=t_rx,
        radar_pos=radar_pos,
        tx_power=tx_power,
        rhat=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64),
    )

    rx_host = np.asarray(jax.device_get(rx))
    time_range_m = np.asarray(jax.device_get(range_from_time_2way(t_rx)))

    if opts.plot_received_baseband:
        figs.extend(plot_complex_signal(time=time_range_m, signal=rx_host, title='Received Baseband', time_label='Range [m]'))

    expected = rx_baseband(t_rx, time_from_range_2way(center_range))
    expected_host = np.asarray(jax.device_get(expected))

    s_mf = apply_matched_filter(jnp.asarray(rx_host), jnp.asarray(expected_host))
    s_mf_host = np.asarray(jax.device_get(s_mf))

    log.debug(f'{t_rx.shape=}; {rx_host.shape=}; {s_mf_host.shape=}')

    if opts.plot_matched_filter:
        figs.extend(plot_complex_signal(time=time_range_m, signal=s_mf_host, title='Matched Filter Output', time_label='Range [m]'))

    if figs:
        plt.show()
    for fig in figs:
        plt.close(fig)
    ####
####

if __name__ == "__main__":
    main()
####