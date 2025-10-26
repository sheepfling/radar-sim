import logging
from argparse import Namespace
from dataclasses import dataclass
from logging import getLogger
from typing import Annotated, Literal, overload, Callable

import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
from numpy import random, arange, array, conj, ndarray, pi, zeros_like, exp, iscomplexobj, newaxis, zeros, vstack, \
    sum as asum
from numpy.fft import fft, ifft
from numpy.linalg import norm


################################################
log = getLogger(__name__)

speed_of_light = 299792458.0

# Vector type for 3D vectors
Vec3 = Annotated[ndarray, (3,)]

def next_fast_len(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()
####

def fftconvolve(
        a: ndarray, b: ndarray, mode: Literal['full', 'same', 'valid'] = 'full',
) -> ndarray:
    # Determine output size
    s1 = a.shape[0]
    s2 = b.shape[0]
    n = next_fast_len(s1 + s2 - 1)  # efficient FFT length

    # FFT-based convolution
    A = fft(a, n)
    B = fft(b, n)
    c = ifft(A * B)

    # Slice according to mode
    if mode == 'full':
        return c[:s1 + s2 - 1]
    elif mode == 'same':
        start = (s2 - 1) // 2
        return c[start:start + s1]
    elif mode == 'valid':
        start = s2 - 1
        return c[start:start + s1 - s2 + 1]
    else:
        raise ValueError("mode must be 'full', 'same', or 'valid'")
    ####
####

@overload
def time_from_range_2way(distance: float) -> float: ...
@overload
def time_from_range_2way(distance: ndarray) -> ndarray: ...

def time_from_range_2way(distance: float | ndarray) -> float | ndarray:
    return distance * (2. / speed_of_light)
####

@overload
def range_from_time_2way(time: float) -> float: ...
@overload
def range_from_time_2way(time: ndarray) -> ndarray: ...

def range_from_time_2way(time: float | ndarray) -> float | ndarray:
    return time * (speed_of_light / 2.)
####

@dataclass
class Scatterer:
    position: Vec3  # initial 3D position (m)
    velocity: Vec3  # 3D velocity (m/s)
    rcs_amplitude: complex  # complex amplitude (includes RCS/phase)
####

@dataclass
class GroupedScatterers:
    positions: Annotated[ndarray, (..., 3)]
    velocities: Annotated[ndarray, (..., 3)]
    rcs_amplitudes: Annotated[ndarray, (...,)]
####

def create_group_from_individual_scatterers(scatterers: list[Scatterer]) -> GroupedScatterers:
    out = GroupedScatterers(
        positions=vstack([s.position for s in scatterers]),
        velocities=vstack([s.velocity for s in scatterers]),
        rcs_amplitudes=array([s.rcs_amplitude for s in scatterers]),
    )
    return out
####

def simulate_received_baseband(
        grouped_scatterers: GroupedScatterers,
        rx_baseband: Callable[[ndarray, ndarray], ndarray],
        t_rx: ndarray,
        tx_power: float = 1.0,
        radar_pos: Vec3 | None = None,
) -> ndarray:
    if radar_pos is None:
        radar_pos = zeros(3, dtype=float)
    ####
    gs = grouped_scatterers
    # For each scatterer compute R(t), tau(t), amplitude, delayed waveform, and carrier phase
    K = (tx_power ** .5) / (4 * pi)
    pos_t = (
            (gs.velocities[newaxis, ...] * t_rx[..., newaxis, newaxis]) +
            (gs.positions - radar_pos[newaxis, ...])[newaxis, ...]
    )
    log.debug(f'{pos_t.shape=}')
    R_t = norm(pos_t, axis=-1)
    log.debug(f'{R_t.shape=}')
    # \boxed{s_\text{rx}(t) = s(t - \tau(t)) \cdot e^{- j 2 \pi f_c \tau(t)}}
    tau_t = time_from_range_2way(R_t)
    s_rx = rx_baseband(t_rx[..., newaxis], tau_t)
    amp = K / (R_t ** 2) * gs.rcs_amplitudes[newaxis, ...]
    rx = asum(s_rx * amp, axis=-1)
    return rx
####

def apply_matched_filter(rx: ndarray, rx_expected: ndarray) -> ndarray:
    """
    Matched filter via convolution with time-reversed conjugate transmit.
    tx should be sampled on the same time axis as rx or be a callable evaluated on rx times.
    """
    h = conj(rx_expected[::-1])
    return fftconvolve(rx, h, mode="same")
####

# Example LFM baseband chirp (centered on t=0 or t in arbitrary support)
def make_lfm_chirp(bw: float, T: float) -> Callable[[ndarray], ndarray]:
    # Makes -T/2<=0<=T/2; 0 centered width T pulse with bw bandwidth
    k = 1j * pi * bw / T
    lb = -T / 2.
    rb = T / 2.
    def w(t: ndarray) -> ndarray:
        out = zeros_like(t, dtype=complex)
        mask = (lb <= t) & (t <= rb)
        tt = t[mask]
        phi = k * (tt ** 2)
        out[mask] = exp(phi)
        return out
    ####
    return w
####

def make_rx_baseband(waveform: Callable[[ndarray | float], ndarray], fc: float) -> Callable[
    [ndarray | float, ndarray | float], ndarray]:
    k = 2j * pi * fc
    def rx(t: ndarray | float, tau: ndarray | float) -> ndarray:
        return waveform(t - tau) * exp(k * tau)
    ####
    return rx
####


def plot_complex_signal(
        time: ndarray,
        signal: ndarray,
        time_label: str | None,
        title: str | None = None,
) -> list[Figure]:
    figs = []
    #####
    fig = plt.figure()
    show_legend = False
    if iscomplexobj(signal):
        plt.plot(time, signal.real, label=r'$\Re(s(t))$')
        plt.plot(time, signal.imag, label=r'$\Im(s(t))$')
        plt.plot(time, abs(signal), 'r--', label=r'$|s(t)|$')
        show_legend = True
    else:
        plt.plot(time, signal)
    ####
    if time_label:
        plt.xlabel(time_label)
    ####
    plt.ylabel('Signal')
    if title:
        plt.title(title)
    ####
    if show_legend:
        plt.legend()
    ####
    plt.grid(True)
    figs.append(fig)
    return figs
#####

def plot_baseband(baseband: Callable[[ndarray], ndarray], pulse_width: float, fs: float) -> list[Figure]:
    Ts = 1. / fs
    t = arange(-pulse_width / 2 * 1.1, pulse_width / 2 * 1.1, Ts)
    return plot_complex_signal(
        time=t * 1e6,
        signal=baseband(t),
        time_label=r'Time [$\mu t$]',
        title='Baseband Signal',
    )
####

# ------------------------
# Minimal demo
# ------------------------
def main():
    opts = Namespace()
    opts.plot_baseband = True
    opts.plot_received_baseband = True
    opts.plot_matched_filter = True

    figs = []

    tx_power = 1.
    pulse_width = 5e-6
    bandwidth = 10e6
    f0 = 10e9
    fs = 5e9
    s_baseband = make_lfm_chirp(
        bw=bandwidth,
        T=pulse_width,
    )
    rx_baseband = make_rx_baseband(
        waveform=s_baseband,
        fc=f0,
    )

    min_range = range_from_time_2way(pulse_width)

    print(f'{pulse_width=}; {range_from_time_2way(pulse_width)=}')

    if opts.plot_baseband:
        figs.extend(plot_baseband(
            baseband=s_baseband,
            pulse_width=pulse_width,
            fs=fs,
        ))
    ####

    radar_pos = array([0.0, 0.0, 0.0])

    s1 = Scatterer(
        position=array([2 * min_range - 5., 0.0, 0.0]),
        velocity=array([-5.0, 0.0, 0.0]),
        rcs_amplitude=1.0 + 0j,
    )
    s2 = Scatterer(
        position=array([2 * min_range + 10., 2.0, 0.0]),
        velocity=array([0.0, 0.0, 0.0]),
        rcs_amplitude=0.8 * exp(1j * pi / 4)
    )
    s3 = Scatterer(
        position=array([1300., 0., 0., ]),
        velocity=array([0.0, 0.0, 5.0]),
        rcs_amplitude=0.8 * exp(1j * pi / 4)
    )
    scatterers = [
        s1,
        s2,
        s3,
    ]
    for i in range(8000):
        scatterers.append(Scatterer(
            position=array([1300., 0., 0., ]) + random.normal(0, 10, size=3),
            velocity=array([0.0, 0.0, 5.0]) + random.normal(0, 0.1, size=3),
            rcs_amplitude=0.8 * exp(1j * random.uniform(0, 2 * pi)) * random.exponential(scale=.1),
        ))
    ####

    center_range = 2 * min_range
    window_m = min_range * 1.1

    # observation times - choose long enough to contain returns
    min_range = center_range - window_m / 2.
    max_range = center_range + window_m / 2.
    min_time = 2 * min_range / speed_of_light
    max_time = 2 * max_range / speed_of_light
    dt = 1.0 / fs
    t_rx = arange(min_time, max_time, dt)
    rx = simulate_received_baseband(
        grouped_scatterers=create_group_from_individual_scatterers(scatterers),
        rx_baseband=rx_baseband,
        t_rx=t_rx,
        radar_pos=radar_pos,
        tx_power=tx_power,
    )

    if opts.plot_received_baseband:
        figs.extend(plot_complex_signal(
            time=range_from_time_2way(t_rx),
            signal=rx,
            title='Received Baseband',
            time_label=r'Range [m]',  # Received Time [$\mu s$]',
        ))
    ####

    s_mf = apply_matched_filter(
        rx=rx,
        rx_expected=rx_baseband(
            t_rx,
            time_from_range_2way(center_range),
        )
    )
    log.debug(f'{t_rx.shape=}; {rx.shape=}; {s_mf.shape=}')

    if opts.plot_matched_filter:
        figs.extend(plot_complex_signal(
            time=range_from_time_2way(t_rx),
            signal=s_mf,
            title='Matched Filter Output',
            time_label=r'Range [m]',  # Received Time [$\mu s$]',
        ))
    ####
    if figs:
        plt.show()
    ####
    for fig in figs:
        plt.close(fig)
    ####
####

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    main()
####
