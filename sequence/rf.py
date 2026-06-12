import math
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt

from constants import GAMMA_RAD
from sequence.core import SequenceObject, determine_times
from sequence.gradient import Gradient
from simulate import BlochDispatcher


class RFPulse(SequenceObject):
    def __init__(self, times: torch.Tensor, waveform: torch.Tensor):
        super().__init__(times, waveform)
        self.asymmetry = 0.5
        self.transmit_phase = 0
        self.transmit_frequency = 0.0

    @property
    def is_phase_modulated(self) -> bool:
        phase_derivative = torch.diff(torch.angle(self._waveform))

        return torch.any((~phase_derivative.eq(0)) & (~torch.abs(phase_derivative).eq(torch.pi))).item()

    @property
    def waveform(self) -> torch.Tensor:
        asymmetrical_shift = round(self.n * (self.asymmetry - 0.5))

        rolled_waveform = torch.roll(super().waveform, asymmetrical_shift) * np.exp(1j * self.transmit_phase)
        if asymmetrical_shift < 0:
            rolled_waveform[asymmetrical_shift:] = 0
        else:
            rolled_waveform[:asymmetrical_shift] = 0

        return rolled_waveform * torch.exp(1j * 2 * torch.pi * self.transmit_frequency * self._current_times * 1e-6)

    @property
    def energy(self) -> float:
        """
        Calculate the energy of the RF pulse in μT²•s.
        """
        return torch.sum(torch.abs(self.waveform * 1e6) ** 2 * self.dt * 1e-6, dim=0).item()
  
    def get_optimal_amplitude(self, desired_flip_angle: float, df_range: float = 5000,
                              b1_range: tuple[float, float] = (0, 30e-6)) -> float:
        n_isochromats = max(1, int(df_range // 500)) * 10
        df = torch.cat((-torch.logspace(math.log(df_range / 2) / math.log(10), 0, n_isochromats // 2),
                        torch.zeros(1),
                        torch.logspace(0, math.log(df_range / 2) / math.log(10), n_isochromats // 2)))

        passband_filter = torch.exp(-(2 * df / df_range) ** 2)

        target_mxy, target_mz = math.sin(desired_flip_angle), math.cos(desired_flip_angle)

        min_b1, max_b1 = b1_range
        search_range = max_b1 - min_b1

        current_best_amplitude = (min_b1 + max_b1) / 2
        current_best_score = torch.inf

        num_points = 20
        max_iterations = 10
        tolerance = 1e-7  # 0.1uT tolerance for amplitude optimisation
        no_improvement_count = 0

        rf_copy = self.__copy__()
        dispatcher = BlochDispatcher()
        dispatcher.set_df(df)
        for iteration in range(0, max_iterations):
            if iteration == 0:
                b1_amplitudes = torch.linspace(min_b1, max_b1, num_points)
            else:
                b1_amplitudes = torch.clip(
                    current_best_amplitude + (torch.rand(num_points) - 0.5) * (max_b1 - min_b1),
                    min_b1, max_b1)

            mxy = torch.zeros((num_points, len(df), len(self) + 1), dtype=torch.float64)
            mz = torch.zeros((num_points, len(df), len(self) + 1), dtype=torch.float64)

            for i in range(num_points):
                rf_copy.amplitude = b1_amplitudes[i]
                dispatcher.set_rf(rf_copy)
                magnetisation = dispatcher.simulate(np.inf, np.inf, self.dt, force_cpu=True, verbose=False)

                mxy[i] = magnetisation.mxy
                mz[i] = magnetisation.mz

            mxy_cost = torch.sum(torch.abs((mxy[..., -1] - target_mxy) * passband_filter) ** 2, dim=1)
            mz_cost = torch.sum(torch.abs((mz[..., -1] - target_mz) * passband_filter) ** 2, dim=1)
            b1_cost = torch.exp(5e3 * b1_amplitudes) - 1

            scores = mxy_cost + mz_cost + b1_cost

            best_index = torch.argmin(scores)
            best_score = scores[best_index]
            best_amplitude = b1_amplitudes[best_index]

            if best_score < current_best_score:
                current_best_score = best_score
                current_best_amplitude = best_amplitude
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Update the search range
            if iteration == 0:
                num_points = 10
                search_range /= 4
            else:
                search_range *= math.exp(-iteration / 3)

            min_b1 = max(min_b1, current_best_amplitude - search_range / 2)
            max_b1 = min(max_b1, current_best_amplitude + search_range / 2)

            if search_range < tolerance or no_improvement_count >= 5:
                return current_best_amplitude

        return current_best_amplitude
  
    def get_fwhm(self, for_inversion: bool = False):
        df = torch.linspace(-5000, 5000, 10001)

        dispatcher = BlochDispatcher()
        dispatcher.set_df(df)
        dispatcher.set_rf(self)

        magnetisation = dispatcher.simulate(np.inf, np.inf, self.dt, force_cpu=True, verbose=False)

        if not for_inversion:
            mxy = magnetisation.mxy[..., -1]
            passband = df[torch.where(mxy >= 0.5)[0]]
        else:
            mz = magnetisation.mz[..., -1]
            passband = df[torch.where(mz <= -0.5)[0]]

        return passband[-1] - passband[0]

    def get_bloch_siegert_shift(self) -> float:
        """
        Calculate the Bloch-Siegert phase shift.

        Returns:
            float: The Bloch-Siegert phase shift.
        """
        omega_0 = 2 * torch.pi * self.transmit_offset
        omega_b1 = GAMMA_RAD * torch.abs(self._normalized_waveform)

        K_bs = torch.sum(omega_b1 ** 2 / (2 * omega_0) * self.dt * 1e-6, dim=0).item()

        return self.amplitude ** 2 * K_bs

    def display(self) -> None:
        fig, ax = plt.subplots(2, 1, sharex=True)

        ax[0].plot(self.times * 1e-3, self.magnitude * 1e6)
        ax[0].set_ylabel('Amplitude (uT)')
        ax[0].grid()

        ax[1].plot(self.times * 1e-3, self.phase)
        ax[1].set_yticks([-torch.pi, -torch.pi / 2, 0, torch.pi / 2, torch.pi], ['-π', '-π/2', '0', 'π/2', 'π'])
        ax[1].set_xlabel('Time (ms)')
        ax[1].set_ylabel('Phase (rad)')
        ax[1].grid()

        plt.subplots_adjust(hspace=0.1)
        plt.show()


def rect(duration: int, *, n: Optional[int] = None, dt: Optional[float] = None) -> RFPulse:
    """
    Generate a rectangular hard pulse with a choice of duration. Sampling can be specified by n or dt.

    Args:
        duration (int): The duration of the pulse in microseconds.
        n (int | None): The number of samples to generate. Defaults to None.
        dt (float | None): The sampling interval in microseconds. Defaults to None.

    Returns:
        RFPulse: An RFPulse object representing the generated rectangular hard pulse.
    """
    times = determine_times(duration, n, dt)
    waveform = torch.ones_like(times)

    return RFPulse(times, waveform)


def hamming_sinc(duration: int, bandwidth: float, *, n: Optional[int] = None, dt: Optional[float] = None) -> RFPulse:
    """
    Generate a Hamming windowed sinc pulse with a choice of duration and bandwidth.
    Sampling can be specified by n or dt.

    Args:
        duration (int): The duration of the pulse in microseconds.
        bandwidth (float): The bandwidth of the pulse in Hz.
        n (int | None): The number of samples to generate. Defaults to None.
        dt (float | None): The sampling interval in microseconds. Defaults to None.

    Returns:
        RFPulse: An RFPulse object representing the generated Hamming windowed sinc pulse.
    """
    times = determine_times(duration, n, dt)

    hamming_filter = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(len(times)) / (len(times) - 1))
    waveform = hamming_filter * torch.sinc(bandwidth * (times - duration / 2) * 1e-6)

    return RFPulse(times, waveform)


def gaussian(duration: int, bandwidth: float, *,
             n: Optional[int] = None, dt: Optional[float] = None) -> RFPulse:
    """
    Generate a Gaussian pulse with a choice of duration and bandwidth. Sampling can be specified by n or dt.

    Args:
        duration (int): The duration of the pulse in microseconds.
        bandwidth (float): The bandwidth of the pulse in Hz.
        n (int | None): The number of samples to generate. Defaults to None.
        dt (float | None): The sampling interval in microseconds. Defaults to None.

    Returns:
        RFPulse: An RFPulse object representing the generated Gaussian pulse.
    """
    times = determine_times(duration, n, dt)
    waveform = torch.exp(-(torch.pi * bandwidth * 1e-6 * (times - duration / 2)) ** 2 / (4 * math.log(2)))

    return RFPulse(times, waveform)


def fermi(duration: int, t0: int, alpha: float, *, n: Optional[int] = None, dt: Optional[float] = None) -> RFPulse:
    """
    Generate a Fermi pulse with a choice of duration and alpha. Sampling can be specified by n or dt.

    Args:
        duration (int): The duration of the pulse in microseconds.
        t0 (int): The pulse full-width half-maximum in microseconds.
        alpha (float): The scaling determining the transition steepness.
        n (int | None): The number of samples to generate. Defaults to None.
        dt (float | None): The sampling interval in microseconds. Defaults to None.

    Returns:
        RFPulse: An RFPulse object representing the generated Fermi pulse.
    """
    times = determine_times(duration, n, dt)
    pulse_width = 2 * t0 + 13.81 * alpha

    waveform = 1 / (1 + torch.exp((torch.abs(times - duration // 2) - t0 / 2) / alpha))
    waveform /= torch.max(waveform)

    return RFPulse(times, waveform)


def mao(duration: int, order: int, *, n: Optional[int] = None, dt: Optional[float] = None) -> RFPulse:
    """
    Generate a Mao pulse with a choice of duration and order. Sampling can be specified by n or dt.

    Args:
        duration (int): The duration of the pulse in microseconds.
        order (int): The order of the Mao pulse.
        n (int | None): The number of samples to generate. Defaults to None.
        dt (float | None): The sampling interval in microseconds. Defaults to None.

    Returns:
        RFPulse: An RFPulse object representing the generated Mao pulse.

    References:
    Mao J, Mareci TH, Andrew ER: Experimental study of optimal selective 180° radiofrequency pulses.
    Journal of Magnetic Resonance (1969) 1988:1–10.
    """
    times = determine_times(duration, n, dt)
    coefficients = {
        4: {
            "C": [-2899.206, 5947.067, -6501.857, 8066.327, -4549.156,
                  1626.142, -957.737, 438.683, -253.037, 113.687,
                  -82.998, 21.608, -34.128, -2.017, -16.472,
                  -6.672, -10.017, -6.742, -7.066, -5.836],
            "S": [0.0, -142.687, 318.650, -597.457, 408.421,
                  -209.755, 135.247, -82.075, 45.875, -29.556,
                  17.243, -9.123, 7.420, -1.966, 3.347,
                  0.275, 1.918, 0.911, 1.349, 0.946],
        },
        5: {
            "C": [-2942.885, 5826.248, -6110.593, 6642.794, -8688.408,
                  5780.883, -2437.819, 1634.761, -482.368, 599.012,
                  -305.518, 239.193, -93.785, 111.894, -17.147,
                  59.095, 7.119, 34.561, 13.097, 22.834],
            "S": [0.0, -143.696, 297.088, -489.633, 859.552,
                  -661.952, 367.254, -276.529, 172.948, -130.452,
                  80.254, -60.032, 31.346, -34.357, 8.583,
                  -20.575, -0.816, -13.345, -4.244, -9.718],
        },
        6: {
            "C": [-2878.787, 5902.160, -5893.040, 6227.559, -6716.007,
                  9045.384, -6663.395, 3093.739, -2200.903, 1226.520,
                  -943.397, 525.994, -439.315, 209.230, -221.335,
                  65.234, -124.294, 7.826, -73.653, -13.052],
            "S": [0.0, -143.793, 291.126, -457.941, 662.951,
                  -1122.466, 939.046, -537.718, 431.690, -279.077,
                  233.593, -148.334, 131.270, -71.222, 77.648,
                  -26.816, 49.969, -4.968, 33.534, 5.277],
        },
        7: {
            "C": [-2952.331, 5826.001, -5996.727, 6014.836, -6372.345,
                  6787.339, -9456.466, 7798.425, -3998.163, 3064.328,
                  -1831.484, 1507.142, -919.137, 812.674, -466.577,
                  460.505, -219.407, 275.812, -83.174, 174.720],
            "S": [0.0, -143.049, 293.800, -445.893, 629.209,
                  -886.068, 1422.129, -1276.539, 1575.395, -875.639,
                  861.162, -414.122, 444.786, -112.378, 154.587,
                  -176.814, 2.467, -70.689, 38.272, -86.905],
        }
    }

    if order not in coefficients:
        raise ValueError(f"Order {order} not supported. Must be 4, 5, 6, or 7.")

    C = coefficients[order]["C"]
    S = coefficients[order]["S"]

    waveform = torch.zeros_like(times, dtype=torch.complex128)
    for index, (c, s) in enumerate(zip(C, S)):
        waveform += (c * torch.cos(2 * torch.pi * index * times / duration) -
                     s * torch.sin(2 * torch.pi * index * times / duration))

    waveform /= torch.max(torch.abs(waveform))

    return RFPulse(times, waveform)


def chirp(duration: int, mu: float, n: Optional[int] = None, dt: Optional[float] = None) -> RFPulse:
    """
    Generate a chirp pulse with a choice of duration. Sampling can be specified by n or dt.

    Args:
        duration (int): The duration of the pulse in microseconds.
        mu (float): The linear frequency sweep amplitude in Hz.
        n (int | None): The number of samples to generate. Defaults to None.
        dt (float | None): The sampling interval in microseconds. Defaults to None.

    Returns:
        RFPulse: An RFPulse object representing the generated chirp pulse.
    """
    times = determine_times(duration, n, dt)
    magnitude = torch.ones_like(times, dtype=torch.complex128)

    frequency_sweep = -mu * (times - 0.5 * duration) / (0.5 * duration)
    phase = torch.cumsum(frequency_sweep, 0) * (times[1] - times[0]).item() * 1e-6
    waveform = magnitude * torch.exp(1j * phase)

    return RFPulse(times, waveform)


def hyperbolic_secant(duration: int, *, mu: float = 1, beta: Optional[float] = None,
                      bandwidth: Optional[float] = None,
                      n: Optional[int] = None, dt: Optional[float] = None) -> RFPulse:
    """
    Generate a hyperbolic secant pulse with a choice of duration, bandwidth, and two of mu, beta, and bandwidth.
    Sampling can be specified by number of samples or sampling interval.

    Args:
        duration (int): The duration of the pulse in microseconds.
        mu (float): A dimensionless scalar constant that scales the frequency sweep rate. Defaults to 1.
        beta (float | None): The modulation bandwidth in radians per second. Defaults to None.
        bandwidth (float | None): The bandwidth of the pulse in Hz. Defaults to None.
        n (int | None): The number of samples to generate.
        dt (float | None): The sampling interval in microseconds.

    Returns:
        RFPulse: An RFPulse object representing the generated hyperbolic secant pulse.

    References:
        Baum J, Tycko R, Pines A: Broadband and adiabatic inversion of a two-level system by phase-modulated pulses.
        Phys Rev A 1985:3435–3447.
    """
    # num_viable_parameters = len([param for param in [mu, beta, bandwidth] if param is not None])

    parameter_key = tuple(param is not None for param in [mu, beta, bandwidth])

    if sum(parameter_key) != 2:
        raise AttributeError(f'{sum(parameter_key)} of 3 parameters provided! '
                             f'Please provide 2 of the 3 parameters: mu, beta, and bandwidth!')

    if bandwidth is None:
        pass
    elif beta is None:
        beta = torch.pi * bandwidth / mu
    elif mu is None:
        mu = torch.pi * bandwidth / beta

    if beta is None or mu is None:
        raise ValueError("Failed to resolve parameters 'mu' and 'beta'.")

    times = determine_times(duration, n, dt)
    magnitude = 1 / torch.cosh(beta * (times - duration / 2) * 1e-6)
    frequency_sweep = -mu * beta * torch.tanh(beta * (times - duration / 2) * 1e-6)
    phase = torch.cumsum(frequency_sweep, 0) * (times[1] - times[0]).item() * 1e-6

    waveform = magnitude * torch.exp(1j * phase)

    return RFPulse(times, waveform)


def foci(duration: int, gradient_ratio: float, *,
         mu: float = 1, beta: Optional[float] = None, bandwidth: Optional[float] = None,
         n: Optional[int] = None, dt: Optional[float] = None) -> tuple[RFPulse, Gradient]:
    parameter_key = tuple(param is not None for param in [mu, beta, bandwidth])

    if sum(parameter_key) != 2:
        raise AttributeError(f'{sum(parameter_key)} of 3 parameters provided! '
                             f'Please provide 2 of the 3 parameters: mu, beta, and bandwidth!')

    if bandwidth is None:
        pass
    elif beta is None:
        beta = torch.pi * bandwidth / mu
    elif mu is None:
        mu = torch.pi * bandwidth / beta

    times = determine_times(duration, n, dt)

    amplitude = torch.where(torch.cosh(beta * (times - duration / 2) * 1e-6) < 1 / gradient_ratio,
                            torch.cosh(beta * (times - duration / 2) * 1e-6), 1 / gradient_ratio)

    magnitude = amplitude / torch.cosh(beta * (times - duration / 2) * 1e-6)
    frequency_sweep = -mu * amplitude * beta * torch.tanh(beta * (times - duration / 2) * 1e-6)
    phase = torch.cumsum(frequency_sweep, 0) * (times[1] - times[0]).item() * 1e-6

    rf = RFPulse(times, magnitude * torch.exp(1j * phase))
    grad = Gradient(times, amplitude)

    return rf, grad


def wurst(duration: int, bandwidth: float, *, exponent: float = 20,
          n: Optional[int] = None, dt: Optional[float] = None) -> RFPulse:
    """
    Generate a wideband, uniform rate, smooth truncation (WURST) pulse with a choice of duration, bandwidth
    and exponent factor. Sampling can be specified by number of samples or sampling interval.

    Args:
        duration (int): The duration of the pulse in microseconds.
        bandwidth (float): The modulation bandwidth in Hz.
        exponent (float): The exponent of the modulation. Defaults to 20.
        n (int | None): The number of samples to generate. Defaults to None.
        dt (float | None): The sampling interval in microseconds. Defaults to None.

    Returns:
        RFPulse: An RFPulse object representing the generated WURST pulse.

    References:
        Kupce Ē, Freeman R: Stretched Adiabatic Pulses for Broadband Spin Inversion.
        Journal of Magnetic Resonance, Series A 1995:246–256.
    """
    times = determine_times(duration, n, dt)

    magnitude = 1 - torch.abs(torch.sin(torch.pi * (times + duration / 2) / duration)) ** exponent
    frequency_sweep = torch.linspace(-bandwidth / 2, bandwidth / 2, len(times))

    phase = torch.cumsum(frequency_sweep, 0) * (times[1] - times[0]).item() * 1e-6
    waveform = magnitude * torch.exp(1j * phase)

    return RFPulse(times, waveform)


def goia_wurst(duration: int, bandwidth: float, *,
               f: float = 0.9, b1_order: int = 16, grad_order: int = 4,
               n: Optional[int] = None, dt: Optional[float] = None) -> tuple[RFPulse, Gradient]:
    """
    Generate a gradient offset independent adiabaticity (GOIA) wideband, uniform rate, smooth truncation (WURST) pulse
    with a choice of duration, bandwidth, gradient modulation factor, order for B1 modulation and order for gradient
    modulation. Sampling can be specified by number of samples or sampling interval.

    Args:
        duration (int): The duration of the pulse in microseconds.
        bandwidth (float): The modulation bandwidth in Hz.
        f (float): The gradient modulation factor. Defaults to 0.9.
        b1_order (int): The order of the B1 modulation. Defaults to 16.
        grad_order (int): The order of the gradient modulation. Defaults to 4.
        n (int | None): The number of samples to generate. Defaults to None.
        dt (float | None): The sampling interval in microseconds. Defaults to None.

    Returns:
        RFPulse: An RFPulse object representing the generated WURST pulse.

    References:
        Andronesi OC, Ramadan S, Ratai E-M, Jennings D, Mountford CE, Sorensen AG: Spectroscopic imaging with improved
        gradient modulated constant adiabaticity pulses on high-field clinical scanners.
        Journal of Magnetic Resonance 2010:283–293.
    """
    times = determine_times(duration, n, dt)

    amplitude = 1 - torch.abs(torch.sin(torch.pi * (times + duration / 2) / duration)) ** b1_order
    gradient = (1 - f) + f * torch.abs(torch.sin(torch.pi * (times + duration / 2) / duration)) ** grad_order

    frequency_sweep = torch.cumsum(amplitude ** 2 / gradient, dim=0) * times[-1] / len(times)
    frequency_sweep = frequency_sweep - frequency_sweep[len(times) // 2 + 1]
    frequency_sweep = frequency_sweep * gradient
    frequency_sweep = frequency_sweep / torch.max(torch.abs(frequency_sweep)) * bandwidth / 2

    phase = torch.cumsum(frequency_sweep, 0) * (times[1] - times[0]).item() * 1e-6
    waveform = amplitude * torch.exp(1j * phase)

    return RFPulse(times, waveform), Gradient(times, gradient)
