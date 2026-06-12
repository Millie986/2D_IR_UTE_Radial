from typing import Optional

import torch
from matplotlib import pyplot as plt

from constants import GAMMA
from sequence.core import SequenceObject, determine_times


class Gradient(SequenceObject):
    def __init__(self, times: torch.Tensor, waveform: torch.Tensor):
        super().__init__(times, waveform)

    @staticmethod
    def get_amplitude(bandwidth: float, dz: float) -> float:
        return bandwidth / (dz * GAMMA)

    @property
    def flat_time(self) -> list[float]:
        flat_times = []
        for axis in range(3):
            flat_indices = torch.where(self.waveform[axis] == self.amplitude[axis])[0]
            if len(flat_indices) > 1:
                flat_duration = (len(flat_indices) - 1) * self.dt
            else:
                flat_duration = 0.0  # No flat section found

            flat_times.append(flat_duration)

        return flat_times

    @property
    def slew_rate(self) -> torch.Tensor:
        return torch.gradient(self.waveform, spacing=self.dt * 1e-6, dim=-1)[0]

    @property
    def zeroth_moment(self) -> torch.Tensor:
        return torch.cumsum(self.waveform * self.dt, dim=-1) * 1e-6

    @property
    def first_moment(self) -> torch.Tensor:
        return torch.cumsum(self.waveform * self.dt * self.times, dim=-1) * 1e-6 ** 2

    @property
    def waveform(self) -> torch.Tensor:
        return self.amplitude[:, torch.newaxis] * self._normalized_waveform

    def display(self) -> None:
        fig, ax = plt.subplots(2, 3, figsize=(12, 9), sharex=True)

        labels = [f'$G_{{x}}$', f'$G_{{y}}$', f'$G_{{z}}$']
        colors = ['k', 'b', 'r']

        max_amplitude = torch.max(torch.abs(self.waveform)) * 1e3 + 1
        max_slew_rate = torch.max(torch.abs(self.slew_rate)) + 10

        for axis in range(3):
            ax[0, axis].plot(self.times * 1e-3, self.waveform[axis] * 1e3, color=colors[axis])
            ax[0, axis].grid()
            ax[0, axis].set_ylim([-max_amplitude, max_amplitude])
            ax[0, axis].set_title(labels[axis], fontsize=16)

            ax[1, axis].plot(self.times * 1e-3, self.slew_rate[axis], color=colors[axis])
            ax[1, axis].grid()
            ax[1, axis].set_ylim([-max_slew_rate, max_slew_rate])

        fig.text(0.5, 0.04, 'Time (ms)', fontsize=16, ha='center', va='center')
        fig.text(0.06, 0.7, 'Amplitude (mT/m)', fontsize=16, va='center', rotation='vertical')
        fig.text(0.06, 0.3, 'Slew Rate (T/m/ms)', fontsize=16, va='center', rotation='vertical')
        plt.show()


def rect(duration: int, axis: int, *, n: Optional[int] = None, dt: Optional[float] = None) -> Gradient:
    """
    Generate a rectangular gradient lobe with a choice of duration. Sampling can be specified by n or dt.

    Args:
        duration (int): The duration of the gradient in microseconds.
        axis (int): The axis of the gradient. 0 for x, 1 for y, 2 for z.
        n (int | None): The number of samples to generate. Defaults to None.
        dt (float | None): The sampling interval in microseconds. Defaults to None.

    Returns:
        Gradient: A Gradient object representing the generated rectangular hard pulse.
    """
    times = determine_times(duration, n, dt)
    waveform = torch.zeros((3, len(times)))
    waveform[axis] = torch.ones(len(times))

    return Gradient(times, waveform)


def trapezium(duration: int, axis: int, ramp_time: int, *,
              n: Optional[int] = None, dt: Optional[float] = None) -> Gradient:
    """
    Generate a trapezium gradient lobe with a choice of duration and ramp time. Sampling can be specified by n or dt.

    Args:
        duration (int): The duration of the gradient in microseconds.
        axis (int): The axis of the gradient. 0 for x, 1 for y, 2 for z.
        ramp_time (int): The ramp time of the gradient in microseconds.
        n (int | None): The number of samples to generate. Defaults to None.
        dt (float | None): The sampling interval in microseconds. Defaults to None.

    Returns:
        Gradient: A Gradient object representing the generated trapezium hard pulse.
    """
    if duration <= 2 * ramp_time:
        raise ValueError("Duration must be greater than 2 * ramp_time.")

    times = determine_times(duration, n, dt)
    if dt is None:
        dt = times[1] - times[0]

    ramp_length = round(ramp_time / dt) + 1
    ramp = torch.linspace(0, 1, ramp_length)

    flat_length = round(duration / dt) - 2 * ramp_length
    flat = torch.ones(flat_length + 1)

    waveform = torch.zeros((3, len(times)))
    waveform[axis] = torch.concatenate((ramp, flat, torch.flip(ramp, dims=[0])))

    return Gradient(times, waveform)


def quarter_sine(duration: int, axis: int, ramp_time: int, *,
                 n: Optional[int] = None, dt: Optional[float] = None) -> Gradient:
    """
    Generate a quarter sine gradient lobe with a choice of duration and ramp time. Sampling can be specified by n or dt.

    Args:
        duration (int): The duration of the gradient in microseconds.
        axis (int): The axis of the gradient. 0 for x, 1 for y, 2 for z.
        ramp_time (int): The ramp time of the gradient in microseconds.
        n (int | None): The number of samples to generate. Defaults to None.
        dt (float | None): The sampling interval in microseconds. Defaults to None.

    Returns:
        Gradient: A Gradient object representing the generated quarter sine hard pulse.
    """
    if duration <= 2 * ramp_time:
        raise ValueError("Duration must be greater than 2 * ramp_time.")

    times = determine_times(duration, n, dt)
    if dt is None:
        dt = times[1] - times[0]

    ramp_length = round(ramp_time / dt) + 1
    ramp = torch.sin(torch.linspace(0, torch.pi / 2, ramp_length))

    flat_length = round(duration / dt) - 2 * ramp_length
    flat = torch.ones(flat_length + 1)

    waveform = torch.zeros((3, len(times)))
    waveform[axis] = torch.concatenate((ramp, flat, torch.flip(ramp, dims=[0])))

    return Gradient(times, waveform)
