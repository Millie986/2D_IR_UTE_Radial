from typing import Optional

import numpy as np
import torch

from constants import EPSILON

class SequenceObject:
    def __init__(self, times: torch.Tensor, waveform: torch.Tensor):
        if len(times) != waveform.shape[-1]:
            raise ValueError("times and waveform must have the same length.")

        # ORIGINAL
        self._times = times
        self._waveform = waveform

        # CURRENT
        if self._waveform.ndim == 1:
            self._normalized_waveform = self._waveform / torch.abs(self._waveform).max()
            self.amplitude = torch.abs(self._waveform).max()
        else:
            amplitudes = torch.max(torch.abs(self._waveform), dim=-1).values
            self._normalized_waveform = self._waveform / (amplitudes[:, torch.newaxis] + EPSILON)
            self.amplitude = amplitudes

        self._current_times = self._times.detach().clone()

    @property
    def n(self) -> int:
        return len(self.times)

    @property
    def dt(self) -> float:
        deltas = torch.unique(self.times[1:] - self.times[:-1])

        if len(deltas) > 1:
            raise ValueError("dt is not constant.")

        return deltas.item()

    @property
    def duration(self) -> int:
        return (self._times.max() - self._times.min()).item()

    @property
    def times(self) -> torch.Tensor:
        return self._current_times

    @property
    def waveform(self) -> torch.Tensor:
        return self.amplitude * self._normalized_waveform

    @property
    def magnitude(self) -> torch.Tensor:
        return torch.abs(self.waveform)

    @property
    def phase(self) -> torch.Tensor:
        return torch.angle(self.waveform)

    def append(self, other: "SequenceObject", gap: int = 0) -> 'SequenceObject':
        if type(self) is not type(other):
            raise ValueError("Can only append SequenceObjects of the same type.")

        if self.dt != other.dt:
            raise ValueError("Can only append SequenceObjects with the same dt.")

        new_duration = self.duration + gap + other.duration

        gap_length = len(determine_times(gap, None, self.dt)) - 1
        new_times = determine_times(new_duration, None, self.dt)
        new_waveform = torch.concatenate((self.waveform, torch.zeros(gap_length), other.waveform[1:]))

        return type(self)(new_times, new_waveform)

    def zero_pad(self, duration: tuple[int, int]) -> 'SequenceObject':
        prior_times = determine_times(duration[0], None, self.dt)
        post_times = prior_times[-1] + self.duration + determine_times(duration[1], None, self.dt)

        prior_waveform = torch.zeros(len(prior_times))
        post_waveform = torch.zeros(len(post_times))

        new_times = torch.concatenate((prior_times, duration[0] + self.times[1:], post_times[1:]))
        new_waveform = torch.concatenate((prior_waveform, self.waveform[1:], post_waveform[1:]))

        return type(self)(new_times, new_waveform)

    def resample(self, *, n: Optional[int] = None, dt: Optional[float] = None) -> 'SequenceObject':
        """
        Resample the sequence object with a new interval in microseconds.
        """
        new_times = determine_times(self.duration, n, dt)
        new_waveform = torch.from_numpy(np.interp(new_times.numpy(), self.times.numpy(), self.waveform.numpy()))

        return type(self)(new_times, new_waveform)

    def __add__(self, other: "SequenceObject") -> "SequenceObject":
        if type(self) is not type(other):
            raise ValueError("Can only append SequenceObjects of the same type.")

        if self.dt != other.dt:
            raise ValueError("Can only append SequenceObjects with the same dt.")

        new_waveform = self.waveform + other.waveform

        return type(self)(self.times, new_waveform)

    def __sub__(self, other: "SequenceObject") -> "SequenceObject":
        if type(self) is not type(other):
            raise ValueError("Can only append SequenceObjects of the same type.")

        if self.dt != other.dt:
            raise ValueError("Can only append SequenceObjects with the same dt.")

        new_waveform = self.waveform - other.waveform

        return type(self)(self.times, new_waveform)

    def __len__(self) -> int:
        return len(self.times)

    def __copy__(self) -> "SequenceObject":
        return type(self)(self.times, self.waveform)


def determine_times(duration: int, n: Optional[int], dt: Optional[float]) -> torch.Tensor:
    if None not in (n, dt):
        raise ValueError("Must specify either N or dt.")

    if n is not None:
        # Define times by subdividing 0 -> T into N points
        return torch.linspace(0, duration, n)

    if dt is not None:
        if duration % dt != 0:
            raise ValueError("dt must be a multiple of duration.")
        # Define times by subdividing 0 -> T into intervals of dt
        total_steps = int(duration / dt)

        return torch.linspace(0, total_steps, total_steps + 1) * dt

    raise ValueError("Must specify either N or dt.")
