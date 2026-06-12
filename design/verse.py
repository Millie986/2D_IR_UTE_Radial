import numba as nb
import numpy as np
import torch
from tqdm import tqdm

from constants import EPSILON
from sequence.core import determine_times
from sequence.gradient import Gradient
from sequence.rf import RFPulse


@nb.njit(nb.types.Tuple((nb.complex128[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.bool_))
             (nb.complex128[::1], nb.float64[::1], nb.float64[::1], nb.float64, nb.float64, nb.float64),
         cache=True, fastmath=True)
def verse_kernel(rf_waveform: np.ndarray, gradient: np.ndarray, dt: np.ndarray, b1_max: float, gradient_max: float,
                 slew_rate_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    ALPHA_MAX = 1.01
    TOLERANCE = 1e-8
    N = len(rf_waveform)

    # Backward adjustment
    rf_waveform[-1], gradient[-1] = 0, 0
    for k in range(N - 1, 0, -1):
        instantaneous_slew = (gradient[k] - gradient[k - 1]) / dt[k - 1]

        if abs(instantaneous_slew) > slew_rate_max:
            A = dt[k - 1] * slew_rate_max * np.sign(instantaneous_slew)
            B = -gradient[k]
            C = gradient[k - 1]
            delta = B ** 2 - 4 * A * C
            sqrt_delta = np.sqrt(delta)
            x1 = (-B - sqrt_delta) / (2 * A)
            x2 = (-B + sqrt_delta) / (2 * A)
            valid_alpha = [x for x in (x1, x2) if x > 0]
            if len(valid_alpha) == 0:
                alpha = ALPHA_MAX
            else:
                alpha = min(valid_alpha)

            alpha = min(alpha, ALPHA_MAX)

            gradient[k - 1] /= alpha
            rf_waveform[k - 1] /= alpha
            dt[k - 1] *= alpha

        if abs(rf_waveform[k - 1]) > b1_max:
            alpha = abs(rf_waveform[k - 1]) / b1_max
            gradient[k - 1] /= alpha
            rf_waveform[k - 1] /= alpha
            dt[k - 1] *= alpha

        if abs(gradient[k - 1]) > gradient_max:
            alpha = abs(gradient[k - 1]) / gradient_max
            gradient[k - 1] /= alpha
            rf_waveform[k - 1] /= alpha
            dt[k - 1] *= alpha

    # Forward adjustment
    gradient[0] = 0
    for k in range(N - 1):
        instantaneous_slew = (gradient[k + 1] - gradient[k]) / dt[k]

        if abs(instantaneous_slew) > slew_rate_max:
            alpha = gradient[k + 1] / (np.sign(instantaneous_slew) * slew_rate_max * dt[k] + gradient[k])
            if alpha == 0 and gradient[k + 1] == 0:  # Edge case
                alpha = EPSILON
            alpha = min(alpha, ALPHA_MAX)

            gradient[k + 1] /= alpha
            rf_waveform[k + 1] /= alpha
            if k < N - 2:
                dt[k + 1] *= alpha

    # Check for slew rate violations
    slew_rate = np.diff(gradient) / dt
    constraint_violated = abs(np.max(np.abs(slew_rate) - slew_rate_max)) >= TOLERANCE

    return rf_waveform, gradient, slew_rate, dt, constraint_violated


def verse(rf: RFPulse, grad: Gradient, axis: int,
          b1_max: float = 30e-6, gradient_max: float = 30e-3, slew_rate_max: float = 200) -> tuple[RFPulse, Gradient]:
    """
    Adjust the RF and gradient waveforms using the VERSE modulation approach.
    """
    rf_waveform = np.array(rf.waveform, dtype=np.complex128)
    gradient = np.array(grad.waveform[axis], dtype=np.float64)
    times = np.array(rf.times, dtype=np.float64) * 1e-6

    dt = np.diff(times)

    # Optimisation-based VERSE modulation
    # 1. Uniformly compress RF until max RF amplitude is reached
    rf_scale_factors = b1_max / np.max(np.abs(rf_waveform))
    rf_waveform *= rf_scale_factors
    gradient *= rf_scale_factors
    dt /= rf_scale_factors

    # 3. Compress RF and gradient together to ensure limits are not exceeded
    joint_scale_factor = np.minimum(b1_max / np.abs(rf_waveform + EPSILON),
                                    gradient_max / np.abs(gradient + EPSILON))
    rf_waveform *= joint_scale_factor
    gradient *= joint_scale_factor
    dt /= joint_scale_factor[:-1]

    # 4. Set endpoints of RF and gradient to zero
    rf_waveform[0], rf_waveform[-1] = 0, 0
    gradient[0], gradient[-1] = 0, 0

    # 5. Recursively adjust to avoid slew rate violations
    constraint_violated = True
    iteration = 0

    progress_bar = tqdm(bar_format='{desc} | {elapsed}')
    while constraint_violated:
        iteration += 1

        rf_waveform, gradient, slew_rate, dt, constraint_violated = verse_kernel(rf_waveform, gradient, dt,
                                                                         b1_max, gradient_max, slew_rate_max)
        progress_bar.set_description_str(f'VERSE Optimisation Iteration #{iteration}, '
                                         f'max |S| = {abs(slew_rate).max():.3f} T/m/s')
        progress_bar.update()

    t = np.concatenate((np.zeros(1), np.cumsum(dt, axis=0)))
    times = t * 1e6
    duration = int(rf.dt * np.ceil(times[-1] / rf.dt).item())

    resampled_times = determine_times(duration, n=None, dt=rf.dt)
    rf_waveform = torch.tensor(np.interp(resampled_times.numpy(), times, rf_waveform))

    gradient_waveform = torch.zeros((3, len(resampled_times)))
    gradient_waveform[axis] = torch.tensor(np.interp(resampled_times.numpy(), times, gradient))

    verse_rf = RFPulse(resampled_times, rf_waveform)
    verse_gradient = Gradient(resampled_times, gradient_waveform)

    return verse_rf, verse_gradient
