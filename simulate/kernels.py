import numba as nb  # type: ignore
import numpy as np
import torch

import simulate.rotations
from constants import GAMMA_RAD


@nb.njit(nb.float64[:, :, ::1](nb.float64, nb.float64, nb.float64, nb.float64[::1], nb.float64[::1]),
         cache=True, fastmath=True, parallel=True)
def cpu_relaxation(t1: float, t2: float, dt: float, df: np.ndarray, initial_magnetisation: np.ndarray) -> np.ndarray:
    """
    Simulate relaxation for an array of isochromats for a time period determined by
    the maximum of 5*T1 and 5*T2 periods. This is done using the CPU in a multithreaded manner.

    Args:
        t1 (float): The T1 relaxation time in seconds.
        t2 (float): The T2 relaxation time in seconds.
        dt (float): The time step in seconds.
        df (np.ndarray): The off-resonance frequencies in Hz.
        initial_magnetisation (np.ndarray): The initial magnetisation.

    Returns:
        np.ndarray: The magnetisation evolution. Shape: (n_isochromats, n_steps + 1, 3)
    """
    # Initialize arrays - batch dimension first to maintain memory contiguity
    n_isochromats = len(df)
    n_steps = int(max(5 * t1 / dt, 5 * t2 / dt))
    magnetisation = np.zeros((n_isochromats, n_steps + 1, 3), dtype=np.float64)
    magnetisation[:, 0] = initial_magnetisation

    # Prepare free precession matrices
    A, B = simulate.rotations.cpu_free_precession(t1, t2, df, dt)

    # Run simulation
    for i in nb.prange(n_isochromats):
        for j in range(n_steps):
            magnetisation[i, j + 1] = A[i] @ magnetisation[i, j] + B[i]

    return magnetisation


@nb.njit(nb.float64[:, :, ::1](nb.float64, nb.float64, nb.float64, nb.float64[::1],
                               nb.complex128[::1], nb.float64[::1]),
         cache=True, fastmath=True, parallel=True)
def cpu_non_selective(t1: float, t2: float, dt: float, df: np.array,
                      rf: np.array, initial_magnetisation: np.array) -> np.array:
    """
    Simulate a non-selective RF excitation for an array of isochromats for the length of the RF pulse. This is done
    using the CPU in a multithreaded manner.

    Args:
        t1 (float): The T1 relaxation time in seconds.
        t2 (float): The T2 relaxation time in seconds.
        dt (float): The time step in seconds.
        df (np.ndarray): The off-resonance frequencies in Hz.
        rf (np.ndarray): The complex RF waveform.
        initial_magnetisation (np.ndarray): The initial magnetisation.

    Returns:
        np.ndarray: The magnetisation evolution. Shape: (n_isochromats, n_steps + 1, 3)
    """
    # Initialize arrays - batch dimension first to maintain memory contiguity
    n_isochromats = len(df)
    n_steps = len(rf)
    magnetisation = np.zeros((n_isochromats, n_steps + 1, 3), dtype=np.float64)
    magnetisation[:, 0] = initial_magnetisation

    # Prepare RF rotation matrices
    rf_mag, rf_phase = GAMMA_RAD * np.abs(rf) * dt, np.angle(rf)
    rf_rotations = simulate.rotations.cpu_rotate_transverse(rf_mag, rf_phase)
    active_rf = np.zeros(n_steps, dtype=np.bool)
    active_rf[np.where(np.abs(rf) > 0)[0]] = True

    # Prepare free precession matrices
    A, B = simulate.rotations.cpu_free_precession(t1, t2, df, dt)

    # Run simulation
    for i in nb.prange(n_isochromats):
        for j in range(n_steps):
            # Apply free precession
            magnetisation[i, j + 1] = A[i] @ magnetisation[i, j] + B[i]

            # Apply RF
            if active_rf[j]:
                magnetisation[i, j + 1] = rf_rotations[j] @ magnetisation[i, j + 1]

    return magnetisation


@nb.njit(nb.float64[:, :, ::1](nb.float64, nb.float64, nb.float64, nb.float64[:, ::1],
                               nb.complex128[::1], nb.float64[:, ::1], nb.float64[::1]),
         cache=True, fastmath=True, parallel=True)
def cpu_spatial_selective(t1: float, t2: float, dt: float, pos: np.array,
                          rf: np.array, grad: np.array, initial_magnetisation: np.array) -> np.array:
    """
    Simulate a spatially-selective RF excitation for an array of isochromats for the length of the RF pulse. This is
    done using the CPU in a multithreaded manner.

    Args:
        t1 (float): The T1 relaxation time in seconds.
        t2 (float): The T2 relaxation time in seconds.
        dt (float): The time step in seconds.
        pos (np.ndarray): The positions of the isochromats in m. Shape: (3, n_isochromats)
        rf (np.ndarray): The complex RF waveform.
        grad (np.ndarray): The gradient waveform. Shape: (3, n_steps)
        initial_magnetisation (np.ndarray): The initial magnetisation.

    Returns:
        np.ndarray: The magnetisation evolution. Shape: (n_isochromats, n_steps + 1, 3)
    """
    # Initialize arrays - batch dimension first to maintain memory contiguity
    n_isochromats = pos.shape[1]
    n_steps = len(rf)
    magnetisation = np.zeros((n_isochromats, n_steps + 1, 3), dtype=np.float64)
    magnetisation[:, 0] = initial_magnetisation

    # Prepare RF rotation matrices
    rf_mag, rf_phase = GAMMA_RAD * np.abs(rf) * dt, np.angle(rf)
    rf_rotations = simulate.rotations.cpu_rotate_transverse(rf_mag, rf_phase)
    active_rf = np.zeros(n_steps, dtype=np.bool)
    active_rf[np.where(np.abs(rf) > 0)[0]] = True

    # Prepare gradient rotation matrices
    active_gradient_axes = [np.where(np.abs(grad[:, j]) > 0)[0] for j in range(n_steps)]
    grad_rot = np.zeros((3, n_steps, n_isochromats, 3, 3), dtype=np.float64)

    for j in nb.prange(n_steps):
        for axis in active_gradient_axes[j]:
            grad_rot[axis, j] = simulate.rotations.cpu_rotate_round_z(
                (GAMMA_RAD * grad[axis, j] * dt * pos[axis, :]))

    # Prepare free precession matrix
    A, B = simulate.rotations.cpu_free_precession(t1, t2, np.zeros(1), dt)

    # Run simulation
    for i in nb.prange(n_isochromats):
        for j in range(n_steps):
            # Apply gradients
            for axis in active_gradient_axes[j]:
                magnetisation[i, j] = grad_rot[axis, j, i] @ magnetisation[i, j]

            # Apply free precession
            magnetisation[i, j + 1] = (A[0] @ magnetisation[i, j] + B[0])

            # Apply RF
            if active_rf[j]:
                magnetisation[i, j + 1] = rf_rotations[j] @ magnetisation[i, j + 1]

    return magnetisation


@nb.njit(nb.float64[:, :, ::1](nb.float64, nb.float64, nb.float64, nb.float64[::1], nb.float64[:, ::1],
                               nb.complex128[::1], nb.float64[:, ::1], nb.float64[::1]),
         cache=True, fastmath=True, parallel=True)
def cpu_spatial_spectral(t1: float, t2: float, dt: float, df: np.array, pos: np.array,
                         rf: np.array, grad: np.array, initial_magnetisation: np.array) -> np.array:
    """
    Simulate spatially-selective RF excitation for an array of isochromats with both position and off-resonance for the
    length of the RF pulse. This is done using the CPU in a multithreaded manner.

    Args:
        t1 (float): The T1 relaxation time in seconds.
        t2 (float): The T2 relaxation time in seconds.
        dt (float): The time step in seconds.
        df (np.ndarray): The off-resonance frequencies in Hz.
        pos (np.ndarray): The positions of the isochromats in m. Shape: (3, n_isochromats)
        rf (np.ndarray): The complex RF waveform.
        grad (np.ndarray): The gradient waveform. Shape: (3, n_steps)
        initial_magnetisation (np.ndarray): The initial magnetisation.

    Returns:
        np.ndarray: The magnetisation evolution. Shape: (n_isochromats, n_steps + 1, 3)
    """
    # Initialize arrays - batch dimension first to maintain memory contiguity
    n_isochromats = len(df)
    n_steps = len(rf)
    magnetisation = np.zeros((n_isochromats, n_steps + 1, 3), dtype=np.float64)
    magnetisation[:, 0] = initial_magnetisation

    # Prepare RF rotation matrices
    rf_mag, rf_phase = GAMMA_RAD * np.abs(rf) * dt, np.angle(rf)
    rf_rotations = simulate.rotations.cpu_rotate_transverse(rf_mag, rf_phase)
    active_rf = np.zeros(n_steps, dtype=np.bool)
    active_rf[np.where(np.abs(rf) > 0)[0]] = True

    # Prepare gradient rotation matrices
    active_gradient_axes = [np.where(np.abs(grad[:, j]) > 0)[0] for j in range(n_steps)]
    grad_rot = np.zeros((3, n_steps, n_isochromats, 3, 3), dtype=np.float64)

    for j in nb.prange(n_steps):
        for axis in active_gradient_axes[j]:
            grad_rot[axis, j] = simulate.rotations.cpu_rotate_round_z(
                (GAMMA_RAD * grad[axis, j] * dt * pos[axis, :]))

    # Prepare free precession matrix
    A, B = simulate.rotations.cpu_free_precession(t1, t2, df, dt)

    # Run simulation
    for i in nb.prange(n_isochromats):
        for j in range(n_steps):
            # Apply gradients
            for axis in active_gradient_axes[j]:
                magnetisation[i, j] = grad_rot[axis, j, i] @ magnetisation[i, j]

            # Apply free precession
            magnetisation[i, j + 1] = (A[i] @ magnetisation[i, j] + B[i])

            # Apply RF
            if active_rf[j]:
                magnetisation[i, j + 1] = rf_rotations[j] @ magnetisation[i, j + 1]

    return magnetisation


def gpu_relaxation(t1: float, t2: float, dt: float, df: torch.Tensor,
                   initial_magnetisation: torch.Tensor) -> torch.Tensor:
    """
    Simulate relaxation for an array of isochromats for a time period determined by
    the maximum of 5*T1 and 5*T2 periods. This is done using the GPU.

    Args:
        t1 (float): The T1 relaxation time in seconds.
        t2 (float): The T2 relaxation time in seconds.
        dt (float): The time step in seconds.
        df (np.ndarray): The off-resonance frequencies in Hz.
        initial_magnetisation (np.ndarray): The initial magnetisation.

    Returns:
        np.ndarray: The magnetisation evolution. Shape: (n_isochromats, n_steps + 1, 3)
    """
    # Initialize arrays - batch dimension first to maintain memory contiguity
    with torch.no_grad():
        n_isochromats = len(df)
        n_steps = int(max(5 * t1 / dt, 5 * t2 / dt))
        magnetisation = torch.zeros((n_steps + 1, n_isochromats, 3, 1), dtype=torch.float64, device='cuda')
        magnetisation[0] = initial_magnetisation[:, torch.newaxis]

        # Prepare free precession matrices
        A, B = simulate.rotations.gpu_free_precession(t1, t2, df, dt)

        # Run simulation
        for i in range(n_steps):
            magnetisation[i + 1] = A @ magnetisation[i] + B

        return torch.permute(magnetisation, (1, 0, 2, 3))


def gpu_non_selective(t1: float, t2: float, dt: float, df: torch.Tensor,
                      rf: torch.Tensor, initial_magnetisation: torch.Tensor) -> torch.Tensor:
    """
    Simulate a non-selective RF excitation for an array of isochromats for the length of the RF pulse. This is done
    using the GPU.

    Args:
        t1 (float): The T1 relaxation time in seconds.
        t2 (float): The T2 relaxation time in seconds.
        dt (float): The time step in seconds.
        df (np.ndarray): The off-resonance frequencies in Hz.
        rf (np.ndarray): The complex RF waveform.
        initial_magnetisation (np.ndarray): The initial magnetisation.

    Returns:
        np.ndarray: The magnetisation evolution. Shape: (n_isochromats, n_steps + 1, 3)
    """
    # Initialize arrays - batch dimension first to maintain memory contiguity
    with torch.no_grad():
        n_isochromats = len(df)
        n_steps = len(rf)
        magnetisation = torch.zeros((n_steps + 1, n_isochromats, 3, 1), dtype=torch.float64, device='cuda')
        magnetisation[0] = initial_magnetisation[:, torch.newaxis]

        # Prepare RF rotation matrices
        rf_mag, rf_phase = GAMMA_RAD * torch.abs(rf) * dt, torch.angle(rf)
        rf_rotations = simulate.rotations.gpu_rotate_transverse(rf_mag, rf_phase)

        # Prepare free precession matrices
        A, B = simulate.rotations.gpu_free_precession(t1, t2, df, dt)

        # Run simulation
        for i in range(n_steps):
            magnetisation[i + 1] = rf_rotations[i] @ (A @ magnetisation[i] + B)

        return torch.permute(magnetisation, (1, 0, 2, 3))[..., 0]


def gpu_spatial_selective(t1: float, t2: float, dt: float, pos: torch.Tensor,
                          rf: torch.Tensor, grad: torch.Tensor, initial_magnetisation: torch.Tensor) -> torch.Tensor:
    """
    Simulate a selective RF excitation for an array of isochromats for the length of the RF pulse. This is done
    using the GPU.

    Args:
        t1 (float): The T1 relaxation time in seconds.
        t2 (float): The T2 relaxation time in seconds.
        dt (float): The time step in seconds.
        pos (np.ndarray): The positions of the isochromats in m. Shape: (3, n_isochromats)
        rf (np.ndarray): The complex RF waveform.
        grad (np.ndarray): The gradient waveform. Shape: (3, n_steps)
        initial_magnetisation (np.ndarray): The initial magnetisation.

    Returns:
        np.ndarray: The magnetisation evolution. Shape: (n_isochromats, n_steps + 1, 3)
    """
    # Initialize arrays - batch dimension first to maintain memory contiguity
    with torch.no_grad():
        n_isochromats = pos.shape[1]
        n_steps = len(rf)
        magnetisation = torch.zeros((n_steps + 1, n_isochromats, 3, 1), dtype=torch.float64, device='cuda')
        magnetisation[0] = initial_magnetisation[:, torch.newaxis]

        # Prepare RF rotation matrices
        rf_mag, rf_phase = GAMMA_RAD * torch.abs(rf) * dt, torch.angle(rf)
        rf_rotations = simulate.rotations.gpu_rotate_transverse(rf_mag, rf_phase)

        # Prepare gradient rotation matrices
        grad_rot = pos[:, :, torch.newaxis] * GAMMA_RAD * grad[:, torch.newaxis, :] * dt
        grad_rot = torch.permute(grad_rot, (0, 2, 1))
        grad_rotations = simulate.rotations.gpu_rotate_round_z(grad_rot)

        # Prepare free precession matrix
        A, B = simulate.rotations.gpu_free_precession(t1, t2, torch.zeros(1), dt)

        active_gradient_axes = torch.where(torch.any(torch.abs(grad) > 0, dim=1))[0]

        # Run simulation
        for i in range(n_steps):
            for axis in active_gradient_axes:
                magnetisation[i] = grad_rotations[axis, i] @ magnetisation[i]

            magnetisation[i + 1] = rf_rotations[i] @ (A @ magnetisation[i] + B)

        return torch.permute(magnetisation, (1, 0, 2, 3))


def gpu_spatial_spectral(t1: float, t2: float, dt: float, df: torch.Tensor, pos: torch.Tensor,
                         rf: torch.Tensor, grad: torch.Tensor, initial_magnetisation: torch.Tensor) -> torch.Tensor:
    """
    Simulate spatially-selective RF excitation for an array of isochromats with both position and off-resonance for the
    length of the RF pulse. This is done using the GPU.

    Args:
        t1 (float): The T1 relaxation time in seconds.
        t2 (float): The T2 relaxation time in seconds.
        dt (float): The time step in seconds.
        df (np.ndarray): The off-resonance frequencies in Hz.
        pos (np.ndarray): The positions of the isochromats in m. Shape: (3, n_isochromats)
        rf (np.ndarray): The complex RF waveform.
        grad (np.ndarray): The gradient waveform. Shape: (3, n_steps)
        initial_magnetisation (np.ndarray): The initial magnetisation.

    Returns:
        np.ndarray: The magnetisation evolution. Shape: (n_isochromats, n_steps + 1, 3)
    """
    # Initialize arrays - batch dimension first to maintain memory contiguity
    with torch.no_grad():
        n_isochromats = len(df)
        n_steps = len(rf)
        magnetisation = torch.zeros((n_steps + 1, n_isochromats, 3, 1), dtype=torch.float64, device='cuda')
        magnetisation[0] = initial_magnetisation[:, torch.newaxis]

        # Prepare RF rotation matrices
        rf_mag, rf_phase = GAMMA_RAD * torch.abs(rf) * dt, torch.angle(rf)
        rf_rotations = simulate.rotations.gpu_rotate_transverse(rf_mag, rf_phase)

        # Prepare gradient rotation matrices
        grad_rot = pos[:, :, torch.newaxis] * GAMMA_RAD * grad[:, torch.newaxis, :] * dt
        grad_rot = torch.permute(grad_rot, (0, 2, 1))
        grad_rotations = simulate.rotations.gpu_rotate_round_z(grad_rot)

        # Prepare free precession matrices
        A, B = simulate.rotations.gpu_free_precession(t1, t2, df, dt)

        active_gradient_axes = torch.where(torch.any(torch.abs(grad) > 0, dim=1))
        # Run simulation
        for i in range(n_steps):
            for axis in active_gradient_axes:
                magnetisation[i] = grad_rotations[axis, i] @ magnetisation[i]

            magnetisation[i + 1] = rf_rotations[i] @ (A @ magnetisation[i] + B)

        return torch.permute(magnetisation, (1, 0, 2, 3))
