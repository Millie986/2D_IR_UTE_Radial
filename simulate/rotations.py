import math

import numba as nb  # type: ignore
import numpy as np
import torch


@nb.njit(nb.float64[:, :, ::1](nb.float64[:]), cache=True, fastmath=True)
def cpu_rotate_round_x(theta: np.ndarray) -> np.ndarray:
    """
    Generate a Nx3x3 rotation matrix around the x-axis.
    Args:
        theta (np.ndarray): The angles of rotation in radians.
    Returns:
        np.ndarray: The Nx3x3 rotation matrix.
    Example:
        >>> cpu_rotate_round_x(np.array(math.pi / 2))
        array([[ 1.000000e+00,  0.000000e+00,  0.000000e+00],
               [ 0.000000e+00,  6.123234e-17, -1.000000e+00],
               [ 0.000000e+00,  1.000000e+00,  6.123234e-17]])
    """
    c = np.cos(theta)
    s = np.sin(theta)

    matrix = np.zeros((len(theta), 3, 3), dtype=np.float64)

    matrix[:, 0, 0] = 1
    matrix[:, 0, 1] = 0
    matrix[:, 0, 2] = 0
    matrix[:, 1, 0] = 0
    matrix[:, 1, 1] = c
    matrix[:, 1, 2] = -s
    matrix[:, 2, 0] = 0
    matrix[:, 2, 1] = s
    matrix[:, 2, 2] = c

    return matrix


def gpu_rotate_round_x(theta: torch.Tensor) -> torch.Tensor:
    """
    Generate a Nx3x3 rotation matrix around the x-axis.
    Args:
        theta (torch.Tensor): The angles of rotation in radians.
    Returns:
        torch.Tensor: The 3x3 rotation matrix.
    Example:
        >>> gpu_rotate_round_x(torch.tensor(math.pi / 2))
        array([[ 1.000000e+00,  0.000000e+00,  0.000000e+00],
               [ 0.000000e+00,  6.123234e-17, -1.000000e+00],
               [ 0.000000e+00,  1.000000e+00,  6.123234e-17]])
    """
    with torch.no_grad():
        c = torch.cos(theta)
        s = torch.sin(theta)

        matrix = torch.zeros((len(theta), 3, 3), dtype=torch.float64, device='cuda')

        matrix[:, 0, 0] = 1
        matrix[:, 0, 1] = 0
        matrix[:, 0, 2] = 0
        matrix[:, 1, 0] = 0
        matrix[:, 1, 1] = c
        matrix[:, 1, 2] = -s
        matrix[:, 2, 0] = 0
        matrix[:, 2, 1] = s
        matrix[:, 2, 2] = c

        return matrix


@nb.njit(nb.float64[:, :, ::1](nb.float64[:]), cache=True, fastmath=True)
def cpu_rotate_round_y(theta: np.ndarray) -> np.ndarray:
    """
    Generate a Nx3x3 rotation matrix around the y-axis.
    Args:
        theta (np.ndarray): The angles of rotation in radians.
    Returns:
        np.ndarray: The 3x3 rotation matrix.
    Example:
        >>> cpu_rotate_round_y(np.array(math.pi / 2))
        array([[ 6.123234e-17,  0.000000e+00,  1.000000e+00],
               [ 0.000000e+00,  1.000000e+00,  0.000000e+00],
               [-1.000000e+00,  0.000000e+00,  6.123234e-17]])
    """
    c = np.cos(theta)
    s = np.sin(theta)

    matrix = np.zeros((len(theta), 3, 3), dtype=np.float64)

    matrix[:, 0, 0] = c
    matrix[:, 0, 1] = 0
    matrix[:, 0, 2] = s
    matrix[:, 1, 0] = 0
    matrix[:, 1, 1] = 1
    matrix[:, 1, 2] = 0
    matrix[:, 2, 0] = -s
    matrix[:, 2, 1] = 0
    matrix[:, 2, 2] = c

    return matrix


def gpu_rotate_round_y(theta: torch.Tensor) -> torch.Tensor:
    """
    Generate a Nx3x3 rotation matrix around the y-axis.
    Args:
        theta (torch.Tensor): The angles of rotation in radians.
    Returns:
        torch.Tensor: The 3x3 rotation matrix.
    Example:
        >>> gpu_rotate_round_y(torch.tensor(math.pi / 2))
        array([[ 6.123234e-17,  0.000000e+00,  1.000000e+00],
               [ 0.000000e+00,  1.000000e+00,  0.000000e+00],
               [-1.000000e+00,  0.000000e+00,  6.123234e-17]])
    """
    with torch.no_grad():
        c = torch.cos(theta)
        s = torch.sin(theta)

        matrix = torch.zeros((len(theta), 3, 3), dtype=torch.float64, device='cuda')

        matrix[:, 0, 0] = c
        matrix[:, 0, 1] = 0
        matrix[:, 0, 2] = s
        matrix[:, 1, 0] = 0
        matrix[:, 1, 1] = 1
        matrix[:, 1, 2] = 0
        matrix[:, 2, 0] = -s
        matrix[:, 2, 1] = 0
        matrix[:, 2, 2] = c

        return matrix


@nb.njit(nb.float64[:, :, ::1](nb.float64[:]), cache=True, fastmath=True)
def cpu_rotate_round_z(theta: np.ndarray) -> np.ndarray:
    """
    Generate a Nx3x3 rotation matrix around the z-axis.
    Args:
        theta (np.ndarray): The angles of rotation in radians.
    Returns:
        np.ndarray: The 3x3 rotation matrix.
    Example:
        >>> cpu_rotate_round_z(np.array(math.pi / 2))
        array([[ 6.123234e-17, -1.000000e+00,  0.000000e+00],
               [ 1.000000e+00,  6.123234e-17,  0.000000e+00],
               [ 0.000000e+00,  0.000000e+00,  1.000000e+00]])
    """
    c = np.cos(theta)
    s = np.sin(theta)

    matrix = np.zeros((len(theta), 3, 3), dtype=np.float64)

    matrix[:, 0, 0] = c
    matrix[:, 0, 1] = -s
    matrix[:, 0, 2] = 0
    matrix[:, 1, 0] = s
    matrix[:, 1, 1] = c
    matrix[:, 1, 2] = 0
    matrix[:, 2, 0] = 0
    matrix[:, 2, 1] = 0
    matrix[:, 2, 2] = 1

    return matrix


def gpu_rotate_round_z(theta: torch.Tensor) -> torch.Tensor:
    """
    Generate a tensor of 3x3 rotation matrix around the z-axis.
    Args:
        theta (torch.Tensor): The angles of rotation in radians.
    Returns:
        torch.Tensor: The rotation matrices.
    Example:
        >>> gpu_rotate_round_z(torch.tensor(math.pi / 2))
        array([[ 6.123234e-17, -1.000000e+00,  0.000000e+00],
               [ 1.000000e+00,  6.123234e-17,  0.000000e+00],
               [ 0.000000e+00,  0.000000e+00,  1.000000e+00]])
    """
    with torch.no_grad():
        c = torch.cos(theta)
        s = torch.sin(theta)

        matrix = torch.zeros((*theta.shape, 3, 3), dtype=torch.float64, device='cuda')

        matrix[..., 0, 0] = c
        matrix[..., 0, 1] = -s
        matrix[..., 0, 2] = 0
        matrix[..., 1, 0] = s
        matrix[..., 1, 1] = c
        matrix[..., 1, 2] = 0
        matrix[..., 2, 0] = 0
        matrix[..., 2, 1] = 0
        matrix[..., 2, 2] = 1

        return matrix


@nb.njit(nb.float64[:, :, ::1](nb.float64[:], nb.float64[:]), cache=True, fastmath=True)
def cpu_rotate_transverse(phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Generate a 3x3 rotation matrix around an axis defined by y = x * tan(theta)

    Args:
        phi (np.ndarray): The angles of rotation around the x-axis in radians.
        theta (np.ndarray): The angles of rotation around the z-axis in radians.

    Returns:
        np.ndarray: The Nx3x3 rotation matrix.

    Example:
        >>> cpu_rotate_transverse(np.array(math.pi / 4), np.array(math.pi / 6))
        array([[ 0.9267767 ,  0.12682648,  0.35355339],
               [ 0.12682648,  0.78033009, -0.61237244],
               [-0.35355339,  0.61237244,  0.70710678]])
    """
    c_x, c_z = np.cos(phi), np.cos(theta)
    s_x, s_z = np.sin(phi), np.sin(theta)

    matrix = np.zeros((len(phi), 3, 3), dtype=np.float64)

    matrix[:, 0, 0] = c_z ** 2 + c_x * s_z ** 2
    matrix[:, 0, 1] = (1 - c_x) * c_z * s_z
    matrix[:, 0, 2] = -s_x * s_z
    matrix[:, 1, 0] = (1 - c_x) * c_z * s_z
    matrix[:, 1, 1] = c_x * c_z ** 2 + s_z ** 2
    matrix[:, 1, 2] = s_x * c_z
    matrix[:, 2, 0] = s_x * s_z
    matrix[:, 2, 1] = -s_x * c_z
    matrix[:, 2, 2] = c_x

    return matrix


def gpu_rotate_transverse(phi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Generate a set of 3x3 rotation matrices around an axis defined by y = x * tan(theta)

    Args:
        phi (float): The angle of rotation around the x-axis in radians.
        theta (float): The angle of rotation around the z-axis in radians.

    Returns:
        torch.Tensor: The 3x3 rotation matrices.

    Example:
        >>> gpu_rotate_transverse(torch.Tensor(math.pi / 4), torch.Tensor(math.pi / 6))
        array([[ 0.9267767 ,  0.12682648,  0.35355339],
               [ 0.12682648,  0.78033009, -0.61237244],
               [-0.35355339,  0.61237244,  0.70710678]])
    """
    with torch.no_grad():
        if phi.shape != theta.shape:
            raise ValueError("phi and theta must have the same shape.")

        c_x, c_z = torch.cos(phi), torch.cos(theta)
        s_x, s_z = torch.sin(phi), torch.sin(theta)

        matrix = torch.zeros((*phi.shape, 3, 3), dtype=torch.float64, device='cuda')

        matrix[..., 0, 0] = c_z ** 2 + c_x * s_z ** 2
        matrix[..., 0, 1] = (1 - c_x) * c_z * s_z
        matrix[..., 0, 2] = -s_x * s_z
        matrix[..., 1, 0] = (1 - c_x) * c_z * s_z
        matrix[..., 1, 1] = c_x * c_z ** 2 + s_z ** 2
        matrix[..., 1, 2] = s_x * c_z
        matrix[..., 2, 0] = s_x * s_z
        matrix[..., 2, 1] = -s_x * c_z
        matrix[..., 2, 2] = c_x

        return matrix


@nb.njit(nb.types.Tuple((nb.float64[:, :, ::1], nb.float64[:, ::1]))(nb.float64, nb.float64,
                                                                     nb.float64[:], nb.float64), cache=True,
         fastmath=True)
def cpu_free_precession(t1: float, t2: float, df: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the propagation matrices for free precession.

    Args:
        t1 (float): The T1 relaxation time in seconds.
        t2 (float): The T2 relaxation time in seconds.
        df (np.ndarray): The off-resonance frequencies in Hz.
        dt (float): The time step in seconds.

    Returns:
        tuple[np.ndarray, np.ndarray]: The decay and recovery matrices.

    Example:
        >>> cpu_free_precession(600e-3,100e-3,np.array(0.0),1e-3)
        (array([[0.99004983, 0.        , 0.        ],
               [0.        , 0.99004983, 0.        ],
               [0.        , 0.        , 0.99833472]]), array([0.        , 0.        , 0.00166528]))
    """
    c, s = np.cos(2 * math.pi * df * dt), np.sin(2 * math.pi * df * dt)

    e1 = math.exp(-dt / t1)
    e2 = math.exp(-dt / t2)

    decay = np.zeros((len(df), 3, 3), dtype=np.float64)
    decay[:, 0, 0] = e2 * c
    decay[:, 0, 1] = e2 * -s
    decay[:, 0, 2] = 0
    decay[:, 1, 0] = e2 * s
    decay[:, 1, 1] = e2 * c
    decay[:, 1, 2] = 0
    decay[:, 2, 0] = 0
    decay[:, 2, 1] = 0
    decay[:, 2, 2] = e1

    recovery = np.zeros((len(df), 3), dtype=np.float64)
    recovery[:, 0] = 0
    recovery[:, 1] = 0
    recovery[:, 2] = 1 - e1

    return decay, recovery


def gpu_free_precession(t1: float, t2: float, df: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate the propagation matrices for free precession.

    Args:
        t1 (float): The T1 relaxation time in seconds.
        t2 (float): The T2 relaxation time in seconds.
        df (torch.Tensor): The off-resonance frequency in Hz.
        dt (float): The time step in seconds.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The decay and recovery matrices.

    Example:
        >>> gpu_free_precession(600e-3,100e-3,torch.tensor(0.0),1e-3)
        (array([[0.99004983, 0.        , 0.        ],
               [0.        , 0.99004983, 0.        ],
               [0.        , 0.        , 0.99833472]]), array([0.        , 0.        , 0.00166528]))
    """
    with torch.no_grad():
        c, s = torch.cos(2 * torch.pi * df * dt), torch.sin(2 * torch.pi * df * dt)

        e1 = math.exp(-dt / t1)
        e2 = math.exp(-dt / t2)

        decay = torch.zeros((*df.shape, 3, 3), dtype=torch.float64, device='cuda')
        decay[..., 0, 0] = e2 * c
        decay[..., 0, 1] = e2 * -s
        decay[..., 0, 2] = 0
        decay[..., 1, 0] = e2 * s
        decay[..., 1, 1] = e2 * c
        decay[..., 1, 2] = 0
        decay[..., 2, 0] = 0
        decay[..., 2, 1] = 0
        decay[..., 2, 2] = e1

        recovery = torch.zeros((*df.shape, 3, 1), dtype=torch.float64, device='cuda')
        recovery[..., 0, 0] = 0
        recovery[..., 1, 0] = 0
        recovery[..., 2, 0] = 1 - e1

        return decay, recovery
