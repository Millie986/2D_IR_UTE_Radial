from __future__ import annotations

import enum
import time
import warnings
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from sequence.gradient import Gradient
    from sequence.rf import RFPulse

import simulate.kernels
from sequence.core import determine_times


class SimulationStyle(enum.Enum):
    RELAXATION = 0
    NON_SELECTIVE = 1
    SPATIAL_SELECTIVE = 2
    SPATIAL_SPECTRAL = 3


kernels: dict[SimulationStyle, dict[bool, callable]] = {
    SimulationStyle.RELAXATION:
        {
            True: simulate.kernels.gpu_relaxation,
            False: simulate.kernels.cpu_relaxation
        },
    SimulationStyle.NON_SELECTIVE:
        {
            True: simulate.kernels.gpu_non_selective,
            False: simulate.kernels.cpu_non_selective,
        },
    SimulationStyle.SPATIAL_SELECTIVE:
        {
            True: simulate.kernels.gpu_spatial_selective,
            False: simulate.kernels.cpu_spatial_selective
        },
    SimulationStyle.SPATIAL_SPECTRAL:
        {
            True: simulate.kernels.gpu_spatial_spectral,
            False: simulate.kernels.cpu_spatial_spectral
        },
}

arguments: dict[SimulationStyle, list[str]] = {
    SimulationStyle.RELAXATION:
        ['df'],
    SimulationStyle.NON_SELECTIVE:
        ['df', '_rf'],
    SimulationStyle.SPATIAL_SELECTIVE:
        ['pos', '_rf', '_grad'],
    SimulationStyle.SPATIAL_SPECTRAL:
        ['df', 'pos', '_rf', '_grad'],
}


class BlochDispatcher:
    @classmethod
    def from_sequence(cls, sequence) -> BlochDispatcher:
        pass

    def __init__(self):
        # Isochromats
        self.df = None
        self.pos = None

        # Sequence objects
        self._rf: Optional[RFPulse] = None
        self._grad: Optional[Gradient] = None

    def set_rf(self, rf: RFPulse) -> None:
        self._rf = rf

    def set_grad(self, grad: Gradient) -> None:
        self._grad = grad

    def set_df(self, df: torch.Tensor) -> None:
        self.df = df

    def set_pos(self, pos: torch.Tensor) -> None:
        if pos.shape[0] != 3:
            raise ValueError("pos must be a 3xN Tensor.")

        self.pos = pos

    def determine_style(self) -> SimulationStyle:
        gradients_active = self._grad is not None
        rf_active = self._rf is not None

        isochromats_have_position = self.pos is not None
        isochromats_have_off_resonance = self.df is not None

        if not gradients_active and not rf_active:
            return SimulationStyle.RELAXATION
        elif not gradients_active and rf_active and isochromats_have_off_resonance:
            return SimulationStyle.NON_SELECTIVE
        elif gradients_active and rf_active and not isochromats_have_off_resonance and isochromats_have_position:
            return SimulationStyle.SPATIAL_SELECTIVE
        elif gradients_active and rf_active and isochromats_have_off_resonance and isochromats_have_position:
            return SimulationStyle.SPATIAL_SPECTRAL
        else:
            raise ValueError("Invalid combination of gradients, rf, and isochromats!")

    def get_simulation_steps(self, style: SimulationStyle) -> float:
        if style == SimulationStyle.RELAXATION:
            return 0
        else:
            sequence_objects = [self._rf, self._grad]
            object_lengths = [len(obj.waveform) for obj in sequence_objects if obj is not None]

            if len(set(object_lengths)) != 1:
                raise ValueError("All sequence objects must have the same length.")

            max_length = max([len(obj.waveform) for obj in sequence_objects if obj is not None])

        return max_length

    def prepare_kernel_arguments(self, style: SimulationStyle, gpu_available: bool) -> list:
        kernel_args = []

        for arg in arguments[style]:
            if arg in ('df', 'pos') and style == SimulationStyle.SPATIAL_SPECTRAL:
                pos_grid = torch.zeros((3, len(self.df), self.pos.shape[1]), dtype=torch.float64)
                freq_grid, pos_grid[0] = torch.meshgrid(self.df, self.pos[0], indexing='ij')
                for axis in range(1, 3):
                    _, pos_grid[axis] = torch.meshgrid(self.df, self.pos[axis], indexing='ij')

                attr = freq_grid.flatten() if arg == 'df' else pos_grid.reshape(3, -1)
            else:
                attr = getattr(self, arg)

            # General handling for other attributes
            if hasattr(attr, 'waveform'):
                attr = attr.waveform

            kernel_args.append(self._convert_to_device(attr, gpu_available))

        return kernel_args

    @staticmethod
    def _convert_to_device(attr, gpu_available: bool):
        if gpu_available and isinstance(attr, torch.Tensor):
            return attr.to(device='cuda')
        elif not gpu_available and isinstance(attr, torch.Tensor):
            dtype = attr.dtype
            if dtype in (torch.float32, torch.float64):
                dtype = np.float64
            elif dtype in (torch.complex64, torch.complex128):
                dtype = np.complex128

            return np.array(attr, dtype=dtype)

        return attr

    def simulate(self, t1: float = np.inf, t2: float = np.inf, dt: float = 1,
                 initial_magnetisation: torch.Tensor = torch.tensor([0., 0., 1.]),
                 force_cpu: bool = False, verbose: bool = True) -> Magnetisation:
        if force_cpu:
            gpu_available = False
        else:
            gpu_available = torch.cuda.is_available()

        style = self.determine_style()

        # Prepare kernel arguments
        if style == SimulationStyle.RELAXATION:
            warnings.warn('Relaxation simulation has been selected. Setting dt to at least 100.0 us!')
            dt = max(100., dt)

        kernel_args = ([t1, t2, dt * 1e-6] + self.prepare_kernel_arguments(style, gpu_available) +
                       [self._convert_to_device(initial_magnetisation, gpu_available)])

        init_time = time.perf_counter()

        # Call the appropriate simulation kernel
        magnetisation = kernels[style][gpu_available](*kernel_args)

        if style == SimulationStyle.SPATIAL_SPECTRAL:
            num_iso = len(self.df) * len(self.pos[0])
        else:
            num_iso = len(self.df) if self.df is not None else self.pos.shape[1]

        simulation_steps = magnetisation.shape[1]

        if verbose:
            print(f'{style} took {time.perf_counter() - init_time:.3f}s for {num_iso} isochromats '
                  f'undergoing {simulation_steps} time steps using {"GPU" if gpu_available else "CPU"}. '
                  f'Total iterations = {num_iso * simulation_steps}, '
                  f'iters/s = {num_iso * simulation_steps / (time.perf_counter() - init_time):.2f}')

        # Move results to CPU if necessary
        simulation_result = magnetisation.to(device='cpu') if gpu_available else torch.from_numpy(magnetisation)

        return Magnetisation(simulation_result, style, dt)


class Magnetisation:
    def __init__(self, magnetisation: torch.Tensor, simulation_style: SimulationStyle, dt: float):
        self.magnetisation = magnetisation.squeeze()
        self.simulation_style = simulation_style
        self.dt = dt

    @property
    def times(self) -> torch.Tensor:
        duration = (self.magnetisation.shape[0] - 1) * self.dt

        return determine_times(int(duration), n=None, dt=self.dt)

    @property
    def num_isochromats(self) -> int:
        return self.magnetisation.shape[0]

    @property
    def simulation_steps(self) -> int:
        return self.magnetisation.shape[1]

    @property
    def mx(self) -> torch.Tensor:
        return self.magnetisation[..., 0]

    @property
    def my(self) -> torch.Tensor:
        return self.magnetisation[..., 1]

    @property
    def mxy(self) -> torch.Tensor:
        return torch.abs(self.mx + 1j * self.my)

    @property
    def mxy_phase(self) -> torch.Tensor:
        return torch.angle(self.mx + 1j * self.my)

    @property
    def mz(self) -> torch.Tensor:
        return self.magnetisation[..., 2]

    @property
    def inversion_efficiency(self) -> torch.Tensor:
        return (1 - self.mz) / 2
