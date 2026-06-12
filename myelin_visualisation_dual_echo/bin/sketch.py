import torch
import sequence
import numpy as np
import matplotlib.pyplot as plt
from simulate.kernels import cpu_non_selective

def transverse_excitation(self, n_readout, t_readout, t_nullout, flip_angle):
    n_nullout_step = int(t_nullout / self.dt)

    pulse_sinc = sequence.rf.hamming_sinc(
        self.duration_sinc, self.bw_sinc, dt=self.dt_sinc
    )
    pulse_sinc.amplitude = pulse_sinc.get_optimal_amplitude(flip_angle)

    rf_sinc = pulse_sinc.waveform.detach().cpu().numpy().astype(np.complex128)

    initial_magnetisation_sinc = self.magnetisation_ir[
        self.n_on_resonance, n_nullout_step, :
    ]

    zero_padding = int((self.time_between_inversion - self.duration_sinc) / self.dt_sinc)
    pulse_sinc_with_padding = np.pad(rf_sinc, (0, zero_padding))

    t_sinc = self.duration_sinc + self.tx_rx                  # us
    t_readout_us = t_readout * 1e6                            # s -> us
    t_readout_index = int((t_sinc + 2 * t_readout_us) / self.dt)
    print('t_readout_index:', t_readout_index)
    n_total_steps = int(self.time_between_inversion / self.dt)
    m = np.zeros((len(self.df), n_total_steps, 3), dtype=np.float64)
    write_start = 0

    for n in range(n_readout):
        print(initial_magnetisation_sinc)
        magnetisation_rf_sinc = cpu_non_selective(
            self.t1,
            self.t2,
            self.non_selective_dt,
            self.df,
            pulse_sinc_with_padding,
            initial_magnetisation_sinc
        )

        # choose how much of this readout to keep
        if n < n_readout - 1:
            segment = magnetisation_rf_sinc[:, :t_readout_index, :]
            initial_magnetisation_sinc = magnetisation_rf_sinc[
                self.n_on_resonance, t_readout_index, :
            ]
        else:
            segment = magnetisation_rf_sinc

        seg_len = segment.shape[1]
        write_end = min(write_start + seg_len, n_total_steps)

        # truncate segment if it would overflow
        m[:, write_start:write_end, :] = segment[:, :write_end - write_start, :]

        write_start = write_end

        if write_start >= n_total_steps:
            break

    return m

t_readout_start_index = excitation_time + int(t_sinc/self.dt)
        print('int(t_sinc/self.dt):', int(t_sinc/self.dt))