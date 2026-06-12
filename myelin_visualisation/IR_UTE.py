import math
import torch
import sequence
import numpy as np
import matplotlib.pyplot as plt
from simulate.kernels import cpu_non_selective

class IR_UTE():
    def __init__(self, dt, df, t1, t2):
        """
        dt: time steps (us)
        df: off-resonance frequency (Hz)
        """
        self.dt = dt
        self.df = df
        self.t1 = t1
        self.t2 = t2
        self.non_selective_dt = self.dt * 1e-6
        self.n_on_resonance = int((len(self.df))/2)

        # Inversion recovery
        self.duration_ir = 1500

        # Transverse excitation
        self.duration_sinc = 470
        self.bw_sinc = 2700
        self.dt_sinc = 10
        self.tx_rx = 7
        self.TR = 5 * 0.2 * 1e6 # us
        self.time_between_inversion = self.TR - self.duration_ir # us
        
        

    def inversion_recovery(self):
        initial_magnetisation = np.array([0.0, 0.0, 1.0])

        pulse = sequence.rf.hyperbolic_secant(duration = self.duration_ir, mu=5, bandwidth=2400, dt= self.dt)
        optimal_amplitude = pulse.get_optimal_amplitude(torch.pi)
        pulse.amplitude = optimal_amplitude
        self.rf = pulse.waveform
        self.rf = self.rf.detach().cpu().numpy().astype(np.complex128)

        zero_padding = int(4.5/self.non_selective_dt)
        rf_with_padding = np.pad(self.rf, (0, zero_padding))
        self.magnetisation_ir = cpu_non_selective(self.t1, self.t2, self.non_selective_dt, self.df, rf_with_padding, initial_magnetisation)
        return self.magnetisation_ir

    def calculate_nulling_point(self):
        """
        Since we aim to null the white matter in this scenario, 
        this equation should only be used to determine the nulling point when the object is white matter.

        Note: call inversion_recovery() before calling the current function.
        """
        n_on_resonance = int((len(self.df))/2)
        n_inversion_steps = self.rf.shape[0]
        mag_z = self.magnetisation_ir[n_on_resonance, n_inversion_steps + 1 :, 2]
        idx = np.argmin(np.abs(mag_z))

        self.n_nullout_step = n_inversion_steps + idx + 1
        t_nullout = int(self.n_nullout_step * self.dt) # time in us
        return t_nullout
    

    def transverse_excitation(self, n_readout, t_readout, t_nullout, flip_angle):
        self.flip_angle = flip_angle
        n_nullout_step = int(t_nullout / self.dt)

        pulse_sinc = sequence.rf.hamming_sinc(self.duration_sinc, self.bw_sinc, dt=self.dt_sinc)
        pulse_sinc.amplitude = pulse_sinc.get_optimal_amplitude(flip_angle)
        rf_sinc = pulse_sinc.waveform.detach().cpu().numpy().astype(np.complex128)
        initial_magnetisation_sinc = self.magnetisation_ir[self.n_on_resonance, n_nullout_step, :]
       
        # if is_white_matter:
        #     initial_magnetisation_sinc = np.array([0.0, 0.0, 0.0])

        
        zero_padding = int((self.time_between_inversion - self.duration_sinc) / self.dt_sinc)
        pulse_sinc_with_padding = np.pad(rf_sinc, (0, zero_padding))

        t_sinc = self.duration_sinc + self.tx_rx                  # us
        t_readout_us = t_readout * 1e6                            # s -> us
        t_readout_end_index = int((t_sinc + 2 * t_readout_us) / self.dt)
        print('t_readout_end_index:', t_readout_end_index)

        n_total_steps = int(self.time_between_inversion / self.dt)
        full_mag_series = np.zeros((len(self.df), n_total_steps, 3), dtype=np.float64)
        write_start = 0
        excitation_time, mag_readout_start, t_mag_readout_start = [], [], []
        for n in range(n_readout):
            print('write_start:', write_start)
            print('initial_mag:', initial_magnetisation_sinc)
            excitation_time.append(write_start)
            t_mag_readout_start.append((write_start + int(np.round(t_sinc/self.dt))))
            magnetisation_rf_sinc = cpu_non_selective(
                self.t1,
                self.t2,
                self.non_selective_dt,
                self.df,
                pulse_sinc_with_padding,
                initial_magnetisation_sinc
            )
            mag_at_t_1 = magnetisation_rf_sinc[self.n_on_resonance, 1, :]
            print(f"Angle={flip_angle}, M after 1 step: {mag_at_t_1}")

            # choose how much of this readout to keep
            if n < n_readout - 1:
                segment = magnetisation_rf_sinc[:, :t_readout_end_index, :]
                initial_magnetisation_sinc = magnetisation_rf_sinc[self.n_on_resonance, t_readout_end_index, :]
                initial_magnetisation_sinc[0] = 0.0
                initial_magnetisation_sinc[1] = 0.0
            else:
                segment = magnetisation_rf_sinc

            seg_len = segment.shape[1]
            write_end = min(write_start + seg_len, n_total_steps)
            mag_readout_start.append(segment[self.n_on_resonance, int(np.round(t_sinc/self.dt)), :])

            # truncate segment if it would overflow
            full_mag_series[:, write_start:write_end, :] = segment[:, :write_end - write_start, :]

            write_start = write_end
            
            if write_start >= n_total_steps:
                break

        excitation_time = np.array(excitation_time)
        mag_readout_start = np.array(mag_readout_start)
        t_mag_readout_start = np.array(t_mag_readout_start)
        self.readout_ends = t_mag_readout_start[-1] + int(np.ceil(t_readout_us/self.dt))
        return full_mag_series, excitation_time, mag_readout_start, t_mag_readout_start
   
                    
        
    def magnetization_display_z(self, magnetization):
        n_steps = magnetization.shape[1] -1
        time = np.arange(n_steps + 1) * self.non_selective_dt  # seconds
        plt.figure(figsize=(6,4))
        plt.plot(time[:self.readout_ends], magnetization[int(len(self.df)/2), :self.readout_ends, 2], label='Myelin (Mz)')
        plt.xlabel('Time (s)')
        plt.ylabel('Longitudinal Magnetisation Mz')
        plt.title('Inversion Recovery (IR)')
        plt.grid(True)
        plt.legend(title='Group')
        plt.show()

    def magnetization_display_xy(self, magnetization, title):
        magnetization_xy = np.sqrt(np.power(magnetization[self.n_on_resonance, :, 0], 2) + np.power(magnetization[self.n_on_resonance, :, 1], 2))
        n_steps =  magnetization.shape[1]

        time = np.arange(n_steps) * self.dt  # micro seconds
        plt.figure(figsize=(6,4))
        plt.plot(time[:self.readout_ends], magnetization_xy[:self.readout_ends], label= title)

        plt.xlabel('Time (us)')
        plt.ylabel('Transverse Magnetisation Mxy')
        #plt.ylim([0,1])
        plt.title(f'{int(np.round(180/math.pi * self.flip_angle))} Degrees Exciation at Resonance (Sinc Pulse)')
        plt.grid(True)
        plt.legend(title = "Group")
        plt.show()


