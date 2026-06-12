import math
from enum import Enum
from typing import Optional

import torch
from matplotlib import pyplot as plt

import sequence.rf
from sequence.core import SequenceObject
from sequence.gradient import Gradient
from sequence.rf import RFPulse


class EventType(Enum):
    RF = 0
    GRADIENT = 1
    ADC = 2


class ADC:
    def __init__(self, num_samples: int, receive_bandwidth: float = 50e3):
        self.num_samples = num_samples
        self.receive_bandwidth = receive_bandwidth

        self.data = torch.zeros((num_samples, 2))

    @property
    def dt(self) -> float:
        return 1e6 / self.receive_bandwidth

    @property
    def times(self) -> torch.Tensor:
        return torch.linspace(0, self.num_samples, self.num_samples) * self.dt

    @property
    def duration(self) -> float:
        return self.num_samples * self.dt

    @property
    def bandwidth_per_pixel(self) -> float:
        return self.receive_bandwidth / self.num_samples

    def set_data(self, data: torch.Tensor):
        self.data = data

    def display(self):
        plt.figure(figsize=(12, 6))

        plt.plot(self.times * 1e-3, torch.ones_like(self.times))
        plt.xlabel('Time (ms)')
        plt.grid()

        plt.subplots_adjust(hspace=0.1)
        plt.show()


SequenceEvent = SequenceObject | ADC


class PulseSequence:
    def __init__(self):
        self.events: dict[EventType, (float, SequenceEvent)] = {
            EventType.RF: [],
            EventType.GRADIENT: [],
            EventType.ADC: [],
        }

    def add_event(self, init_time: float, sequence_event: SequenceEvent):
        if isinstance(sequence_event, RFPulse):
            other_events = self.get_rf()
            event_type = EventType.RF
        elif isinstance(sequence_event, Gradient):
            other_events = self.get_gradients()
            event_type = EventType.GRADIENT
        elif isinstance(sequence_event, ADC):
            other_events = self.get_adc()
            event_type = EventType.ADC
        else:
            raise NotImplementedError

        if not other_events:
            self.events[event_type].append((init_time, sequence_event))

            return

        # Check for overlap
        for t0, event in other_events:
            start_time = t0
            end_time = t0 + event.duration

            if start_time <= init_time <= end_time or start_time <= init_time + sequence_event.duration <= end_time:
                raise ValueError('Pulse sequence contains overlapping pulses!')

        self.events[event_type].append((init_time, sequence_event))

    def get_rf(self) -> Optional[list[tuple[float, SequenceEvent]]]:
        rf_events = [(init_time, event) for init_time, event in self.events[EventType.RF]]

        if len(rf_events) == 0:
            return None
        else:
            return rf_events

    def get_gradients(self) -> Optional[list[tuple[float, SequenceEvent]]]:
        gradient_events = [(init_time, event) for init_time, event in self.events[EventType.GRADIENT]]

        if len(gradient_events) == 0:
            return None
        else:
            return gradient_events

    def get_adc(self) -> Optional[list[tuple[float, SequenceEvent]]]:
        adc_events = [(init_time, event) for init_time, event in self.events[EventType.ADC]]

        if len(adc_events) == 0:
            return None
        else:
            return adc_events

    @staticmethod
    def get_segments(init_times: list[float], max_time: float, events: list[SequenceEvent]):
        covered_segments = []
        for init_time, event in zip(init_times, events):
            covered_segments.append((init_time, init_time + event.duration))

        covered_segments.sort()

        empty_segments = []
        if covered_segments[0][0] > 0:
            empty_segments.append((0, covered_segments[0][0]))
        for i in range(len(covered_segments) - 1):
            empty_segments.append((covered_segments[i][1], covered_segments[i + 1][0]))

        if covered_segments[-1][1] < max_time:
            empty_segments.append((covered_segments[-1][1], max_time))

        for idx, (init_time, end_time) in enumerate(empty_segments):
            if end_time < init_time:
                empty_segments.pop(idx)

        return covered_segments, empty_segments

    def display(self):
        fig, ax = plt.subplots(figsize=(12, 6))

        rf_objects = self.get_rf() or []
        grad_objects = self.get_gradients() or []
        adc_objects = self.get_adc() or []
        all_events = rf_objects + grad_objects + adc_objects

        kernel_length = max([event.duration + init_time for init_time, event in all_events])
        max_length = math.ceil(kernel_length / 2000.0) * 2000.0

        if len(adc_objects) == 0:
            plt.plot([0, max_length * 1e-3], [1, 1], color='k')
        else:
            adc_start_times, adc_objects = zip(*self.get_adc())
            adc_segments, empty_adc_segments = self.get_segments(adc_start_times, max_length, adc_objects)

            for init_time, end_time in empty_adc_segments:
                plt.plot([init_time * 1e-3, end_time * 1e-3], [1, 1], color='k')
            for init_time, adc in zip(adc_start_times, adc_objects):
                plt.plot([init_time * 1e-3, init_time * 1e-3], [1, 2], color='k')
                plt.plot((init_time + adc.times) * 1e-3, 1 + torch.ones_like(adc.times), color='k')
                plt.plot([(init_time + adc.duration) * 1e-3, (init_time + adc.duration) * 1e-3], [1, 2], color='k')

        if len(grad_objects) == 0:
            plt.plot([0, max_length * 1e-3], [3, 3], color='k')
            plt.plot([0, max_length * 1e-3], [5, 5], color='k')
            plt.plot([0, max_length * 1e-3], [7, 7], color='k')
        else:
            grad_start_times, grad_objects = zip(*self.get_gradients())
            grad_segments, empty_grad_segments = self.get_segments(grad_start_times, max_length, grad_objects)

            max_gradient_amplitudes = max([torch.max(grad.waveform).item() for grad in grad_objects])

            for init_time, end_time in empty_grad_segments:
                plt.plot([init_time * 1e-3, end_time * 1e-3], [3, 3], color='k')
                plt.plot([init_time * 1e-3, end_time * 1e-3], [5, 5], color='k')
                plt.plot([init_time * 1e-3, end_time * 1e-3], [7, 7], color='k')
            for init_time, grad in zip(grad_start_times, grad_objects):
                plt.plot((init_time + grad.times) * 1e-3, 3 + grad.waveform[0] / max_gradient_amplitudes, color='k')
                plt.plot((init_time + grad.times) * 1e-3, 5 + grad.waveform[1] / max_gradient_amplitudes, color='k')
                plt.plot((init_time + grad.times) * 1e-3, 7 + grad.waveform[2] / max_gradient_amplitudes, color='k')

        if len(rf_objects) == 0:
            plt.plot([0, max_length * 1e-3], [9, 9], color='k')
        else:
            rf_start_times, rf_objects = zip(*rf_objects)
            rf_segments, empty_rf_segments = self.get_segments(rf_start_times, max_length, rf_objects)

            max_rf_amplitude = max([max(rf.magnitude) for rf in rf_objects])

            for init_time, end_time in empty_rf_segments:
                plt.plot([init_time * 1e-3, end_time * 1e-3], [9, 9], color='k')
            for init_time, rf in zip(rf_start_times, rf_objects):
                plt.plot((init_time + rf.times) * 1e-3, 9 + rf.magnitude / max_rf_amplitude, color='k')

        plt.xlabel('Time (ms)', fontsize=16)
        plt.xlim([0, max_length * 1e-3])
        plt.xticks(fontsize=14)

        plt.ylim([0, 10.1])
        plt.yticks([1, 3, 5, 7, 9], ['$ADC$', '$G_x$', '$G_y$', '$G_z$', '$RF$'], fontsize=14)

        plt.tick_params(axis='both', length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plt.show()


dt = 10
TE = 15e3
TR = 1000e3

ramp_time = 400
pulse_duration = 2400

excitation_gradient = sequence.gradient.trapezium(pulse_duration + 2 * ramp_time, ramp_time=ramp_time, axis=2, dt=dt)
excitation_gradient.amplitude[2] = 20e-3

excitation_pulse = sequence.rf.hamming_sinc(pulse_duration, 3200, dt=dt)
excitation_pulse.amplitude = excitation_pulse.set_optimal_amplitude(torch.pi / 3)

inversion_pulse = sequence.rf.hyperbolic_secant(5000, mu=5, bandwidth=2682, dt=dt)
inversion_pulse.amplitude = inversion_pulse.set_optimal_amplitude(torch.pi)

adc = ADC(384)

inversion_recovery = PulseSequence()
inversion_recovery.add_event(0, excitation_gradient)
inversion_recovery.add_event(ramp_time, excitation_pulse)
inversion_recovery.add_event(TE / 2 - inversion_pulse.duration / 2, inversion_pulse)
inversion_recovery.add_event(TE - adc.duration / 2, adc)

inversion_recovery.display()
