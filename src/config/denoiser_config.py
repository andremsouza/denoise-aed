from dataclasses import dataclass


@dataclass
class DenoiserConfig(object):
    process_variance: float = 1e-5
    initial_measurement_noise: float = 1e-4
    adaptation_interval: int = 100
    window_size: int = 1024 
    hop_size: int = 512
    noise_reduction_factor: float = 1.0
    noise_window_duration: float = 0.1
    sample_rate: int = 16000
    alpha: float = 0.95
    beta: float = 0.98
    adaptive: bool = True