import numpy as np
from pystoi import stoi
from typing import Tuple, Union


def adjust_audio_length(
    ref_audio: np.ndarray, proc_audio: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Ajusta os áudios para terem o mesmo comprimento (o menor dos dois)"""
    min_len = min(len(ref_audio), len(proc_audio))
    return ref_audio[:min_len], proc_audio[:min_len]


def calculate_snr(original_audio: np.ndarray, processed_audio: np.ndarray) -> float:
    """
    Calcula a relação sinal-ruído (SNR) entre o áudio original e o processado.

    Args:
        original_audio: Sinal de áudio original (shape: [n_samples]).
        processed_audio: Sinal de áudio processado (shape: [n_samples]).

    Returns:
        SNR em dB. Retorna `-np.inf` se o áudio original for silêncio
        ou `np.inf` se o ruído for nulo.
    """
    original_audio, processed_audio = adjust_audio_length(
        original_audio, processed_audio
    )

    if len(original_audio) != len(processed_audio):
        raise ValueError("Os áudios devem ter o mesmo comprimento")

    ref_energy = np.dot(original_audio, original_audio)
    if ref_energy < 1e-10:  # Evita divisão por zero
        return float("-inf")

    optimal_scaling = np.dot(original_audio, processed_audio) / ref_energy
    projection = optimal_scaling * original_audio
    noise = processed_audio - projection

    signal_power = np.sum(projection**2)
    noise_power = np.sum(noise**2)

    return 10 * np.log10(
        signal_power / (noise_power + 1e-10)
    )  # epsilon para estabilidade


def calculate_stoi(
    original_audio: np.ndarray, processed_audio: np.ndarray, sr: int
) -> float:
    """
    Calcula o STOI (Short-Time Objective Intelligibility) entre os áudios.

    Args:
        original_audio: Sinal de áudio original (shape: [n_samples]).
        processed_audio: Sinal de áudio processado (shape: [n_samples]).
        sr: Taxa de amostragem (Hz).

    Returns:
        Valor do STOI entre 0 (ininteligível) e 1 (perfeito).

    Raises:
        ValueError: Se os áudios forem mais curtos que 1 segundo.
    """
    min_len = min(len(original_audio), len(processed_audio))
    if min_len < sr:
        raise ValueError(f"Áudio muito curto ({min_len/sr:.2f}s). Mínimo: 1 segundo.")

    return stoi(original_audio[:min_len], processed_audio[:min_len], sr, extended=False)


def calculate_sdr(reference_audio: np.ndarray, estimated_audio: np.ndarray) -> float:
    """
    Calcula a relação sinal-distorção (SDR) entre os áudios.
    Versão simplificada (SDR não depende da taxa de amostragem).

    Args:
        reference_audio: Sinal de referência (shape: [n_samples]).
        estimated_audio: Sinal estimado (shape: [n_samples]).

    Returns:
        SDR em dB. Retorna `np.inf` se a distorção for nula.
    """
    reference_audio, estimated_audio = adjust_audio_length(
        reference_audio, estimated_audio
    )
    if len(reference_audio) != len(estimated_audio):
        raise ValueError("Os áudios devem ter o mesmo comprimento")

    signal_power = np.sum(reference_audio**2)
    distortion_power = np.sum((estimated_audio - reference_audio) ** 2)

    return 10 * np.log10(signal_power / (distortion_power + 1e-10))


def combined_metric(
    metrics: Tuple[float, float, float],
    weights: Tuple[float, float, float] = (0.3, 0.3, 0.4),
) -> float:
    """
    Combina SNR, STOI e SDR em uma métrica única.

    Args:
        metrics: Tupla com (SNR, STOI, SDR).
        weights: Pesos para (SNR, STOI, SDR). Soma deve ser 1.

    Returns:
        Métrica combinada (SNR e SDR em dB, STOI escalado).
    """
    snr, stoi_value, sdr = metrics
    return weights[0] * snr + weights[1] * stoi_value + weights[2] * sdr
