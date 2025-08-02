import os
import librosa
import soundfile as sf
import numpy as np
from typing import Tuple, Optional

def load_audio(file_path: str, sr: int = 16000, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Carrega um arquivo de áudio com tratamento de erros.
    
    Args:
        file_path: Caminho para o arquivo de áudio.
        sr: Taxa de amostragem desejada (None para manter original).
        mono: Forçar áudio mono se True.
        
    Returns:
        (audio_data, sr) - Dados de áudio (shape: [n_samples] ou [channels, n_samples]) e taxa real.
        
    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o arquivo não for um áudio válido.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    try:
        audio, sr = librosa.load(file_path, sr=sr, mono=mono)
        return audio, sr
    except Exception as e:
        raise ValueError(f"Falha ao carregar {file_path}: {str(e)}")

def save_audio(
    file_path: str, 
    audio_data: np.ndarray, 
    sr: int, 
    output_directory: str, 
    prefix: str = "processed",
    subtype: str = "PCM_16"
) -> str:
    """
    Salva áudio em formato WAV com metadados.
    
    Args:
        file_path: Caminho original (para extrair nome do arquivo).
        audio_data: Dados de áudio (shape: [n_samples] ou [channels, n_samples]).
        sr: Taxa de amostragem.
        output_directory: Diretório de saída (criado se não existir).
        prefix: Prefixo do arquivo de saída.
        subtype: Formato de amostra (e.g., "PCM_16", "FLOAT").
        
    Returns:
        Caminho completo do arquivo salvo.
    """
    os.makedirs(output_directory, exist_ok=True)
    filename = f"{prefix}_{os.path.basename(file_path)}"
    output_path = os.path.join(output_directory, filename)
    
    sf.write(
        output_path, 
        audio_data.T if audio_data.ndim > 1 else audio_data, 
        sr, 
        subtype=subtype
    )
    return output_path

def normalize_audio(audio_data: np.ndarray, target_range: Tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
    """
    Normaliza o áudio para um intervalo específico.
    
    Args:
        audio_data: Dados de áudio (shape qualquer).
        target_range: Intervalo alvo (min, max).
        
    Returns:
        Áudio normalizado.
    """
    min_val, max_val = np.min(audio_data), np.max(audio_data)
    if np.abs(max_val - min_val) < 1e-6:
        return audio_data * 0  # Evita divisão por zero
        
    audio_normalized = (audio_data - min_val) / (max_val - min_val)
    return audio_normalized * (target_range[1] - target_range[0]) + target_range[0]

def trim_silence(audio_data: np.ndarray, top_db: int = 20, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Remove silêncio do início/fim com parâmetros personalizáveis.
    
    Args:
        audio_data: Dados de áudio (shape: [n_samples]).
        top_db: Limiar em dB para detecção de silêncio.
        frame_length: Tamanho do frame para análise.
        hop_length: Hop size para análise.
        
    Returns:
        Áudio com silêncio removido.
    """
    return librosa.effects.trim(
        audio_data, 
        top_db=top_db, 
        frame_length=frame_length, 
        hop_length=hop_length
    )[0]