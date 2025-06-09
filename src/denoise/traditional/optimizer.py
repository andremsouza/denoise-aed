import os
import importlib
from typing import Dict, Any, Tuple

import librosa
import mlflow
from omegaconf import OmegaConf

from src.traditional.spectral_subtraction import SpectralSubtraction
from src.traditional.kalman import AdaptiveKalman
from src.traditional.sdrom import SDROM
from src.utils.metrics import (
    calculate_snr,
    calculate_stoi,
    calculate_sdr,
    combined_metric,
)


class TraditionalOptimizer:
    @staticmethod
    def optimize(filter_type, input_path, output_dir, config):
        """
        Otimiza um filtro tradicional chamando o método 'train' da classe especificada.
        """
        # Mapeamento do filtro para módulo e classe
        filter_to_class = {
            "sdrom": ("sdrom", "SDROM"),
            "kalman": ("kalman", "AdaptiveKalman"),
            "spectral": ("spectral_subtraction", "SpectralSubtraction"),
        }

        module_info = filter_to_class.get(filter_type)
        if module_info is None:
            raise ValueError(f"Filtro '{filter_type}' não encontrado!")

        module_name, class_name = module_info
        try:
            # Importar módulo e classe
            module = importlib.import_module(f"tradicional.{module_name}")
            cls = getattr(module, class_name)
            instance = cls()  # Instanciar classe

            # Passar os argumentos diretamente de 'config'
            output_path, metrics = instance.train(
                input_path=input_path,
                output_dir=output_dir,
                **config,  # Parâmetros específicos para cada modelo
            )

            print(f"Processamento concluído com o filtro: {filter_type}")
            return output_path, metrics

        except ImportError as e:
            raise ImportError(f"Erro ao importar o módulo '{module_name}': {str(e)}")
        except AttributeError as e:
            raise AttributeError(
                f"A classe '{class_name}' não encontrada no módulo '{module_name}': {str(e)}"
            )

    @staticmethod
    def optimize_all(
        input_path: str,
        output_dir: str,
        configs: Dict[str, Dict[str, Any]],
        mode: str = "combined",
    ) -> Dict[str, Tuple[str, dict]]:
        """Otimiza todos os filtros configurados."""
        results = {}
        for filter_type, config in configs.items():
            results[filter_type] = TraditionalOptimizer.optimize(
                filter_type, input_path, output_dir, config, mode
            )
        return results

    @staticmethod
    def evaluate(
        filter_type: str, input_path: str, reference_path: str, config: dict
    ) -> float:
        """
        Avalia um filtro tradicional usando as métricas definidas.
        """
        # Processar áudio
        output_path, _ = TraditionalOptimizer.optimize(
            filter_type=filter_type,
            input_path=input_path,
            output_dir=os.path.join("temp_eval"),
            config=config,
        )

        # Carregar áudios
        reference_audio, sr = librosa.load(reference_path, sr=None)
        processed_audio, _ = librosa.load(output_path, sr=sr)

        # Calcular métricas
        min_len = min(len(reference_audio), len(processed_audio))
        ref = reference_audio[:min_len]
        proc = processed_audio[:min_len]

        snr = calculate_snr(ref, proc)
        stoi_val = calculate_stoi(ref, proc, sr)
        sdr = calculate_sdr(ref, proc)

        return combined_metric((snr, stoi_val, sdr))

    @staticmethod
    def optimize_with_evaluation(
        filter_type: str,
        input_path: str,
        reference_path: str,
        output_dir: str,
        config: dict,
    ) -> Tuple[str, dict]:
        """
        Otimiza e avalia o resultado com métricas de qualidade.
        Retorna o caminho do arquivo processado e as métricas calculadas.
        """
        # Processa o áudio
        output_path, _ = TraditionalOptimizer.optimize(
            filter_type=filter_type,
            input_path=input_path,
            output_dir=output_dir,
            config=config,
        )

        # Carrega os áudios
        reference_audio, sr = librosa.load(reference_path, sr=None)
        processed_audio, _ = librosa.load(output_path, sr=sr)

        # Calcula métricas
        metrics = {
            "snr": calculate_snr(reference_audio, processed_audio),
            "stoi": calculate_stoi(reference_audio, processed_audio, sr),
            "sdr": calculate_sdr(reference_audio, processed_audio),
        }
        metrics["combined"] = combined_metric(
            (metrics["snr"], metrics["stoi"], metrics["sdr"])
        )

        return output_path, metrics

    @staticmethod
    def load_config():
        """
        Carrega as configurações do arquivo models.yaml.
        """
        config_path = "configs/models.yaml"  # Substitua com o caminho real do arquivo
        config = OmegaConf.load(config_path)
        return config.traditional
