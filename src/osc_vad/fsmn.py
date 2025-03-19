from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from osc_vad.utils import ORTInference
from typing import Dict, Tuple, List, Literal
import onnxruntime


DEFAULT_F32_MODEL = Path(__file__).parent / "assets" / "fsmn" / "fp32.onnx"
DEFAULT_F16_MODEL = Path(__file__).parent / "assets" / "fsmn" / "fp16.onnx"


class FSMN:
    def __init__(
        self,
        precision: Literal["fp32", "fp16"] = "fp32",
        one_minus_speech_thresh: float = 1.0,
        background_noise_db_init: float = 40.0,
        snr_thresh: float = 10.0,
        fusion_thresh: float = 0.5,
        min_speech_duration: float = 0.2,
        speaking_score: float = 0.5,
        silence_score: float = 0.5,
        session_options: onnxruntime.SessionOptions = None,
    ):
        """FSMN model for voice activity detection.

        Args:
            one_minus_speech_thresh (float, optional): The judge factor for VAD model. Defaults to 1.0.
            background_noise_db_init (float, optional): An initial value for the background. More smaller values indicate a quieter environment. Unit: dB. When using denoised audio, set this value to be smaller.
            snr_thresh (float, optional): The judge factor for VAD model. Unit: dB. Defaults to 0.0.
            fusion_thresh (float, optional): The judge factor for VAD model. Defaults to 0.5.
            min_speech_duration (float, optional): A judgment factor used to filter the vad results. Unit: seconds. Defaults to 0.2.
            speaking_score (float, optional): A judgment factor used to determine whether the state is speaking or not. A larger value makes activation more difficult. Defaults to 0.5.
            silence_score (float, optional): A judgment factor used to determine whether the state is silent or not. A larger value makes it easier to cut off speaking. Defaults to 0.5.
            session_options (onnxruntime.SessionOptions, optional): ONNX Runtime session options. Defaults to None.
        """
        if precision == "fp32":
            onnx_model_path = DEFAULT_F32_MODEL
        elif precision == "fp16":
            onnx_model_path = DEFAULT_F16_MODEL
        else:
            raise ValueError(f"Invalid precision: {precision}")
        self.infer = ORTInference(
            onnx_model_path=onnx_model_path, session_options=session_options
        )

        self.noise_average_dB = np.array(
            [background_noise_db_init + snr_thresh], dtype=self.infer.model_dtype
        )
        self.one_minus_speech_thresh = np.array(
            [one_minus_speech_thresh], dtype=self.infer.model_dtype
        )
        self.background_noise_db_init = background_noise_db_init
        self.snr_thresh = snr_thresh
        self.speaking_score = speaking_score
        self.silence_score = silence_score

        self.caches: Dict[
            str, Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, List[bool]]
        ] = {}

    def process_chunk(self, chunk: np.ndarray, cache_id: str) -> bool:
        """Process a chunk of audio data.
        Args:
            chunk (np.ndarray): The chunk of audio data to process.
            cache_id (str): The cache_id to use for the cache.
        Returns:
            bool: True if the chunk is active, False otherwise.
        """
        assert len(chunk.shape) == 1, "Chunk must be 1D array."
        chunk = chunk[None, None, :]
        if cache_id not in self.caches:
            _ = self.create_cache(cache_id)
        cache_0, cache_1, cache_2, cache_3, noise_average_dB, silence_ls = (
            self.caches.get(cache_id)
        )
        score, cache_0, cache_1, cache_2, cache_3, noisy_dB = self.infer.run(
            [
                chunk,
                cache_0,
                cache_1,
                cache_2,
                cache_3,
                self.noise_average_dB,
                self.one_minus_speech_thresh,
            ]
        )
        if silence_ls[-1]:
            if score >= self.speaking_score:
                silence_ls.append(False)
        else:
            if score <= self.silence_score:
                silence_ls.append(True)
        noise_average_dB = 0.5 * (noise_average_dB + noisy_dB) + self.snr_thresh
        self.caches[cache_id] = (
            cache_0,
            cache_1,
            cache_2,
            cache_3,
            noise_average_dB,
            silence_ls,
        )
        is_active = not silence_ls[-1]
        return is_active

    def create_cache(self, cache_id: str):
        """Create a cache for the given cache_id.
        Args:
            cache_id (str): The cache_id to create a cache for.
        """
        dtype = self.infer.model_dtype
        cache0 = np.zeros((1, 128, 19, 1), dtype=dtype)
        cache1 = np.zeros((1, 128, 19, 1), dtype=dtype)
        cache2 = np.zeros((1, 128, 19, 1), dtype=dtype)
        cache3 = np.zeros((1, 128, 19, 1), dtype=dtype)
        noise_average_dB = np.array(
            [self.background_noise_db_init + self.snr_thresh], dtype=dtype
        )
        silence_ls = [True]
        self.caches[cache_id] = (
            cache0,
            cache1,
            cache2,
            cache3,
            noise_average_dB,
            silence_ls,
        )
        return self.caches[cache_id]
