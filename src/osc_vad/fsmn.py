from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple, List
from funasr_onnx import Fsmn_vad_online
import time


DEFAULT_MODEL_DIR = Path(__file__).parent / "assets" / "fsmn"


class FSMN:
    def __init__(
        self,
        device_id: int = -1,
        intra_op_num_threads: int = 0,
        max_end_sil: int | None = None,
        model_dir: Path | str = DEFAULT_MODEL_DIR,
    ):
        """FSMN model for voice activity detection.
        """
        self.infer = Fsmn_vad_online(
            model_dir=model_dir,
            device_id=device_id,
            intra_op_num_threads=intra_op_num_threads,
            max_end_sil=max_end_sil,
        )

        self.caches: Dict[
            str, Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, List[bool]]
        ] = {}

    def process_chunk(self, chunk: np.ndarray, cache_id: str, is_final: bool = False) -> bool:
        """Process a chunk of audio data.
        Args:
            chunk (np.ndarray): The chunk of audio data to process.
            cache_id (str): The cache_id to use for the cache.
        Returns:
            bool: True if the chunk is active, False otherwise.
        """
        assert len(chunk.shape) == 1, "Chunk must be 1D array."
        if cache_id not in self.caches:
            self.caches[cache_id] = []
        in_cache = self.caches.get(cache_id)
        param_dict = {"in_cache": in_cache, "is_final": is_final}
        chunk_segments = self.infer(audio_in=chunk, param_dict=param_dict)
        if len(chunk_segments) > 0:
            chunk_segments = chunk_segments[0]
        self.caches[cache_id] = param_dict["in_cache"]
        return chunk_segments

    def test(self, wav_path: str | Path):
        """Test the FSMN model.
        Args:
            wav_path (str | Path): The path to the wav file to test.
        """
        import soundfile
        speech, sample_rate = soundfile.read(wav_path)
        assert sample_rate == 16000, f"Sample rate must be 16000. Got {sample_rate}."
        speech_length = speech.shape[0]
        duration = speech_length / sample_rate
        sample_offset = 0
        step = 1600
        all_segments = []
        start = time.perf_counter()
        for sample_offset in range(0, speech_length, min(step, speech_length - sample_offset)):
            if sample_offset + step >= speech_length - 1:
                step = speech_length - sample_offset
                is_final = True
            else:
                is_final = False
            segments = self.process_chunk(
                chunk=speech[sample_offset: sample_offset + step],
                cache_id="test",
                is_final=is_final,
            )
            if segments:
                all_segments.extend(segments)
        spent = round(time.perf_counter() - start, 4)
        rtf = round(spent / duration, 4)
        speed = round(duration / spent, 4)
        print(f"Duration: {duration}s, Spent: {spent}s, RTF: {rtf}, Speed: {speed}x")
        return all_segments
