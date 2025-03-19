import onnxruntime as ort
from typing import List
import numpy as np
from pathlib import Path
from loguru import logger


class ORTInference:
    def __init__(
        self,
        onnx_model_path: str | Path | None = None,
        device_id: int | None = None,
        intra_op_num_threads: int = 0,
        verbose: bool = True,
        session_options: ort.SessionOptions | None = None,
    ):
        self.onnx_model_path = onnx_model_path
        self.device_id = device_id
        self.intra_op_num_threads = intra_op_num_threads
        self.verbose = verbose
        self.set_session(session_options=session_options)

    def __call__(self, input_content: List[np.ndarray]) -> np.ndarray:
        return self.run(input_content)

    def run(self, input_content: List[np.ndarray]) -> np.ndarray:
        if self.session is None:
            raise ONNXRuntimeError("ONNXRuntime session is not set.")
        input_dict = dict(zip(self.get_input_names(), input_content))
        try:
            return self.session.run(self.get_output_names(), input_dict)
        except Exception as e:
            raise ONNXRuntimeError(f"ONNXRuntime inference failed: {e}")

    def get_input_names(
        self,
    ):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(
        self,
    ):
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key: str = "character"):
        return self.meta_dict[key].splitlines()

    def have_key(self, key: str = "character") -> bool:
        self.meta_dict = self.session.get_modelmeta().custom_metadata_map
        if key in self.meta_dict.keys():
            return True
        return False

    @staticmethod
    def _verify_model(model_path):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exists.")
        if not model_path.is_file():
            raise FileExistsError(f"{model_path} is not a file.")

    def set_session(self, session_options: ort.SessionOptions | None = None):
        if session_options is None:
            sess_opt = ort.SessionOptions()
            sess_opt.intra_op_num_threads = self.intra_op_num_threads
            sess_opt.log_severity_level = 4
            sess_opt.enable_cpu_mem_arena = False
            sess_opt.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        else:
            sess_opt = session_options
        EP_list = []
        cuda_ep = "CUDAExecutionProvider"
        cpu_ep = "CPUExecutionProvider"
        if ort.get_device() == "GPU" and cuda_ep in ort.get_available_providers():
            if self.device_id is None:
                device_id = 0
            else:
                device_id = self.device_id
            cuda_provider_options = {
                "device_id": str(device_id),
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": "true",
            }
            EP_list.append((cuda_ep, cuda_provider_options))
            if self.verbose:
                logger.info(f"using onnxruntime-gpu with device_id: {device_id}")
        else:
            cpu_provider_options = {
                "arena_extend_strategy": "kSameAsRequested",
            }
            EP_list.append((cpu_ep, cpu_provider_options))
            if self.verbose:
                logger.info("using onnxruntime-cpu")

        self._verify_model(self.onnx_model_path)
        self.session = ort.InferenceSession(
            self.onnx_model_path, sess_options=sess_opt, providers=EP_list
        )

    @property
    def model_dtype(self):
        self.onnx_model_path = Path(self.onnx_model_path)
        if self.onnx_model_path.stem.endswith("fp32"):
            return np.float32
        elif self.onnx_model_path.stem.endswith("fp16"):
            return np.float16
        else:
            raise ValueError("model dtype not supported.")


class ONNXRuntimeError(Exception):
    pass
