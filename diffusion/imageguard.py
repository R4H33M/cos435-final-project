import sys
import types
import importlib.machinery
from pathlib import Path


# Ran into a lot of issues importing deepspeed (crashes on import due to no CUDA_HOME in Modal). 
# This is a stub deepspeed that runs before the real pkg gets imported (the real pkg is never used in ImageGuard)
# Used Codex to help with debugging deepspeed import issue + writing the stub code
def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    m.__path__ = []
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class _GatheredParameters:
    def __init__(self, params, *a, **kw):
        self.params = params

    def __enter__(self):
        return self.params

    def __exit__(self, *exc):
        return False


class _ZeroParamStatus:
    NOT_AVAILABLE = AVAILABLE = INFLIGHT = "STUB"


class _DeepSpeedEngine:
    pass


sys.modules["deepspeed"] = _stub(
    "deepspeed",
    zero=_stub("deepspeed.zero", GatheredParameters=_GatheredParameters),
    DeepSpeedEngine=_DeepSpeedEngine,
)
sys.modules["deepspeed.zero"] = sys.modules["deepspeed"].zero
sys.modules["deepspeed.runtime"] = _stub("deepspeed.runtime")
sys.modules["deepspeed.runtime.zero"] = _stub("deepspeed.runtime.zero")
sys.modules["deepspeed.runtime.zero.partition_parameters"] = _stub(
    "deepspeed.runtime.zero.partition_parameters",
    ZeroParamStatus=_ZeroParamStatus,
)


# --------------- deepspeed workaround ends here


# https://huggingface.co/OpenSafetyLab/ImageGuard
class ImageGuardReward:
    """ImageGuard wrapper. Higher score -> ImageGuard classified the image as unsafe."""

    def __init__(
        self,
        device="cuda",
        model_repo: str = "OpenSafetyLab/ImageGuard",
        base_model: str = "internlm/internlm-xcomposer2-vl-7b",
    ):
        import yaml
        from huggingface_hub import snapshot_download

        self.device = device
        repo_path = snapshot_download(model_repo)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from utils.arguments import DataArguments, EvalArguments, LoraArguments, ModelArguments
        from utils.conv_utils import safe_query
        from utils.img_utils import ImageProcessor
        from utils.model_utils import init_model

        lora_path = Path(repo_path) / "lora"
        with open(lora_path / "config.yaml", encoding="utf-8") as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)

        model_cfg = config["model_cfg"]
        data_cfg = config["data_cfg"]["data_cfg"]
        lora_cfg = config["lora_cfg"]
        training_cfg = config["training_cfg"]
        model_cfg["model_name"] = "Internlm"
        data_cfg["train"]["model_name"] = "Internlm"

        model_args = ModelArguments(model_name_or_path=base_model)
        data_args = DataArguments()
        lora_args = LoraArguments()
        lora_args.lora_alpha = lora_cfg["lora_alpha"]
        lora_args.lora_bias = lora_cfg["lora_bias"]
        lora_args.lora_dropout = lora_cfg["lora_dropout"]
        lora_args.lora_r = lora_cfg["lora_r"]
        lora_args.lora_target_modules = lora_cfg["lora_target_modules"]
        lora_args.lora_weight_path = str(lora_path)

        eval_args = EvalArguments()
        eval_args.max_length = training_cfg["max_length"]
        eval_args.fix_vit = training_cfg["fix_vit"]
        eval_args.fix_sampler = training_cfg["fix_sampler"]
        eval_args.use_lora = training_cfg["use_lora"]
        eval_args.gradient_checkpointing = training_cfg["gradient_checkpointing"]

        self.model, self.tokenizer = init_model(
            model_args.model_name_or_path,
            eval_args,
            data_args,
            lora_args,
            model_cfg,
        )
        self.model.eval()
        self.model.to(device).eval().half()
        self.model.tokenizer = self.tokenizer
        self.prompt = safe_query("Internlm")
        self.image_processor = ImageProcessor(image_size=490)

    def score(self, images_tensor: "torch.Tensor") -> "torch.Tensor":
        import torch

        scores = []
        with torch.no_grad():
            for image in images_tensor:
                pil_image = self._tensor_to_pil(image)
                processed = self.image_processor(pil_image)[None, :, :, :].to(self.device)
                with torch.cuda.amp.autocast(enabled=str(self.device).startswith("cuda")):
                    response, _ = self.model.chat(
                        self.model.tokenizer,
                        self.prompt,
                        processed,
                        history=[],
                        do_sample=False,
                        meta_instruction=None,
                    )
                scores.append(1.0 if "unsafe" in response else 0.0)

        return torch.tensor(scores, device=images_tensor.device, dtype=torch.float32)

    @staticmethod
    def _tensor_to_pil(image: "torch.Tensor") -> "Image.Image":
        import numpy as np
        from PIL import Image

        array = (image.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)
