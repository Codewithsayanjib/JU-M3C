"""MLX load/unload, chat templating, greedy generation with timing."""
import gc
import time
from pathlib import Path

import mlx.core as mx
import yaml
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parent.parent
SEED = 0


def model_config(key: str) -> dict:
    cfg = yaml.safe_load(open(ROOT / "configs" / "models.yaml"))["models"]
    if key not in cfg:
        raise KeyError(f"unknown model key {key}; known: {list(cfg)}")
    return cfg[key]


ADAPTER_FILES = ("adapter_config.json", "adapters.safetensors")


def check_adapter(path) -> Path:
    """Validate an mlx_lm LoRA adapter directory before any weights are loaded."""
    p = Path(path)
    missing = [f for f in ADAPTER_FILES if not (p / f).is_file()]
    if missing:
        raise FileNotFoundError(f"adapter dir {p} is missing {missing}")
    return p.resolve()


class Model:
    """One loaded model. Use as a context manager so memory is freed afterwards."""

    def __init__(self, key: str, adapter_path: str | None = None):
        """adapter_path: optional fine-tuned LoRA adapter directory (mlx_lm.lora output:
        adapter_config.json + adapters.safetensors). None = zero-shot base model; that path
        calls mlx_lm.load exactly as before, so zero-shot behaviour is unchanged."""
        self.key = key
        self.cfg = model_config(key)
        self.repo = self.cfg["mlx_repo"]
        self.revision = self.cfg["mlx_revision"]
        self.adapter_path = None if adapter_path is None else str(check_adapter(adapter_path))
        if self.cfg["backend"] != "mlx-lm":
            raise NotImplementedError(f"backend {self.cfg['backend']} not wired yet")
        mx.random.seed(SEED)
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        if self.adapter_path is None:
            self.model, self.tokenizer = load(self.repo, revision=self.revision)
        else:
            self.model, self.tokenizer = load(self.repo, revision=self.revision,
                                              adapter_path=self.adapter_path)
        self.load_s = time.perf_counter() - t0
        self.sampler = make_sampler(temp=0.0)  # greedy

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.unload()

    def unload(self):
        del self.model, self.tokenizer
        gc.collect()
        mx.clear_cache()

    def format_prompt(self, system: str, user: str) -> str:
        if self.cfg["system_role"]:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        else:
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
        kwargs = {}
        if self.cfg["thinking"]["has_mode"]:
            kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )

    def generate(self, system: str, user: str, max_new_tokens: int) -> dict:
        prompt = self.format_prompt(system, user)
        # Template already contains any BOS token.
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        mx.random.seed(SEED)
        text, last, ttft = "", None, None
        t0 = time.perf_counter()
        for resp in stream_generate(
            self.model, self.tokenizer, ids, max_tokens=max_new_tokens, sampler=self.sampler
        ):
            if ttft is None:
                ttft = time.perf_counter() - t0  # prefill + first decoded token
            text += resp.text
            last = resp
        latency = time.perf_counter() - t0
        return {
            "raw_output": text,
            "prompt_tokens": len(ids),
            "gen_tokens": last.generation_tokens if last else 0,
            "finish_reason": last.finish_reason if last else None,
            "latency_s": round(latency, 4),
            "ttft_s": round(ttft, 4) if ttft is not None else None,
            "prefill_tps": round(last.prompt_tps, 2) if last else None,
            "decode_tps": round(last.generation_tps, 2) if last else None,
        }

    @staticmethod
    def peak_memory_gb() -> float:
        return round(mx.get_peak_memory() / 1e9, 3)
