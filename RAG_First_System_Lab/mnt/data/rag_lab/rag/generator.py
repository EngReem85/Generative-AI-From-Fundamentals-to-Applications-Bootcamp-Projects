from __future__ import annotations

import re
from functools import lru_cache

import torch
from transformers import pipeline


class Generator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    def generate(self, prompt: str, max_new_tokens: int = 220) -> str:
        output = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        text = output[0]["generated_text"]
        # Some model versions can echo a trailing prompt marker.
        text = re.sub(r"\s*ANSWER:\s*$", "", text).strip()
        return text
