"""
On chip SLM classifier wrapper around llama.cpp.

We run a fine tuned SmolLM2-135M Instruct in Q4_K_M gguf form. On a Pi Zero W
(ARMv6, 512 MB) the Q4_K_M file is around 88 MB and reaches 4 to 6 tokens per
second, which is enough because the assistant only emits a single label token.

If llama_cpp is not available (e.g. during dev on a laptop) the wrapper falls
back to a feature thresholded rule classifier so the rest of the pipeline keeps
working. The rule classifier is also a useful lower bound baseline for the
paper.
"""

from __future__ import annotations

import os
from feature_extractor import to_prompt
from physics_sim import LABELS, TRIGGER_LABELS

DEFAULT_MODEL_PATH = os.environ.get(
    "SAFETY_SLM_PATH",
    "models/smollm2-135m-instruct-safety-q4_k_m.gguf",
)


class RuleFallback:
    """Hand crafted thresholds tuned on the synthetic distribution."""

    def classify(self, feats):
        ap = feats["acc_peak_g"]
        am = feats["acc_mean_g"]
        jp = feats["jerk_peak"]
        jm = feats["jerk_mean"]
        gs = feats["gyro_std_dps"]
        gp = feats["gyro_peak_dps"]
        # Postures
        if ap < 1.05 and gp < 5:
            return "sitting" if feats["az_std"] < 0.005 else "standing"
        # Sitting with shaking foot
        if am < 1.6 and gp < 400 and feats["ay_std"] + feats["az_std"] > 0.05:
            return "shake_leg"
        # Repeated kick: many high jerk events
        if jm > 100 and gs > 500 and ap > 12:
            return "repeated_kick"
        # Single hard kick
        if jp > 500 and gp > 1500 and am < 5 and gs < 500:
            return "hard_kick"
        # Soft kick
        if jp > 350 and 200 < gp < 1500 and am < 3.5:
            return "soft_kick"
        # Running has sustained high acc with periodic peaks
        if am > 8 and jm > 60 and ap > 10:
            return "running"
        # Walking: moderate periodic
        if 1.3 < am < 4 and gp < 700:
            return "walking"
        # Stomp: brief impact
        if jp > 250 and ap > 3 and am < 3:
            return "stomp"
        # Jumping: ballistic phase has near zero g
        if feats["acc_min_g"] < 0.4 and ap > 4:
            return "jumping"
        return "walking"


class SafetySLM:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, n_threads: int = 4):
        self._llm = None
        self._fallback = RuleFallback()
        try:
            from llama_cpp import Llama  # type: ignore

            if os.path.exists(model_path):
                self._llm = Llama(
                    model_path=model_path,
                    n_ctx=512,
                    n_threads=n_threads,
                    n_batch=64,
                    verbose=False,
                )
        except Exception:
            self._llm = None

    def classify(self, feats) -> str:
        if self._llm is None:
            return self._fallback.classify(feats)
        prompt = to_prompt(feats)
        out = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
        )
        text = out["choices"][0]["message"]["content"].strip().lower()
        for lbl in LABELS:
            if lbl in text:
                return lbl
        return self._fallback.classify(feats)

    def is_trigger(self, label: str) -> bool:
        return label in TRIGGER_LABELS
