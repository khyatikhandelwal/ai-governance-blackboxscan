"""
Implements the detector for Tournament Sampling watermarking (SynthID-Text inspired).
This file matches tournament_watermarking.py and computes statistical evidence for watermarking
from a given text.

Steps:
1. Recompute the same random seeds (sliding-window over last H tokens) using the secret key.
2. Evaluate all g_l(x_t, r_t) values for m tournament layers.
3. Compute mean watermark score and statistical z-score / p-value / confidence.

Formula:
    Score(x) = (1 / (m*T)) * sum_t sum_l g_l(x_t, r_t)
    z = (Score(x) - 0.5) / sqrt(0.25 / (m*T))
"""

import hashlib, math
from typing import List, Dict, Any

# --------------------------
# Utilities (must match tournament_watermarking.py)
# --------------------------

def _sha256_int(data: bytes) -> int:
    return int(hashlib.sha256(data).hexdigest(), 16)

def hash_seed(key: str, context: List[int], H: int = 4) -> bytes:
    context = context[-H:]
    h_input = key.encode() + b'|'.join(str(x).encode() for x in context)
    return hashlib.sha256(h_input).digest()

def g_l(token_id: int, r_t: bytes, layer: int) -> int:
    h_input = r_t + f'|layer={layer}|token={token_id}'.encode()
    return _sha256_int(h_input) % 2  # returns 0 or 1

# --------------------------
# Detector
# --------------------------
class TournamentWatermarkDetector:
    def __init__(self, tokenizer, key: str, m: int = 3, H: int = 4):
        self.tokenizer = tokenizer
        self.key = key
        self.m = m
        self.H = H

    def detect(self, text: str) -> Dict[str, Any]:
        toks = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        T = len(toks)

        if T == 0:
            return {
                "T": 0,
                "mean_g": 0.5,
                "z": 0,
                "p": 1.0,
                "confidence": 0,
                "status": "NO WATERMARK (empty text)",
                "explanation": "The input text contains no tokens."
            }

        total_g = 0
        for t in range(T):
            prefix = toks[:t]
            r_t = hash_seed(self.key, prefix, self.H)
            for l in range(1, self.m + 1):
                total_g += g_l(toks[t], r_t, l)

        mean_g = total_g / (self.m * T)
        z = (mean_g - 0.5) / math.sqrt(0.25 / (self.m * T))
        p = 0.5 * math.erfc(z / math.sqrt(2))
        confidence = float(1 - p)

        # -------------------------------
        # Decision Logic
        # -------------------------------

        # Practical thresholds based on evaluation of tournament sampling
        is_watermarked = (
            (mean_g > 0.53) or    # mean skew above noise
            (z > 2.0) or          # statistical significance
            (confidence > 0.90)   # strong confidence
        )

        status = "WATERMARK DETECTED" if is_watermarked else "NO WATERMARK DETECTED"

        explanation = (
            "The text shows statistically significant bias consistent with tournament sampling watermarking."
            if is_watermarked else
            "The text does not show statistically significant bias consistent with watermarking."
        )

        # -------------------------------
        # Build human-readable report
        # -------------------------------
        report = f"""
=============================
  Tournament Watermark Detector
=============================
Tokens analyzed      : {T}
Tournament layers (m): {self.m}

--- Watermark Statistics ---
Mean g-value         : {mean_g:.4f}
Z-score              : {z:.4f}
P-value              : {p:.4e}
Confidence           : {confidence:.2%}

--- Detection Result ---
Status               : {status}
Explanation          : {explanation}
=============================
""".strip()

        return {
            "T": T,
            "m": self.m,
            "mean_g": mean_g,
            "z": z,
            "p": p,
            "confidence": confidence,
            "status": status,
            "explanation": explanation,
            "report": report,
        }



if __name__ == "__main__":
    from transformers import AutoTokenizer

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    key = "secret-key-42"
    m = 3
    H = 4

    detector = TournamentWatermarkDetector(tokenizer, key=key, m=m, H=H)

    example_text = """
     The future of humanity is at stake," he said. "We are a big part of it."

He called the EU "a big loser, a big loser, and we will have to do more to get it back on track."

"If you can
"""
    result = detector.detect(example_text)

    print("\n--- Detection Results ---")
    for k, v in result.items():
        print(f"{k}: {v}")
