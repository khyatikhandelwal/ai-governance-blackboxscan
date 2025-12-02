"""
Implements a Tournament Sampling watermark generator compatible with HuggingFace models.
This method reproduces the SynthID-Text core idea: per-step random seed generation (sliding-window)
combined with tournament-style selection among sampled candidate tokens based on multiple random
watermarking functions g_l.

Functions:
- TournamentSampler: class to sample a token with watermark embedding.
- hash_seed: derive random seed from last H tokens + secret key.
- g_l: random watermarking functions assigning scores to tokens.

Usage:
- Call TournamentSampler.sample_next_token(probs, context_ids) instead of torch.multinomial.
"""

import torch, hashlib, math, random
from typing import List, Dict, Any

# --------------------------
# Utilities: seed and scoring
# --------------------------

def _sha256_int(data: bytes) -> int:
    return int(hashlib.sha256(data).hexdigest(), 16)

def hash_seed(key: str, context: List[int], H: int = 4) -> bytes:
    context = context[-H:]
    h_input = key.encode() + b'|'.join(str(x).encode() for x in context)
    return hashlib.sha256(h_input).digest()

def g_l(token_id: int, r_t: bytes, layer: int) -> int:
    h_input = r_t + f'|layer={layer}|token={token_id}'.encode()
    return _sha256_int(h_input) % 2  # 0 or 1

# --------------------------
# Tournament Sampler
# --------------------------
class TournamentSampler:
    def __init__(self, key: str, m: int = 3, H: int = 4):
        """Tournament Sampling watermark generator.

        Args:
            key: secret watermark key
            m: number of tournament layers
            H: sliding-window context length for seed generation
        """
        self.key = key
        self.m = m
        self.H = H

    def tournament_sample(self, probs: torch.Tensor, context: List[int], candidate_ids=None) -> int:
        """
        probs: probabilities (len = K)
        candidate_ids: actual vocab ids corresponding to probs
        """
        if candidate_ids is None:
            candidate_ids = list(range(len(probs)))

        vocab_candidates = candidate_ids
        r_t = hash_seed(self.key, context, self.H)
        M = 2 * self.m

        # Sample M candidates using candidate_ids
        sampled_idx = torch.multinomial(probs, num_samples=M, replacement=True).tolist()
        candidates = [vocab_candidates[i] for i in sampled_idx]   # real token ids

        # tournament rounds
        for layer in range(1, self.m + 1):
            random.shuffle(candidates)
            next_round = []
            for i in range(0, len(candidates), 2):
                if i + 1 >= len(candidates):
                    next_round.append(candidates[i])
                    break
                c1, c2 = candidates[i], candidates[i + 1]
                g1 = g_l(c1, r_t, layer)
                g2 = g_l(c2, r_t, layer)
                winner = c1 if g1 >= g2 else c2
                next_round.append(winner)
            candidates = next_round
            if len(candidates) == 1:
                break

        return candidates[0]  # real vocab token id


    def generate_with_tournament(self, model, tokenizer, prompt: str, max_new_tokens: int = 50, top_k: int = 50, temperature: float = 1.0, device: str = "cpu") -> str:
        model.eval()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generated = input_ids[0].tolist()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(torch.tensor([generated]).to(device))
                logits = outputs.logits[:, -1, :] / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
                # restrict to top_k
                if top_k is not None:
                    topk_probs, topk_indices = torch.topk(probs, top_k)
                    topk_probs = topk_probs / topk_probs.sum()

                    token_id = self.tournament_sample(
                        probs=topk_probs,
                        context=generated,
                        candidate_ids=topk_indices.tolist()  # <-- REAL vocab ids
                    )
                else:
                    token_id = self.tournament_sample(
                        probs=probs,
                        context=generated,               # full vocab, ids = [0..V-1]
                        candidate_ids=None
                    )



            generated.append(token_id)
            if token_id == tokenizer.eos_token_id:
                break
        return tokenizer.decode(generated, skip_special_tokens=True)


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from detecting_text import TournamentWatermarkDetector  # adjust import to your path

    model_name = "gpt2"
    key = "secret-key-42"
    m = 3
    H = 4

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    sampler = TournamentSampler(key=key, m=m, H=H)
    text = sampler.generate_with_tournament(
        model,
        tokenizer,
        "The future of humanity is",
        top_k=50,
        temperature=0.7,
        device=device
    )

    print("\n--- Generated text (tournament watermark) ---\n", text)

    detector = TournamentWatermarkDetector(tokenizer, key=key, m=m, H=H)
    result = detector.detect(text)

    print("\n--- Detection Results ---")
    for k, v in result.items():
        print(f"{k}: {v}")

