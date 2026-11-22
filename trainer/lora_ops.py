"""Torch/PEFT operations: building columns, lateral blocks, EWC, perplexity.

Lateral connections as frozen rank blocks
-----------------------------------------
A LoRA column contributes dW = s * B @ A (s = alpha / r). Concatenating ranks,
    A_cat = [w_1 A_1 ; w_2 A_2 ; A_new],   B_cat = [B_1, B_2, B_new]
gives dW_cat = s * (w_1 B_1 A_1 + w_2 B_2 A_2 + B_new A_new) exactly - the
parents' knowledge is injected through *frozen* blocks while only the new block
trains. Because alpha = 2r is kept for every column, s = 2 everywhere, so the
parent terms are reproduced exactly. The result is a plain PEFT LoRA adapter,
so it exports to GGUF / Ollama like any other.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SCALE = 2.0  # alpha / r, kept constant across all columns


def lora_config(rank: int):
    from peft import LoraConfig

    return LoraConfig(r=rank, lora_alpha=int(SCALE * rank), lora_dropout=0.05, bias="none",
                      target_modules=TARGETS, task_type="CAUSAL_LM")


def load_base(model_name: str, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.config.use_cache = False
    return model, tok


def _adapter_tensors(adapter_dir: str | Path) -> dict[str, torch.Tensor]:
    p = Path(adapter_dir)
    st = p / "adapter_model.safetensors"
    if st.exists():
        from safetensors.torch import load_file

        return load_file(str(st))
    return torch.load(p / "adapter_model.bin", map_location="cpu")


def _module_key(name: str) -> str:
    """'...q_proj.lora_A.default.weight' or '...q_proj.lora_A.weight' -> '...q_proj'"""
    for tag in (".lora_A", ".lora_B"):
        if tag in name:
            return name.split(tag)[0].replace("base_model.model.", "")
    return name


def new_column(base, rank_own: int, parents: list[tuple[str, float, int]]):
    """parents: [(adapter_dir, weight, parent_rank)]. Returns (peft_model, frozen_rank)."""
    from peft import get_peft_model

    frozen = sum(r for _, _, r in parents)
    model = get_peft_model(base, lora_config(frozen + rank_own))
    if parents:
        params = dict(model.named_parameters())
        a_by_mod = {_module_key(n): p for n, p in params.items() if ".lora_A." in n}
        b_by_mod = {_module_key(n): p for n, p in params.items() if ".lora_B." in n}
        offset = 0
        with torch.no_grad():
            for pdir, w, pr in parents:
                t = _adapter_tensors(pdir)
                for k, v in t.items():
                    mod = _module_key(k)
                    if ".lora_A" in k and mod in a_by_mod:
                        a_by_mod[mod][offset:offset + pr, :].copy_(w * v.to(a_by_mod[mod].dtype))
                    elif ".lora_B" in k and mod in b_by_mod:
                        b_by_mod[mod][:, offset:offset + pr].copy_(v.to(b_by_mod[mod].dtype))
                offset += pr
            # the trainable block starts at zero contribution: B_new = 0 (A_new keeps its init)
            for mod, b in b_by_mod.items():
                b[:, frozen:].zero_()
    return model, frozen


def load_column(base, adapter_dir: str | Path):
    from peft import PeftModel

    return PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)


def freeze_lateral_blocks(model, frozen_rank: int) -> list:
    """Zero the gradients of the first `frozen_rank` rank rows/cols (parent blocks)."""
    hooks = []
    if frozen_rank <= 0:
        return hooks
    for n, p in model.named_parameters():
        if ".lora_A." in n:
            def hook_a(g, fr=frozen_rank):
                g = g.clone(); g[:fr, :] = 0; return g
            hooks.append(p.register_hook(hook_a))
        elif ".lora_B." in n:
            def hook_b(g, fr=frozen_rank):
                g = g.clone(); g[:, :fr] = 0; return g
            hooks.append(p.register_hook(hook_b))
    return hooks


def trainable_lora(model) -> dict[str, torch.nn.Parameter]:
    return {n: p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n}


# ------------------------------------------------------------------ data
def batches(tok, texts: list[str], max_len: int, bs: int, device: str):
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        yield {k: v.to(device) for k, v in enc.items()} | {"labels": labels.to(device)}


@torch.no_grad()
def perplexity(model, tok, texts: list[str], max_len: int = 512, bs: int = 4, device: str = "cuda") -> float:
    if not texts:
        return float("nan")
    model.eval()
    nll, ntok = 0.0, 0
    for b in batches(tok, texts, max_len, bs, device):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            out = model(**b)
        n = int((b["labels"][:, 1:] != -100).sum())
        nll += float(out.loss) * n
        ntok += n
    return math.exp(nll / max(ntok, 1))


# ------------------------------------------------------------------ EWC
def fisher_diagonal(model, tok, texts: list[str], max_len: int, device: str, n_batches: int = 32) -> dict:
    """Empirical Fisher diag on trainable LoRA params, normalised to mean 1."""
    params = trainable_lora(model)
    fisher = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params.items()}
    model.eval()
    k = 0
    for b in batches(tok, texts, max_len, 1, device):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            loss = model(**b).loss
        loss.backward()
        for n, p in params.items():
            if p.grad is not None:
                fisher[n] += p.grad.detach().float() ** 2
        k += 1
        if k >= n_batches:
            break
    model.zero_grad(set_to_none=True)
    mean = torch.cat([f.flatten() for f in fisher.values()]).mean().clamp_min(1e-12)
    return {n: (f / max(k, 1)) / mean for n, f in fisher.items()}


def save_ewc(path: str | Path, model, fisher: dict) -> None:
    star = {n: p.detach().float().cpu() for n, p in trainable_lora(model).items()}
    torch.save({"fisher": {n: f.cpu() for n, f in fisher.items()}, "star": star}, path)


def load_ewc(path: str | Path, device: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    d = torch.load(p, map_location=device)
    return {"fisher": {k: v.to(device) for k, v in d["fisher"].items()},
            "star": {k: v.to(device) for k, v in d["star"].items()}}


def ewc_penalty(model, ewc: dict | None, lam: float) -> torch.Tensor:
    """sum_i (lambda/2) F_i (theta_i - theta*_i)^2 over LoRA params."""
    if not ewc or lam <= 0:
        return torch.zeros((), device=next(model.parameters()).device)
    total = 0.0
    for n, p in trainable_lora(model).items():
        if n in ewc["fisher"]:
            total = total + (ewc["fisher"][n] * (p.float() - ewc["star"][n]) ** 2).sum()
    return 0.5 * lam * total


# ------------------------------------------------------------------ training loop
def train_lora(model, tok, texts: list[str], *, ewc: dict | None, lam: float, frozen_rank: int,
               epochs: int, lr: float, bs: int, grad_accum: int, max_len: int, device: str,
               log=print) -> dict:
    hooks = freeze_lateral_blocks(model, frozen_rank)
    params = list(trainable_lora(model).values())
    for p in params:
        p.data = p.data.float()  # keep LoRA weights in fp32 for stable fp16 AMP
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    step, hist = 0, []
    for ep in range(epochs):
        import random
        random.shuffle(texts)
        for i, b in enumerate(batches(tok, texts, max_len, bs, device)):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                ce = model(**b).loss
            pen = ewc_penalty(model, ewc, lam)
            loss = (ce.float() + pen) / grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
            hist.append((float(ce), float(pen)))
        ce_m = sum(h[0] for h in hist[-50:]) / max(1, len(hist[-50:]))
        log(f"  epoch {ep + 1}/{epochs}  ce={ce_m:.3f}  ewc={hist[-1][1]:.4f}")
    opt.zero_grad(set_to_none=True)
    for h in hooks:
        h.remove()
    return {"steps": step, "final_ce": hist[-1][0] if hist else None}
