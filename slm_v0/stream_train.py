#!/usr/bin/env python3
import ast
import argparse
import json
import shutil
import time
from typing import Optional
import arxiv
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import subprocess
import requests

CATEGORIES = ["physics", "physics.optics", "physics.quant-ph", "hep-th", "gr-qc"]
DEFAULT_MAX_PAPERS = 2000
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEQ_LEN = 512
DEFAULT_DELAY = 3
ROOT = Path(__file__).resolve().parent
if (ROOT / "subword_tokenizer").exists():
    TOKENIZER_PROJECT_DIR = ROOT / "subword_tokenizer"
elif (ROOT.parent / "Cargo.toml").exists():
    TOKENIZER_PROJECT_DIR = ROOT.parent
else:
    TOKENIZER_PROJECT_DIR = ROOT / "subword_tokenizer"
TOKENIZER_MODEL = TOKENIZER_PROJECT_DIR / "model_32k.json"
TOKENIZER_ACTIVE_MODEL = TOKENIZER_PROJECT_DIR / "model.json"
TOKENIZER_BIN = TOKENIZER_PROJECT_DIR / "target" / "release" / "bpe-tokenizer"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

EOS_TOKEN_ID = 2
DEFAULT_GENERATION_PROMPTS = [
    "In quantum mechanics, the wave function describes",
    "The uncertainty principle states that",
    "Entropy in thermodynamics is",
    "In general relativity, spacetime curvature",
]


def _resolve_vocab_size(model_path: Path) -> int:
    import json

    with model_path.open("r", encoding="utf-8") as f:
        model = json.load(f)

    vocab = model.get("vocab", {}) if isinstance(model, dict) else {}
    if isinstance(vocab, dict):
        return len(vocab)
    if isinstance(vocab, list):
        return len(vocab)
    return 32000


def _activate_tokenizer_model(model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
    shutil.copy2(model_path, TOKENIZER_ACTIVE_MODEL)


def _tokenize_with_our_model(text: str) -> list[int]:
    if TOKENIZER_BIN.exists():
        cmd = [str(TOKENIZER_BIN), "tokenize", text]
        cwd = str(TOKENIZER_PROJECT_DIR)
    else:
        cmd = ["cargo", "run", "--release", "--manifest-path", str(TOKENIZER_PROJECT_DIR / "Cargo.toml"), "--", "tokenize", text]
        cwd = str(ROOT)

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Tokenizer failed: {result.stderr.strip() or result.stdout.strip()}")

    for line in result.stdout.splitlines():
        if line.startswith("IDs:"):
            return ast.literal_eval(line.replace("IDs:", "").strip())
    return []

def _build_arxiv_query(category: str) -> str:
    if any(op in category for op in ["cat:", "all:", "AND", "OR"]):
        return category
    if "." in category:
        return f"cat:{category}"
    return f"all:{category}"


def stream_arxiv(
    category="physics",
    max_results=2000,
    delay_seconds=3,
    min_tokens=12,
    max_retries=20,
    retry_backoff_seconds=5,
):
    """Stream tokenized arXiv entries with retry/backoff so long runs do not crash on transient network timeouts."""
    query = _build_arxiv_query(category)
    retries = 0

    while True:
        try:
            client = arxiv.Client(page_size=1000, delay_seconds=delay_seconds, num_retries=5)
            search = arxiv.Search(query=query, sort_by=arxiv.SortCriterion.SubmittedDate, max_results=max_results)

            for result in client.results(search):
                text = f"Title: {result.title}\nAbstract: {result.summary}\n\n"
                tokens = _tokenize_with_our_model(text)
                if len(tokens) >= min_tokens:
                    yield tokens

            return
        except (requests.exceptions.RequestException, TimeoutError) as e:
            retries += 1
            if retries > max_retries:
                print(f"[stream_arxiv] Max retries exceeded ({max_retries}). Last error: {e}")
                return
            sleep_s = retry_backoff_seconds * min(retries, 12)
            print(f"[stream_arxiv] Network timeout/error ({e}). Retry {retries}/{max_retries} in {sleep_s}s...")
            time.sleep(sleep_s)
        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"[stream_arxiv] Unexpected error; stopping stream after {max_retries} retries. Last error: {e}")
                return
            sleep_s = retry_backoff_seconds * min(retries, 12)
            print(f"[stream_arxiv] Unexpected error ({e}). Retry {retries}/{max_retries} in {sleep_s}s...")
            time.sleep(sleep_s)

def make_batches(token_stream, batch_size=DEFAULT_BATCH_SIZE, seq_len=DEFAULT_SEQ_LEN):
    buffer = []
    for tokens in token_stream:
        buffer.extend(tokens)
        while len(buffer) >= batch_size * seq_len + 1:
            batch = buffer[:batch_size * seq_len + 1]
            buffer = buffer[batch_size * seq_len:]
            x = torch.tensor(batch[:-1], dtype=torch.long).view(batch_size, seq_len)
            y = torch.tensor(batch[1:], dtype=torch.long).view(batch_size, seq_len)
            yield x.to(DEVICE), y.to(DEVICE)

class TinyLM(nn.Module):
    def __init__(self, vocab_size=32000, d_model=256, n_layers=2, n_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(1024, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True), n_layers)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.head(self.transformer(self.embed(x) + self.pos_embed(pos)))

def _estimate_param_count(vocab_size: int, d_model: int, n_layers: int) -> int:
    embed = vocab_size * d_model
    lm_head = d_model * vocab_size
    per_layer = 12 * (d_model ** 2)
    return embed + lm_head + (n_layers * per_layer)


@torch.no_grad()
def _estimate_generation_tokens(
    model: nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int = EOS_TOKEN_ID,
) -> int:
    if not prompt_ids:
        prompt_ids = [1]
    x = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    generated = x
    count = 0
    for _ in range(max_new_tokens):
        logits = model(generated)
        next_token = torch.argmax(logits[0, -1, :], dim=-1).item()
        if next_token == eos_token_id:
            break
        next_token_t = torch.tensor([[next_token]], dtype=torch.long, device=DEVICE)
        generated = torch.cat([generated, next_token_t], dim=1)
        count += 1
    return count


@torch.no_grad()
def _evaluate_generation_health(
    model: nn.Module,
    prompts: list[str],
    max_new_tokens: int,
    eos_token_id: int = EOS_TOKEN_ID,
) -> dict:
    model_was_training = model.training
    model.eval()

    token_counts = []
    eos_probs = []

    for prompt in prompts:
        prompt_ids = _tokenize_with_our_model(prompt)
        if not prompt_ids:
            prompt_ids = [1]
        x = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

        logits = model(x)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        eos_probs.append(float(probs[eos_token_id].item()))

        token_counts.append(
            _estimate_generation_tokens(
                model=model,
                prompt_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )
        )

    if model_was_training:
        model.train()

    avg_gen_tokens = float(sum(token_counts) / max(len(token_counts), 1))
    max_gen_tokens = int(max(token_counts) if token_counts else 0)
    avg_eos_prob = float(sum(eos_probs) / max(len(eos_probs), 1))

    return {
        "avg_gen_tokens": avg_gen_tokens,
        "max_gen_tokens": max_gen_tokens,
        "avg_eos_prob": avg_eos_prob,
        "token_counts": token_counts,
        "eos_probs": eos_probs,
    }


def train(
    tokenizer_model: Path,
    category: str,
    max_papers: int,
    batch_size: int,
    seq_len: int,
    epochs: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    lr: float,
    max_steps_per_epoch: int,
    delay_seconds: int,
    min_tokens: int,
    stream_max_retries: int,
    stream_retry_backoff_seconds: int,
    save_every_steps: int,
    checkpoint_prefix: str,
    checkpoints_dir: Path,
    eos_token_id: int,
    non_eos_threshold: float,
    eos_penalty_weight: float,
    generation_eval_every_steps: int,
    generation_max_new_tokens: int,
    early_stop_avg_gen_tokens: float,
    early_stop_patience: int,
):
    _activate_tokenizer_model(tokenizer_model)
    vocab_size = _resolve_vocab_size(tokenizer_model)
    model = TinyLM(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    estimated_params = _estimate_param_count(vocab_size, d_model, n_layers)
    print(f"Training on {DEVICE} | Tokenizer: {tokenizer_model}")
    print(
        f"Config => category={category}, max_papers={max_papers}, batch_size={batch_size}, "
        f"seq_len={seq_len}, epochs={epochs}, min_tokens={min_tokens}, "
        f"stream_max_retries={stream_max_retries}, stream_retry_backoff_seconds={stream_retry_backoff_seconds}, "
        f"save_every_steps={save_every_steps}"
    )
    print(f"Model => vocab_size={vocab_size}, d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, approx_params={estimated_params:,}")

    checkpoints_dir = Path(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    generation_history = []
    global_step = 0
    consecutive_generation_hits = 0

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1} ===")
        total_loss, steps = 0, 0
        token_stream = stream_arxiv(
            category=category,
            max_results=max_papers,
            delay_seconds=delay_seconds,
            min_tokens=min_tokens,
            max_retries=stream_max_retries,
            retry_backoff_seconds=stream_retry_backoff_seconds,
        )
        for x, y in make_batches(token_stream, batch_size=batch_size, seq_len=seq_len):
            optimizer.zero_grad()
            logits = model(x)

            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            probs = F.softmax(logits, dim=-1)
            eos_probs = probs[..., eos_token_id]
            eos_max_prob = max(0.0, 1.0 - non_eos_threshold)
            eos_penalty = torch.relu(eos_probs - eos_max_prob).pow(2).mean()

            loss = ce_loss + (eos_penalty_weight * eos_penalty)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            global_step += 1
            if steps % 10 == 0:
                print(
                    f"  Step {steps} | Loss: {loss.item():.4f} "
                    f"(ce={ce_loss.item():.4f}, eos_pen={eos_penalty.item():.6f}, eos_mean={eos_probs.mean().item():.4f})"
                )

            if generation_eval_every_steps > 0 and global_step % generation_eval_every_steps == 0:
                health = _evaluate_generation_health(
                    model=model,
                    prompts=DEFAULT_GENERATION_PROMPTS,
                    max_new_tokens=generation_max_new_tokens,
                    eos_token_id=eos_token_id,
                )
                generation_history.append({"global_step": global_step, **health})
                print(
                    f"  [gen-eval] step={global_step} avg_gen_tokens={health['avg_gen_tokens']:.2f} "
                    f"max_gen_tokens={health['max_gen_tokens']} avg_eos_prob={health['avg_eos_prob']:.4f}"
                )

                if health["avg_gen_tokens"] >= early_stop_avg_gen_tokens:
                    consecutive_generation_hits += 1
                    print(
                        f"  [gen-eval] improvement hit {consecutive_generation_hits}/{early_stop_patience} "
                        f"(target avg_gen_tokens >= {early_stop_avg_gen_tokens})"
                    )
                else:
                    consecutive_generation_hits = 0

                if early_stop_patience > 0 and consecutive_generation_hits >= early_stop_patience:
                    print(
                        f"  [early-stop] generation target reached for {early_stop_patience} evals. "
                        "Stopping training early."
                    )
                    break

            if save_every_steps > 0 and steps % save_every_steps == 0:
                step_ckpt_path = checkpoints_dir / f"{checkpoint_prefix}_epoch{epoch+1}_step{steps}.pt"
                torch.save(model.state_dict(), step_ckpt_path)
                print(f"  Saved step checkpoint: {step_ckpt_path}")
            if max_steps_per_epoch > 0 and steps >= max_steps_per_epoch:
                print(f"  Reached max_steps_per_epoch={max_steps_per_epoch}, stopping epoch early.")
                break

        if early_stop_patience > 0 and consecutive_generation_hits >= early_stop_patience:
            print("Early stopping completed at epoch boundary.")
            ckpt_path = checkpoints_dir / f"{checkpoint_prefix}_epoch{epoch+1}_earlystop.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved early-stop checkpoint: {ckpt_path}")
            break

        if steps == 0:
            print("No batches produced. Increase max_papers or reduce seq_len/batch_size.")
            continue
        print(f"Epoch {epoch+1} avg loss: {total_loss/steps:.4f}")
        ckpt_path = checkpoints_dir / f"{checkpoint_prefix}_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    summary = {
        "device": DEVICE,
        "category": category,
        "max_papers": max_papers,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "epochs": epochs,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "lr": lr,
        "max_steps_per_epoch": max_steps_per_epoch,
        "min_tokens": min_tokens,
        "stream_max_retries": stream_max_retries,
        "stream_retry_backoff_seconds": stream_retry_backoff_seconds,
        "save_every_steps": save_every_steps,
        "checkpoints_dir": str(checkpoints_dir),
        "tokenizer_model": str(tokenizer_model),
        "estimated_params": estimated_params,
        "generation_aware": {
            "eos_token_id": eos_token_id,
            "non_eos_threshold": non_eos_threshold,
            "eos_penalty_weight": eos_penalty_weight,
            "generation_eval_every_steps": generation_eval_every_steps,
            "generation_max_new_tokens": generation_max_new_tokens,
            "early_stop_avg_gen_tokens": early_stop_avg_gen_tokens,
            "early_stop_patience": early_stop_patience,
        },
        "generation_history": generation_history,
    }
    summary_path = checkpoints_dir / f"{checkpoint_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Run summary: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a lightweight arXiv-streaming LM with project tokenizer.")
    parser.add_argument("--mode", choices=["prototype", "standard"], default="prototype")
    parser.add_argument("--tokenizer-model", default=str(TOKENIZER_MODEL))
    parser.add_argument("--category", default="physics")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--delay-seconds", type=int, default=DEFAULT_DELAY)
    parser.add_argument("--min-tokens", type=int, default=None)
    parser.add_argument("--stream-max-retries", type=int, default=20)
    parser.add_argument("--stream-retry-backoff-seconds", type=int, default=5)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--checkpoint-prefix", default=None)
    parser.add_argument("--checkpoints-dir", default=None)

    # Generation-aware loss and monitoring
    parser.add_argument("--eos-token-id", type=int, default=EOS_TOKEN_ID)
    parser.add_argument("--non-eos-threshold", type=float, default=0.20,
                        help="Target minimum non-EOS probability (enforces P(non-EOS) >= threshold)")
    parser.add_argument("--eos-penalty-weight", type=float, default=1.5,
                        help="Weight for EOS overconfidence penalty term")
    parser.add_argument("--generation-eval-every-steps", type=int, default=100,
                        help="Run generation health evaluation every N global steps")
    parser.add_argument("--generation-max-new-tokens", type=int, default=32)
    parser.add_argument("--early-stop-avg-gen-tokens", type=float, default=10.0,
                        help="Early stop when avg generated tokens reaches this target")
    parser.add_argument("--early-stop-patience", type=int, default=2,
                        help="Consecutive generation eval hits required for early stop")
    return parser.parse_args()


def _defaults_for_mode(mode: str):
    if mode == "prototype":
        return {
            "max_papers": 300,
            "batch_size": 2,
            "seq_len": 128,
            "epochs": 1,
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "max_steps_per_epoch": 50,
            "min_tokens": 12,
            "checkpoint_prefix": "prototype",
            "checkpoints_dir": str(ROOT / "checkpoints"),
        }
    return {
        "max_papers": DEFAULT_MAX_PAPERS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "seq_len": DEFAULT_SEQ_LEN,
        "epochs": 2,
        "d_model": 256,
        "n_layers": 2,
        "n_heads": 4,
        "max_steps_per_epoch": 0,
        "min_tokens": 50,
        "checkpoint_prefix": "standard",
        "checkpoints_dir": str(ROOT / "checkpoints"),
    }

if __name__ == "__main__":
    args = parse_args()
    defaults = _defaults_for_mode(args.mode)

    train(
        tokenizer_model=Path(args.tokenizer_model),
        category=args.category,
        max_papers=args.max_papers if args.max_papers is not None else defaults["max_papers"],
        batch_size=args.batch_size if args.batch_size is not None else defaults["batch_size"],
        seq_len=args.seq_len if args.seq_len is not None else defaults["seq_len"],
        epochs=args.epochs if args.epochs is not None else defaults["epochs"],
        d_model=args.d_model if args.d_model is not None else defaults["d_model"],
        n_layers=args.n_layers if args.n_layers is not None else defaults["n_layers"],
        n_heads=args.n_heads if args.n_heads is not None else defaults["n_heads"],
        lr=args.lr,
        max_steps_per_epoch=args.max_steps_per_epoch if args.max_steps_per_epoch is not None else defaults["max_steps_per_epoch"],
        delay_seconds=args.delay_seconds,
        min_tokens=args.min_tokens if args.min_tokens is not None else defaults["min_tokens"],
        stream_max_retries=args.stream_max_retries,
        stream_retry_backoff_seconds=args.stream_retry_backoff_seconds,
        save_every_steps=args.save_every_steps,
        checkpoint_prefix=args.checkpoint_prefix if args.checkpoint_prefix else defaults["checkpoint_prefix"],
        checkpoints_dir=Path(args.checkpoints_dir) if args.checkpoints_dir else Path(defaults["checkpoints_dir"]),
        eos_token_id=args.eos_token_id,
        non_eos_threshold=args.non_eos_threshold,
        eos_penalty_weight=args.eos_penalty_weight,
        generation_eval_every_steps=args.generation_eval_every_steps,
        generation_max_new_tokens=args.generation_max_new_tokens,
        early_stop_avg_gen_tokens=args.early_stop_avg_gen_tokens,
        early_stop_patience=args.early_stop_patience,
    )
