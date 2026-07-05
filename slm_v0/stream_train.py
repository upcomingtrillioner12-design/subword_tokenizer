#!/usr/bin/env python3
import ast
import argparse
import json
import shutil
import time
import arxiv
import torch
import torch.nn as nn
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

    checkpoints_dir = ROOT / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

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
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            if steps % 10 == 0:
                print(f"  Step {steps} | Loss: {loss.item():.4f}")
            if save_every_steps > 0 and steps % save_every_steps == 0:
                step_ckpt_path = checkpoints_dir / f"{checkpoint_prefix}_epoch{epoch+1}_step{steps}.pt"
                torch.save(model.state_dict(), step_ckpt_path)
                print(f"  Saved step checkpoint: {step_ckpt_path}")
            if max_steps_per_epoch > 0 and steps >= max_steps_per_epoch:
                print(f"  Reached max_steps_per_epoch={max_steps_per_epoch}, stopping epoch early.")
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
        "tokenizer_model": str(tokenizer_model),
        "estimated_params": estimated_params,
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
    )
