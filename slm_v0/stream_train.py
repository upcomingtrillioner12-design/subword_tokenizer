#!/usr/bin/env python3
import arxiv
import tiktoken
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

CATEGORIES = ["physics", "physics.optics", "physics.quant-ph", "hep-th", "gr-qc"]
MAX_PAPERS = 2000
BATCH_SIZE = 4
SEQ_LEN = 512
DELAY = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")

def stream_arxiv(category="physics", max_results=2000):
    client = arxiv.Client(page_size=1000, delay_seconds=DELAY, num_retries=5)
    search = arxiv.Search(query=f"cat:{category}", sort_by=arxiv.SortCriterion.SubmittedDate, max_results=max_results)
    for result in client.results(search):
        text = f"Title: {result.title}\nAbstract: {result.summary}\n\n"
        tokens = enc.encode(text, allowed_special="all")
        if len(tokens) > 50:
            yield tokens

def make_batches(token_stream, batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
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
    def __init__(self, vocab_size=50257, d_model=256, n_layers=2, n_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(1024, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, batch_first=True), n_layers)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.head(self.transformer(self.embed(x) + self.pos_embed(pos)))

def train():
    model = TinyLM().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"Training on {DEVICE} | Streaming arXiv live...")
    for epoch in range(2):
        print(f"\n=== Epoch {epoch+1} ===")
        total_loss, steps = 0, 0
        for x, y in make_batches(stream_arxiv("physics", MAX_PAPERS)):
            optimizer.zero_grad()
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            if steps % 10 == 0:
                print(f"  Step {steps} | Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1} avg loss: {total_loss/steps:.4f}")
        torch.save(model.state_dict(), f"slm_v0/checkpoint_epoch{epoch+1}.pt")

if __name__ == "__main__":
    train()
