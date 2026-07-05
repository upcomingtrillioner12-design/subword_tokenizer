# Subword Tokenizer + Prototype Training Architecture

```mermaid
flowchart LR
    A[Live arXiv Stream] --> B[stream_train.py]
    B --> C[Retry/Backoff Network Layer]
    C --> D[Rust Tokenizer CLI\\nbpe-tokenizer tokenize]
    D --> E[Token IDs]
    E --> F[TinyLM Training Loop\\nMPS/CUDA/CPU]
    F --> G[Step Checkpoints]
    F --> H[Epoch Checkpoint + Summary]
    H --> I[evaluate_checkpoints.py]
    I --> J[best_checkpoint_*.json]

    K[Tokenizer Training CLI\\ntrain / expand / prepare] --> L[model_32k.json]
    L --> D
```

## Notes

- Training tokenization uses the project tokenizer model (`model_32k.json`) via Rust CLI.
- Long runs include retry/backoff to tolerate transient arXiv read timeouts.
- Evaluation ranks checkpoints by average eval loss and picks best automatically.
