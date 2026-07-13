# Physics Research Assistant SLM - Complete Project Roadmap

**Final Goal:** Build a Physics Research Assistant using a 3B–7B parameter SLM, fine-tuned with LoRA on physics papers, combined with RAG and tool-using agents.

---

## Executive Summary

This document outlines the complete journey from current state to production-ready Physics Research Assistant. The project is structured in 6 phases over 8–12 weeks, with clear milestones, dependencies, and success criteria.

---

## Current Project Status (as of June 22, 2026)

### ✅ Completed
- Subword tokenizer (32K vocab, BPE) — **FROZEN**
- CLI for tokenization/detokenization
- Tokenizer validation (32K vs 50K trade-off analysis)
- Updated project documentation with authors and references

### ⏳ Not Started
- SLM model training pipeline
- LoRA fine-tuning setup
- RAG (Retrieval-Augmented Generation) integration
- Tool-using agent framework
- Physics paper dataset preparation

### 🔧 Current Tech Stack
- **Tokenizer**: Rust (subword-tokenizer) + 32K vocab
- **Environment**: Python 3.10+ (venv at `/Users/jdsingh/slm_v0/venv`)
- **Hardware**: Apple Silicon Mac (MPS support)

---

## Phase 1: Environment & Data Pipeline (Week 1–2)

### Goal
Prepare development environment, organize physics papers dataset, and build data preprocessing pipeline.

### 1.1 Python Environment Setup
**Framework/Software:**
- Python 3.10+
- PyTorch (with MPS support for Mac)
- Hugging Face Transformers, Datasets, Accelerate
- Pandas, NumPy

**Steps:**
1. Activate existing venv:
   ```bash
   source /Users/jdsingh/slm_v0/venv/bin/activate
   ```

2. Install core ML packages:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install transformers datasets accelerate peft pydantic loguru
   ```

3. Verify PyTorch MPS support:
   ```python
   import torch
   print(torch.backends.mps.is_available())  # Should be True
   ```

**Expected Outcome:** Reproducible Python environment with all dependencies.

---

### 1.2 Physics Papers Dataset Acquisition
**Sources:**
- arXiv API (physics category)
- OpenAlex (academic papers)
- Local corpus if available

**Steps:**
1. Download arXiv physics papers (10K–50K papers recommended for first run):
   ```bash
   pip install arxiv
   python scripts/download_arxiv.py --category physics --limit 10000
   ```

2. Organize into directory:
   ```
   /Users/jdsingh/slm_v0/data/
   ├── physics_papers/
   │   ├── raw/
   │   └── processed/
   ```

**Expected Outcome:** ~1–5 GB of physics papers in JSON/text format.

---

### 1.3 Data Preprocessing Pipeline
**Framework/Software:**
- Python (Pandas, NumPy, regex)
- Custom ETL scripts

**Steps:**
1. Create preprocessing script:
   ```python
   # scripts/preprocess_physics.py
   - Extract abstract + intro + methodology from papers
   - Remove noise, duplicates, non-English text
   - Normalize whitespace
   - Split long papers into chunks (max 4K words)
   - Output: clean JSONL (1 doc per line)
   ```

2. Run pipeline:
   ```bash
   python scripts/preprocess_physics.py \
       --input-dir data/physics_papers/raw \
       --output-dir data/physics_papers/processed \
       --chunk-size 4000
   ```

3. Statistics check:
   ```bash
   wc -l data/physics_papers/processed/corpus.jsonl
   # Expected: 50K–200K documents
   ```

**Expected Outcome:** Clean, deduplicated physics corpus ready for tokenization.

---

### 1.4 Tokenize Dataset (Using Existing Tokenizer)
**Steps:**
1. Freeze tokenizer at 32K vocab:
   ```bash
   cd /Users/jdsingh/slm_v0/subword_tokenizer
   cargo run --release -- train data/physics_papers/processed/corpus.txt 32000 --output tokenizer_32k.json
   ```

2. Create tokenization script:
   ```python
   # scripts/tokenize_corpus.py
   - Load tokenizer_32k.json
   - Tokenize all documents
   - Split into train/val/test (98%/1%/1%)
   - Pack into fixed 512-token sequences
   - Save as .bin or .npy files
   ```

3. Run:
   ```bash
   python scripts/tokenize_corpus.py \
       --corpus data/physics_papers/processed/corpus.jsonl \
       --tokenizer /Users/jdsingh/slm_v0/subword_tokenizer/tokenizer_32k.json \
       --seq-len 512 \
       --output data/physics_papers/tokenized/
   ```

**Expected Outcome:** 
- Train/val/test token sequences ready for SLM training
- ~300M–1B tokens total (depending on dataset size)

---

## Phase 2: Base SLM Training (Week 3–5)

### Goal
Train a 3B–7B parameter decoder-only SLM from scratch on physics papers.

### 2.1 Model Architecture & Configuration
**Framework/Software:**
- PyTorch
- Hugging Face Transformers (or custom implementation)

**Recommended Config (3B-7B range):**

```yaml
# config/base_slm_config.yaml
model_name: "physics_slm_base_3b"
architecture: "GPT2" or "LLaMA-style"

# Architecture details
hidden_size: 2048         # or 3072 for 7B
num_hidden_layers: 24     # or 32 for 7B
num_attention_heads: 16   # or 24 for 7B
intermediate_size: 8192   # 4 * hidden_size
max_seq_length: 512
vocab_size: 32000

# Training
batch_size: 32
gradient_accumulation_steps: 4
learning_rate: 5e-4
warmup_steps: 1000
total_training_steps: 100000
save_every: 5000
eval_every: 1000

# Hardware
device: "mps"
dtype: "float16"  # or "bfloat16"
max_memory_per_gpu: "16GB"
```

**Steps:**
1. Initialize model from scratch:
   ```python
   from transformers import AutoConfig, AutoModelForCausalLM
   
   config = AutoConfig.from_pretrained("gpt2", 
       vocab_size=32000, 
       hidden_size=2048,
       num_hidden_layers=24)
   model = AutoModelForCausalLM.from_config(config)
   ```

2. Create training script using Accelerate:
   ```bash
   pip install accelerate
   accelerate config  # Choose device, precision, etc.
   ```

**Expected Outcome:** Model checkpoint at iteration 0.

---

### 2.2 Training Loop
**Framework:** Hugging Face Transformers + Accelerate

**Steps:**
1. Build training pipeline:
   ```python
   # scripts/train_base_slm.py
   from accelerate import Accelerator
   from torch.utils.data import DataLoader
   from transformers import AdamW, get_cosine_schedule_with_warmup
   
   accelerator = Accelerator()
   model, optimizer, train_dataloader, scheduler = accelerator.prepare(...)
   
   for epoch in range(num_epochs):
       for batch_idx, batch in enumerate(train_dataloader):
           outputs = model(**batch)
           loss = outputs.loss
           accelerator.backward(loss)
           optimizer.step()
           scheduler.step()
           optimizer.zero_grad()
           
           # Checkpointing, logging
   ```

2. Launch training:
   ```bash
   accelerate launch scripts/train_base_slm.py \
       --config-path config/base_slm_config.yaml \
       --data-dir data/physics_papers/tokenized/
   ```

3. Monitor:
   - Loss curves (train/val)
   - Perplexity
   - Memory usage
   - Training speed (tokens/sec)

**Expected Outcome:** 
- Trained base SLM checkpoint (~3B–7B params)
- Validation perplexity <20 (typical for physics domain)

---

### 2.3 Validation & Checkpointing
**Success Criteria:**
- Training loss decreases smoothly
- Validation loss decreases (no overfitting)
- Generated samples are coherent
- Peak memory < 16GB (on M1/M2 Mac)

**Steps:**
1. Periodic evaluation:
   ```python
   def evaluate(model, val_loader, device):
       model.eval()
       total_loss = 0
       with torch.no_grad():
           for batch in val_loader:
               loss = model(**batch).loss
               total_loss += loss.item()
       return total_loss / len(val_loader)
   ```

2. Save best checkpoint:
   ```python
   if val_loss < best_loss:
       model.save_pretrained("checkpoints/best_base_slm")
       best_loss = val_loss
   ```

**Expected Outcome:** Production-ready base SLM weights.

---

## Phase 3: Physics Paper Fine-tuning with LoRA (Week 6–7)

### Goal
Fine-tune base SLM on physics papers using Parameter-Efficient Fine-Tuning (LoRA).

### 3.1 LoRA Setup
**Framework/Software:**
- PEFT (Parameter-Efficient Fine-Tuning from HF)
- LoRA rank: 16–64 (typically 32)
- LoRA alpha: 32

**Why LoRA?**
- Reduces trainable parameters from 3B→50M
- Enables fine-tuning on consumer GPUs/Mac
- Maintains performance comparable to full fine-tuning
- Easy to load/swap for different tasks

**Steps:**
1. Install PEFT:
   ```bash
   pip install peft
   ```

2. Configure LoRA:
   ```python
   from peft import get_peft_model, LoraConfig, TaskType
   
   lora_config = LoraConfig(
       task_type=TaskType.CAUSAL_LM,
       r=32,
       lora_alpha=32,
       lora_dropout=0.05,
       target_modules=["q_proj", "v_proj"],  # or ["c_attn"] for GPT2
       bias="none"
   )
   
   model = get_peft_model(base_model, lora_config)
   model.print_trainable_parameters()
   # Should show ~1–2% trainable params
   ```

---

### 3.2 Fine-tuning Script
**Steps:**
1. Create fine-tuning pipeline:
   ```python
   # scripts/finetune_physics_lora.py
   - Load base SLM
   - Apply LoRA config
   - Train on physics-specific data
   - Save only LoRA weights (~50MB)
   ```

2. Training config:
   ```yaml
   # config/finetune_lora.yaml
   model_name: "checkpoints/best_base_slm"
   
   lora:
     r: 32
     alpha: 32
     dropout: 0.05
   
   training:
     batch_size: 16
     learning_rate: 2e-4
     num_epochs: 3–5
     warmup_steps: 500
     save_every: 500
   ```

3. Launch:
   ```bash
   python scripts/finetune_physics_lora.py \
       --base-model checkpoints/best_base_slm \
       --train-data data/physics_papers/tokenized/train \
       --val-data data/physics_papers/tokenized/val \
       --output-dir checkpoints/physics_lora_weights
   ```

**Expected Outcome:** 
- LoRA weights saved (~50–100MB)
- Fine-tuned SLM ready for RAG integration

---

## Phase 4: RAG (Retrieval-Augmented Generation) Integration (Week 8–9)

### Goal
Build retrieval pipeline to augment SLM with physics paper knowledge on-the-fly.

### 4.1 Vector Database & Embeddings
**Framework/Software:**
- FAISS (Facebook AI Similarity Search) or Pinecone
- Sentence-Transformers for embeddings
- LangChain (optional, for orchestration)

**Steps:**
1. Install dependencies:
   ```bash
   pip install faiss-cpu sentence-transformers langchain
   ```

2. Create embedding model:
   ```python
   from sentence_transformers import SentenceTransformer
   
   embedding_model = SentenceTransformer("all-mpnet-base-v2")  # or physics-specific
   # Dimensions: 384 or 768
   ```

3. Build vector index from physics papers:
   ```python
   # scripts/build_vector_index.py
   import faiss
   
   - Load physics papers chunks
   - Generate embeddings for each chunk
   - Build FAISS index
   - Save index to disk
   
   index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
   index.add(embeddings)
   faiss.write_index(index, "data/physics_index.faiss")
   ```

**Expected Outcome:** 
- FAISS index (~1–5GB, depends on corpus size)
- Fast retrieval capability (ms-level)

---

### 4.2 RAG Pipeline
**Steps:**
1. Create RAG inference script:
   ```python
   # scripts/rag_inference.py
   
   def rag_query(query_text, model, tokenizer, retriever, k=5):
       # 1. Embed query
       query_embedding = embedding_model.encode(query_text)
       
       # 2. Retrieve top-k relevant chunks
       distances, indices = index.search(query_embedding, k)
       retrieved_docs = [corpus[i] for i in indices]
       
       # 3. Build context prompt
       context = "\n".join(retrieved_docs)
       prompt = f"Context:\n{context}\n\nQuestion: {query_text}\nAnswer:"
       
       # 4. Generate with fine-tuned SLM
       inputs = tokenizer(prompt, return_tensors="pt")
       outputs = model.generate(**inputs, max_length=256)
       response = tokenizer.decode(outputs[0])
       
       return response
   ```

2. Test on sample physics questions:
   ```bash
   python scripts/rag_inference.py \
       --query "What is quantum entanglement?" \
       --model checkpoints/physics_lora_weights \
       --index data/physics_index.faiss
   ```

**Expected Outcome:** 
- RAG system generates physics-grounded responses
- Accuracy > 70% on reference questions

---

## Phase 5: Tool-Using Agents (Week 10–11)

### Goal
Integrate tools (search, calculator, code executor) for agentic reasoning.

### 5.1 Tool Definition Framework
**Framework/Software:**
- LangChain (ReAct framework)
- Tool decorators/specifications
- Function calling (JSON schema)

**Steps:**
1. Install LangChain:
   ```bash
   pip install langchain langgraph
   ```

2. Define tools:
   ```python
   # scripts/physics_tools.py
   
   from langchain.tools import tool
   
   @tool
   def search_arxiv(query: str) -> str:
       """Search arXiv for physics papers matching query"""
       # Implementation using arXiv API
       pass
   
   @tool
   def solve_equation(equation: str) -> str:
       """Solve mathematical equations using SymPy"""
       from sympy import solve, symbols
       x = symbols('x')
       result = solve(equation, x)
       return str(result)
   
   @tool
   def execute_python(code: str) -> str:
       """Execute Python code safely for calculations"""
       # Sandboxed execution
       pass
   
   tools = [search_arxiv, solve_equation, execute_python]
   ```

---

### 5.2 Agent Loop (ReAct)
**Steps:**
1. Create agent orchestrator:
   ```python
   # scripts/physics_agent.py
   
   from langchain.agents import AgentExecutor, Tool
   from langchain.llms import HuggingFaceLLM
   
   class PhysicsAgent:
       def __init__(self, model, tools):
           self.model = model
           self.tools = tools
       
       def run(self, query):
           # Thought → Action → Observation → Thought loop
           steps = []
           context = ""
           
           for i in range(max_steps):
               # 1. Generate thought + action
               thought_action = self.model.generate(
                   prompt=f"Context:\n{context}\n\nQuestion: {query}",
                   max_tokens=256
               )
               
               # 2. Parse action (tool name + args)
               tool_name, args = parse_action(thought_action)
               
               # 3. Execute tool
               if tool_name in self.tools:
                   observation = self.tools[tool_name](**args)
                   context += f"Tool: {tool_name}\nResult: {observation}\n"
               
               # 4. Check for final answer
               if "Final Answer:" in thought_action:
                   return thought_action.split("Final Answer:")[-1]
           
           return "Could not find answer"
   ```

2. Test agent:
   ```bash
   python scripts/physics_agent.py \
       --query "What are the implications of Heisenberg uncertainty principle?"
   ```

**Expected Outcome:** 
- Agent reasons through multi-step queries
- Uses tools appropriately
- Provides grounded, coherent answers

---

### 5.3 Tool Integration & Safety
**Steps:**
1. Sandbox execution:
   ```python
   import subprocess
   
   def safe_python_exec(code):
       try:
           result = subprocess.run(
               ["python", "-c", code],
               timeout=5,
               capture_output=True,
               text=True
           )
           return result.stdout or result.stderr
       except subprocess.TimeoutExpired:
           return "Code execution timed out"
   ```

2. Error handling:
   ```python
   try:
       result = tool(args)
   except Exception as e:
       result = f"Tool failed: {str(e)}"
       context += result
   ```

**Expected Outcome:** 
- Safe, sandboxed tool execution
- Graceful error recovery

---

## Phase 6: Integration & Deployment (Week 12+)

### Goal
Package everything into a production-ready Physics Research Assistant.

### 6.1 Unified Application
**Framework/Software:**
- FastAPI (REST API)
- Streamlit (UI, optional)
- Docker (containerization)

**Steps:**
1. Create FastAPI endpoint:
   ```python
   # app/main.py
   from fastapi import FastAPI, HTTPException
   from pydantic import BaseModel
   
   app = FastAPI()
   
   # Load all components
   base_model = load_base_model()
   lora_weights = load_lora()
   merged_model = merge_lora(base_model, lora_weights)
   agent = PhysicsAgent(merged_model, tools)
   
   class QueryRequest(BaseModel):
       question: str
       use_tools: bool = True
   
   @app.post("/query")
   async def query(request: QueryRequest):
       try:
           if request.use_tools:
               answer = agent.run(request.question)
           else:
               answer = merged_model.generate(request.question)
           return {"answer": answer}
       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))
   
   # Launch: uvicorn app.main:app --reload
   ```

2. Optional Streamlit UI:
   ```python
   # app/ui.py
   import streamlit as st
   import requests
   
   st.title("Physics Research Assistant")
   query = st.text_input("Ask a physics question:")
   use_tools = st.checkbox("Use tool-using agent", value=True)
   
   if st.button("Submit"):
       response = requests.post("http://localhost:8000/query", 
           json={"question": query, "use_tools": use_tools})
       st.write(response.json()["answer"])
   ```

---

### 6.2 Deployment
**Steps:**
1. Create Dockerfile:
   ```dockerfile
   FROM python:3.10-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 8000
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0"]
   ```

2. Docker build & run:
   ```bash
   docker build -t physics-assistant:latest .
   docker run -p 8000:8000 physics-assistant:latest
   ```

3. Cloud deployment (optional):
   - Hugging Face Spaces
   - AWS Lambda / EC2
   - Google Cloud Run
   - Azure

**Expected Outcome:** 
- Production-ready REST API
- Easy deployment & scaling

---

## Complete Software & Framework Requirements

| Phase | Component | Framework/Software | Version | Purpose |
|-------|-----------|-------------------|---------|---------|
| 1 | Data Processing | Python, Pandas, NumPy | 3.10+, latest | ETL, cleaning |
| 1 | Dataset | arXiv API | latest | Physics papers |
| 1 | Tokenization | Subword-tokenizer (Rust) | v0.1 | 32K vocab |
| 2 | Base SLM Training | PyTorch, Transformers, Accelerate | 2.0+, 4.30+, latest | Model training |
| 3 | LoRA Fine-tuning | PEFT, PyTorch | latest | Parameter-efficient tuning |
| 4 | RAG | FAISS, Sentence-Transformers, LangChain | latest | Retrieval + embeddings |
| 5 | Agents | LangChain, LanGraph | latest | Tool orchestration |
| 6 | API | FastAPI, Uvicorn | latest | REST endpoint |
| 6 | UI (optional) | Streamlit | latest | User interface |
| 6 | Deployment | Docker | latest | Containerization |

---

## Timeline & Milestones

| Week | Phase | Milestone | Deliverable |
|------|-------|-----------|-------------|
| 1–2 | Phase 1 | Data pipeline ready | 300M–1B tokenized tokens |
| 3–5 | Phase 2 | Base SLM trained | 3B–7B model checkpoint |
| 6–7 | Phase 3 | LoRA fine-tuning done | 50MB LoRA weights |
| 8–9 | Phase 4 | RAG integrated | Vector index + retrieval working |
| 10–11 | Phase 5 | Agents operational | Multi-step reasoning demo |
| 12+ | Phase 6 | Production deployment | API + UI live |

---

## Hardware Requirements

### Minimum (M1/M2 Mac):
- CPU: Apple Silicon
- RAM: 16GB+ (with swap)
- Storage: 100GB SSD

### Recommended:
- GPU: GPU with 24GB+ VRAM (or cloud)
- RAM: 32–64GB
- Storage: 500GB NVMe SSD

---

## Success Criteria (Final)

✅ Physics Research Assistant generates:
- Coherent physics-grounded answers
- Uses tools when appropriate
- Cites retrieved papers
- Handles multi-step reasoning
- API response time < 5s
- Deployment-ready

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Out of memory | Reduce batch size, use quantization, cloud GPU |
| Slow training | Use mixed precision, gradient checkpointing |
| Low-quality RAG results | Improve embeddings, use domain-specific model |
| Agent loop infinite | Set max steps, timeout handling |
| Deployment fails | Docker testing, staging environment |

---

## Next Immediate Actions

1. **Today/Tomorrow:**
   - Prepare Python environment (Phase 1.1)
   - Download physics papers (~10K initially) (Phase 1.2)

2. **This Week:**
   - Preprocess dataset (Phase 1.3)
   - Tokenize with existing tokenizer (Phase 1.4)
   - Start base SLM training (Phase 2)

3. **Next Week:**
   - Monitor training, validate results
   - Prepare LoRA fine-tuning setup

---

## Questions to Clarify Before Starting

1. **Dataset size:** How many physics papers? (Affects timeline)
2. **Hardware:** Will you use cloud GPU or Mac MPS?
3. **Model size:** Prefer 3B or 7B? (Speed vs quality tradeoff)
4. **Deployment:** Local/cloud/both?
5. **Priority:** Fast prototype or high accuracy?

---

**Document Version:** 1.0  
**Last Updated:** June 22, 2026  
**Authors:** Jaydip Singh, Linkan Kumbhar
