#!/usr/bin/env python3
import json, sys, re
from collections import Counter
import time

def train_bpe_fast(corpus_path, vocab_size, output_path):
    print(f"Loading: {corpus_path}")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Corpus: {len(text):,} chars")
    
    words = text.split()
    word_tokens = {}
    vocab = set()
    
    for word in words:
        if not word: continue
        chars = tuple(word)
        word_tokens[chars] = word_tokens.get(chars, 0) + 1
        for c in chars:
            vocab.add(c)
    
    print(f"Unique words: {len(word_tokens):,}")
    print(f"Char vocab: {len(vocab)}")
    
    target = vocab_size - len(vocab)
    print(f"Target merges: {target}")
    
    merges = []
    start = time.time()
    
    for i in range(target):
        pairs = Counter()
        for wt, cnt in word_tokens.items():
            for j in range(len(wt)-1):
                pairs[(wt[j], wt[j+1])] += cnt
        
        if not pairs: break
        best, count = pairs.most_common(1)[0]
        if count <= 1: 
            print(f"Early stop at {i}")
            break
            
        merged = best[0] + best[1]
        vocab.add(merged)
        merges.append(best)
        
        new_words = {}
        for wt, cnt in word_tokens.items():
            nw = []
            j = 0
            while j < len(wt):
                if j+1 < len(wt) and wt[j] == best[0] and wt[j+1] == best[1]:
                    nw.append(merged)
                    j += 2
                else:
                    nw.append(wt[j])
                    j += 1
            new_words[tuple(nw)] = new_words.get(tuple(nw), 0) + cnt
        word_tokens = new_words
        
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{target} | vocab={len(vocab)} | time={time.time()-start:.1f}s | best={best} ({count}x)")
    
    print(f"\nDone! Vocab={len(vocab)} | Merges={len(merges)} | Time={time.time()-start:.1f}s")
    
    special = ["<unk>", "<pad>", "<sos>", "<eos>", "<space>"]
    vocab_list = sorted(vocab)
    
    t2i, i2t, idx = {}, {}, 0
    for t in special:
        t2i[t] = idx; i2t[idx] = t; idx += 1
    for t in vocab_list:
        if t not in t2i:
            t2i[t] = idx; i2t[idx] = t; idx += 1
    
    model = {
        "vocab": vocab_list,
        "merges": [[a,b] for a,b in merges],
        "token_to_id": t2i,
        "id_to_token": {str(k):v for k,v in i2t.items()},
        "special_tokens": special
    }
    
    with open(output_path, 'w') as f:
        json.dump(model, f, indent=2)
    print(f"Saved: {output_path}")
    return model

def tokenize(model, text):
    merges = [(a,b) for a,b in model["merges"]]
    words = re.findall(r'\S+|\s+', text)
    out = []
    for w in words:
        if w.isspace():
            out.append(" ")
            continue
        toks = list(w)
        for a,b in merges:
            m = a+b
            new = []
            i = 0
            while i < len(toks):
                if i+1 < len(toks) and toks[i]==a and toks[i+1]==b:
                    new.append(m); i += 2
                else:
                    new.append(toks[i]); i += 1
            toks = new
        out.extend(toks)
    return out

def encode(model, text):
    toks = tokenize(model, text)
    t2i = model["token_to_id"]
    ids = []
    for t in toks:
        if t == " ":
            ids.append(t2i.get("<space>", 0))
        elif t in t2i:
            ids.append(t2i[t])
        else:
            for c in t:
                ids.append(t2i.get(c, t2i.get("<unk>", 0)))
    return ids

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 fast_bpe.py <corpus> <vocab_size> [output.json]")
        sys.exit(1)
    
    model = train_bpe_fast(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv)>3 else "model.json")
    
    test = "hello world how are you today"
    print(f"\nTest: '{test}'")
    print(f"Tokens: {tokenize(model, test)}")
    print(f"IDs: {encode(model, test)}")
