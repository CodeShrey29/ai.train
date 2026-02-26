#!/usr/bin/env python3
"""
Autonomous Progressive Growth AI Training System
ADVANCED TRAINING-ONLY VERSION - merged improvements from ai.py

Features:
- Continuously fetches data from Wikipedia & many knowledge domains
- Progressive Growth: 10M → 50M → 100M → 200M → 350M → 500M
- Smart checkpointing: Never overwrites, timestamped saves
- Persistent trained-hashes to avoid re-training same data
- Memory optimized: dynamic batching, memory monitor
- Robust DataFetcher with fetched history and expanded category coverage (1000+ generated categories)
- Advanced weight transfer during growth

This file updates the original train.py to include the stronger training pipeline from ai.py while keeping it training-only (no chat server).
"""

import os
import sys
import time
import json
import math
import random
import requests
import hashlib
import threading
import re
import gc
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Optional DirectML support (if available)
try:
    import torch_directml
    DIRECTML_AVAILABLE = True
except Exception:
    DIRECTML_AVAILABLE = False

# Optional imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[Warning] psutil not installed - memory monitoring disabled")

try:
    from PIL import Image
    import io
    import base64
    IMAGING_AVAILABLE = True
except ImportError:
    IMAGING_AVAILABLE = False
    print("[Warning] PIL not installed - image features disabled")

# =============================
# CONFIGURATION
# =============================
class ModelConfig:
    PRESETS = {
        "10M": {"hidden_dim": 512, "num_layers": 8, "num_heads": 8, "ffn_mult": 4, "batch_size": 2, "grad_accum": 8},
        "50M": {"hidden_dim": 768, "num_layers": 12, "num_heads": 12, "ffn_mult": 4, "batch_size": 8, "grad_accum": 2},
        "100M": {"hidden_dim": 1024, "num_layers": 18, "num_heads": 16, "ffn_mult": 4, "batch_size": 4, "grad_accum": 4},
        "200M": {"hidden_dim": 1280, "num_layers": 24, "num_heads": 20, "ffn_mult": 4, "batch_size": 2, "grad_accum": 8},
        "350M": {"hidden_dim": 1536, "num_layers": 28, "num_heads": 24, "ffn_mult": 4, "batch_size": 2, "grad_accum": 12},
        "500M": {"hidden_dim": 2048, "num_layers": 32, "num_heads": 32, "ffn_mult": 4, "batch_size": 1, "grad_accum": 16},
    }
    GROWTH_PATH = ["10M", "50M", "100M", "200M", "350M", "500M"]

    def __init__(self, size="10M"):
        preset = self.PRESETS.get(size, self.PRESETS["10M"]) 
        self.SIZE = size
        self.HIDDEN_DIM = preset["hidden_dim"]
        self.NUM_LAYERS = preset["num_layers"]
        self.NUM_HEADS = preset["num_heads"]
        self.NUM_KV_HEADS = max(1, preset["num_heads"] // 2)
        self.FFN_DIM = preset["hidden_dim"] * preset["ffn_mult"]
        self.HEAD_DIM = self.HIDDEN_DIM // self.NUM_HEADS

        self.VOCAB_SIZE = 32000
        self.MAX_SEQ_LEN = 512
        self.DROPOUT = 0.1
        self.BIAS = False
        self.ROPE_THETA = 10000.0

        # baseline preset batch & grad accum
        self.BATCH_SIZE = preset["batch_size"]
        self.GRAD_ACCUMULATION_STEPS = preset["grad_accum"]

        self.LEARNING_RATE = 3e-4
        self.WARMUP_STEPS = 2000
        self.WEIGHT_DECAY = 0.1
        self.GRAD_CLIP = 1.0
        self.BETA1 = 0.9
        self.BETA2 = 0.95
        self.EPS = 1e-8

        self.DATA_FILE = "data.txt"
        self.TOKENIZER_FILE = "tokenizer.json"
        self.CHECKPOINT_DIR = "checkpoints"
        self.MODEL_FILE = "model.pt"
        self.GROWTH_LOG_FILE = "growth_log.json"

        # checkpoint pruning: keep this many most recent checkpoints
        self.CHECKPOINT_INTERVAL = 300
        self.CHECKPOINT_MAX_KEEP = 3
        self.FLUSH_INTERVAL = 50
        self.MIN_DATA_LINES = 3000
        self.FETCH_BATCH = 10
        self.FETCH_DELAY = 5
        self.PREFETCH_TARGET_LINES = 2_000_000

        self.GROWTH_LOSS_THRESHOLD = 3.0
        self.GROWTH_STABLE_STEPS = 1000

        # Device selection: DirectML > CUDA > CPU
        if DIRECTML_AVAILABLE:
            try:
                self.DEVICE = torch_directml.device()
                self.DEVICE_TYPE = 'directml'
            except Exception:
                self.DEVICE = torch.device('cpu')
                self.DEVICE_TYPE = 'cpu'
        elif torch.cuda.is_available():
            self.DEVICE = torch.device('cuda')
            self.DEVICE_TYPE = 'cuda'
        else:
            self.DEVICE = torch.device('cpu')
            self.DEVICE_TYPE = 'cpu'

        self.SEED = 42

        # Aggressive memory targeting: use most of system RAM (95%) for MAX_MEMORY_GB
        if PSUTIL_AVAILABLE:
            total_bytes = psutil.virtual_memory().total
            total_gb = total_bytes / (1024 ** 3)
            # target 95% of system RAM
            self.MAX_MEMORY_GB = round(total_gb * 0.95, 1)
            # reserve ~5% of system RAM (in bytes) for OS and safety; used by MemoryMonitor
            self.RESERVE_BYTES = int(total_bytes * 0.05)
            # auto-scale batch size conservatively based on RAM: every 4GB lets us multiply batch
            scale = max(1, int(total_gb // 4))
            # cap batch size to avoid runaway values
            self.BATCH_SIZE = min(self.BATCH_SIZE * scale, 64)
        else:
            self.MAX_MEMORY_GB = 5.6
            self.RESERVE_BYTES = 64 * 1024 * 1024

        self.ENABLE_MIXED_PRECISION = (self.DEVICE_TYPE == "cuda")

# =============================
# TOKENIZER (BPE simple)
# =============================
class BPETokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.special_tokens = {'<pad>':0,'<unk>':1,'<bos>':2,'<eos>':3,'<sep>':4,'<cls>':5,'<mask>':6}
        self.vocab = {}
        self.merges = {}
        self.cache = {}
        self.inverse_vocab = {}

    def train(self, texts, min_frequency=2):
        print("Training BPE tokenizer...")
        word_counts = {}
        for text in texts:
            for word in text.strip().split():
                word_with_end = ' '.join(list(word)) + ' </w>'
                word_counts[word_with_end] = word_counts.get(word_with_end,0) + 1
        vocab = set()
        for word in word_counts:
            for char in word.split():
                vocab.add(char)
        for token in self.special_tokens:
            vocab.add(token)
        num_merges = max(0, self.vocab_size - len(vocab))
        word_items = list(word_counts.items())
        for i in range(num_merges):
            pair_counts = {}
            for word, count in word_items:
                symbols = word.split()
                for j in range(len(symbols)-1):
                    pair = (symbols[j], symbols[j+1])
                    pair_counts[pair] = pair_counts.get(pair,0) + count
            if not pair_counts:
                break
            best_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
            self.merges[best_pair] = i
            merged = ''.join(best_pair)
            vocab.add(merged)
            new_items = []
            for word, count in word_items:
                symbols = word.split()
                new_symbols = []
                j = 0
                while j < len(symbols):
                    if j < len(symbols)-1 and (symbols[j], symbols[j+1]) == best_pair:
                        new_symbols.append(merged)
                        j += 2
                    else:
                        new_symbols.append(symbols[j])
                        j += 1
                new_items.append((' '.join(new_symbols), count))
            word_items = new_items
        sorted_vocab = sorted(list(vocab))
        self.vocab = {symbol: idx for idx, symbol in enumerate(sorted_vocab)}
        # ensure special tokens map to their ids
        for token, idx in self.special_tokens.items():
            if token in self.vocab:
                old_idx = self.vocab[token]
                self.vocab[token] = idx
                for s,i in list(self.vocab.items()):
                    if i == idx and s != token:
                        self.vocab[s] = old_idx
                        break
            else:
                self.vocab[token] = idx
        self.inverse_vocab = {v:k for k,v in self.vocab.items()}
        print(f"Tokenizer trained. Final vocabulary: {len(self.vocab)} tokens")

    def encode(self, text, add_special_tokens=True):
        if text in self.cache:
            return self.cache[text].copy()
        words = text.strip().split()
        tokens = []
        if add_special_tokens:
            tokens.append(self.special_tokens['<bos>'])
        for word in words:
            word_symbols = list(word) + ['</w>']
            changed = True
            while changed and len(word_symbols) > 1:
                changed = False
                best_pair = None
                best_score = -1
                best_idx = 0
                for i in range(len(word_symbols)-1):
                    pair = (word_symbols[i], word_symbols[i+1])
                    if pair in self.merges:
                        score = self.merges[pair]
                        if score > best_score:
                            best_score = score
                            best_pair = pair
                            best_idx = i
                if best_pair is not None:
                    word_symbols = (word_symbols[:best_idx] + [''.join(best_pair)] + word_symbols[best_idx+2:])
                    changed = True
            for symbol in word_symbols:
                if symbol in self.vocab:
                    tokens.append(self.vocab[symbol])
                else:
                    tokens.append(self.special_tokens['<unk>'])
        if add_special_tokens:
            tokens.append(self.special_tokens['<eos>'])
        self.cache[text] = tokens.copy()
        if len(self.cache) > 10000:
            self.cache.pop(next(iter(self.cache)))
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        text = ""
        for tid in token_ids:
            token = self.inverse_vocab.get(tid, '<unk>')
            if skip_special_tokens and token in self.special_tokens:
                continue
            if token == '</w>':
                text += ' '
            else:
                text += token
        return text.strip()

    def save(self, path):
        data = {'vocab': self.vocab, 'merges': {f"{k[0]}|||{k[1]}":v for k,v in self.merges.items()}, 'special_tokens': self.special_tokens}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {path}")

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.merges = {tuple(k.split('|||')):v for k,v in data['merges'].items()}
        self.special_tokens = data.get('special_tokens', self.special_tokens)
        self.inverse_vocab = {v:k for k,v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        print(f"Tokenizer loaded from {path}: {len(self.vocab)} tokens")

# =============================
# MEMORY MONITOR
# =============================
class MemoryMonitor:
    # small default reserve; ModelConfig will pass a more accurate reserve_bytes when available
    RESERVE_BYTES = 64 * 1024 * 1024
    def __init__(self, max_memory_gb=5.0, reserve_bytes=None):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        if reserve_bytes is not None:
            self.RESERVE_BYTES = reserve_bytes
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
        else:
            self.process = None
    def get_memory_usage(self):
        if self.process:
            return self.process.memory_info().rss
        return 0
    def get_memory_usage_gb(self):
        return self.get_memory_usage() / (1024 ** 3)
    def get_free_ram_gb(self):
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().available / (1024 ** 3)
        return 999.0
    def is_memory_safe(self):
        if not self.process:
            return True
        if PSUTIL_AVAILABLE:
            # consider memory safe if available is greater than the configured reserve
            return psutil.virtual_memory().available > self.RESERVE_BYTES
        return self.get_memory_usage() < self.max_memory_bytes * 0.9
    def can_boost(self):
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().available > self.RESERVE_BYTES
        return False
    def force_cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if DIRECTML_AVAILABLE:
            gc.collect()

# =============================
# MODEL GROWTH MANAGER
# =============================
class ModelGrowthManager:
    def __init__(self, config):
        self.config = config
        self.growth_log_file = config.GROWTH_LOG_FILE
        self.growth_history = self.load_growth_log()
    def load_growth_log(self):
        if os.path.exists(self.growth_log_file):
            try:
                with open(self.growth_log_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []
    def save_growth_log(self):
        try:
            with open(self.growth_log_file, 'w') as f:
                json.dump(self.growth_history, f, indent=2)
        except Exception:
            pass
    def log_growth_event(self, from_size, to_size, step, loss, metrics):
        event = {'timestamp': datetime.now().isoformat(),'from_size':from_size,'to_size':to_size,'training_step':step,'loss_at_growth':loss,'metrics':metrics}
        self.growth_history.append(event)
        self.save_growth_log()
        print(f"\n{'='*60}")
        print(f"  GROWTH EVENT: {from_size} → {to_size} | step {step} | loss {loss:.4f}")
        print(f"{'='*60}\n")
    def get_next_size(self, current_size):
        path = ModelConfig.GROWTH_PATH
        if current_size not in path:
            return None
        idx = path.index(current_size)
        if idx >= len(path)-1:
            return None
        return path[idx+1]
    def transfer_weights(self, old_model, new_model):
        print("Transferring weights to larger model...")
        old_state = old_model.state_dict()
        new_state = new_model.state_dict()
        # Smart transfer: copy matching tensors, slice partially where needed
        transferred = 0
        for k, v in old_state.items():
            if k in new_state:
                if v.shape == new_state[k].shape:
                    new_state[k] = v.clone()
                    transferred += 1
                else:
                    # try to copy partial slices
                    try:
                        target = new_state[k]
                        if v.ndim == 2 and target.ndim == 2:
                            min0 = min(v.shape[0], target.shape[0])
                            min1 = min(v.shape[1], target.shape[1])
                            target[:min0,:min1] = v[:min0,:min1].clone()
                            new_state[k] = target
                            transferred += 1
                        elif v.ndim == 1 and target.ndim == 1:
                            m = min(v.shape[0], target.shape[0])
                            target[:m] = v[:m].clone()
                            new_state[k] = target
                            transferred += 1
                    except Exception:
                        pass
        new_model.load_state_dict(new_state)
        print(f"Weight transfer complete. Transferred ~{transferred} tensors")
        return new_model

# =============================
# TRANSFORMER (lightweight)
# =============================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    def forward(self, seq_len):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.to(q.dtype).unsqueeze(0).unsqueeze(2)
    sin = sin.to(q.dtype).unsqueeze(0).unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.HIDDEN_DIM
        self.num_heads = config.NUM_HEADS
        self.num_kv_heads = config.NUM_KV_HEADS
        self.head_dim = config.HEAD_DIM
        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=config.BIAS)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.BIAS)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.BIAS)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=config.BIAS)
        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=config.MAX_SEQ_LEN, theta=config.ROPE_THETA)
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(config.DROPOUT)
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        cos, sin = self.rotary(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // max(1, self.num_kv_heads)
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        attn_weights = torch.matmul(q, k.transpose(-2,-1)) * self.scale
        if attention_mask is None:
            causal = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
            attn_weights = attn_weights + causal
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.HIDDEN_DIM, config.FFN_DIM, bias=config.BIAS)
        self.w3 = nn.Linear(config.HIDDEN_DIM, config.FFN_DIM, bias=config.BIAS)
        self.w2 = nn.Linear(config.FFN_DIM, config.HIDDEN_DIM, bias=config.BIAS)
        self.dropout = nn.Dropout(config.DROPOUT)
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.HIDDEN_DIM)
        self.ffn_norm = RMSNorm(config.HIDDEN_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)
    def forward(self, x, attention_mask=None):
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, attention_mask)
        x = self.dropout(x)
        x = residual + x
        residual = x
        x = self.ffn_norm(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        return x

class AdvancedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.VOCAB_SIZE, config.HIDDEN_DIM)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.NUM_LAYERS)])
        self.norm = RMSNorm(config.HIDDEN_DIM)
        self.output = nn.Linear(config.HIDDEN_DIM, config.VOCAB_SIZE, bias=False)
        self._init_weights()
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized: {total_params:,} parameters")
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, input_ids, attention_mask=None):
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        return self.output(x)

# =============================
# DATASET (persistent hashes)
# =============================
class ConsumingDataset:
    TRAINED_HASHES_FILE = "trained_data.json"
    def __init__(self, file_path, tokenizer, max_length=512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lock = threading.Lock()
        self.lines = []
        self.pending_hashes = set()
        self.trained_hashes = set()
        self.consumed_count = 0
        self._load_trained_hashes()
        self._load_lines()
    def _load_trained_hashes(self):
        if os.path.exists(self.TRAINED_HASHES_FILE):
            try:
                with open(self.TRAINED_HASHES_FILE, 'r') as f:
                    data = json.load(f)
                self.trained_hashes = set(data.get('hashes', []))
                print(f"[Data] Loaded {len(self.trained_hashes)} trained data hashes")
            except Exception:
                self.trained_hashes = set()
    def _save_trained_hashes(self):
        try:
            tmp = self.TRAINED_HASHES_FILE + ".tmp"
            with open(tmp, 'w') as f:
                json.dump({'hashes': list(self.trained_hashes), 'count': len(self.trained_hashes)}, f)
            if os.path.exists(self.TRAINED_HASHES_FILE):
                os.remove(self.TRAINED_HASHES_FILE)
            os.rename(tmp, self.TRAINED_HASHES_FILE)
        except Exception:
            pass
    def _hash(self, line):
        return hashlib.md5(line.encode('utf-8', errors='ignore')).hexdigest()
    def _load_lines(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                pass
            self.lines = []
            return
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = [l.strip() for l in f if l.strip() and len(l.strip()) > 20]
        with self.lock:
            self.lines = []
            self.pending_hashes = set()
            for line in raw:
                h = self._hash(line)
                if h not in self.trained_hashes and h not in self.pending_hashes:
                    self.pending_hashes.add(h)
                    self.lines.append(line)
            self.consumed_count = 0
        if len(raw) != len(self.lines):
            self._rewrite_file()
    def _rewrite_file(self):
        try:
            tmp = self.file_path + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                for line in self.lines:
                    f.write(line + '\n')
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
            os.rename(tmp, self.file_path)
        except Exception:
            pass
    def available(self):
        with self.lock:
            return len(self.lines) - self.consumed_count
    def total_lines(self):
        with self.lock:
            return len(self.lines)
    def get_batch(self, batch_size):
        with self.lock:
            if self.consumed_count >= len(self.lines):
                return None
            batch_lines = self.lines[self.consumed_count:self.consumed_count + batch_size]
            self.consumed_count += len(batch_lines)
        if self.tokenizer is None:
            return None
        all_input = []
        all_target = []
        for line in batch_lines:
            tokens = self.tokenizer.encode(line)
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                pad = self.tokenizer.special_tokens['<pad>']
                tokens = tokens + [pad] * (self.max_length - len(tokens))
            all_input.append(tokens[:-1])
            all_target.append(tokens[1:])
        return torch.tensor(all_input), torch.tensor(all_target)
    def flush_trained(self):
        with self.lock:
            if self.consumed_count == 0:
                return 0
            trained_lines = self.lines[:self.consumed_count]
            for line in trained_lines:
                self.trained_hashes.add(self._hash(line))
            self.lines = self.lines[self.consumed_count:]
            flushed = self.consumed_count
            self.consumed_count = 0
        self._save_trained_hashes()
        self._rewrite_file()
        return flushed
    def flush_consumed(self):
        return 0
    def add_lines(self, new_lines):
        clean = [l.strip() for l in new_lines if l.strip() and len(l.strip()) > 20]
        if not clean:
            return
        unique = []
        with self.lock:
            for line in clean:
                h = self._hash(line)
                if h not in self.trained_hashes and h not in self.pending_hashes:
                    self.pending_hashes.add(h)
                    self.lines.append(line)
                    unique.append(line)
        if unique:
            try:
                with open(self.file_path, 'a', encoding='utf-8') as f:
                    for line in unique:
                        f.write(line + '\n')
            except Exception:
                pass
    def reload_from_file(self):
        self._load_lines()
# =============================
# DATA FETCHER (expanded categories)
# =============================
class DataFetcher:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.running = True
        self.fetched_count = 0
        self.session = requests.Session()
        self.session.headers.update({'User-Agent':'AutonomousAITrainer/1.0'})
        self.fetched_urls = set()
        self.history_file = Path("fetched_history.json")
        if self.history_file.exists():
            try:
                data = json.load(self.history_file.open('r'))
                if isinstance(data, dict):
                    self.fetched_urls = set(data.get('fetched_urls', []))
                    self.category_idx = data.get('category_idx', 0)
                    self.fetched_count = data.get('fetched_count', 0)
                    print(f"[Fetcher] Resumed — {len(self.fetched_urls)} sources already fetched, category_idx={self.category_idx}")
                else:
                    self.fetched_urls = set(data)
            except Exception:
                self.fetched_urls = set()
        # base categories (combined from original files)
        base = [
            "Science", "Technology", "Mathematics", "Physics", "Chemistry", "Biology",
            "Computer_science", "Engineering", "Medicine", "Astronomy", "Geology",
            "History", "Philosophy", "Psychology", "Sociology", "Economics",
            "Politics", "Geography", "Literature", "Art", "Music"
        ]
        # programmatically expand to 1000+ categories by creating sensible variants
        self.WIKI_CATEGORIES = self._expand_categories(base, target=1100)
        self.category_idx = 0
        self.gutenberg_ids = list(range(1, 400))  # larger set for diversity
        self.vital_levels = [1,2,3,4,5]
        self.search_queries = [
            "fundamental physics concepts", "cellular biology mechanisms",
            "historical civilizations development", "modern technology advances",
            "mathematical theorems proofs", "climate science research",
            "economic theories models", "psychological phenomena studies",
            "philosophical arguments debates", "linguistic structures analysis"
        ]

    def _expand_categories(self, base_list, target=1100):
        out = []
        seen = set()
        suffixes = ["", "_history", "_overview", "_theory", "_applications", "_biography", "_concepts", "_research", "_introduction"]
        for b in base_list:
            for s in suffixes:
                cand = (b + s).replace(' ', '_')
                if cand not in seen:
                    out.append(cand)
                    seen.add(cand)
        # Generate subtopics by combining words
        words = sorted({w for b in base_list for w in re.split(r'[_\s]+', b)})
        i = 0
        while len(out) < target:
            a = random.choice(words)
            b = random.choice(words)
            cand = f"{a}_{b}"
            if cand not in seen:
                out.append(cand)
                seen.add(cand)
            i += 1
            if i > target * 5:
                break
        return out

    def _save_history(self):
        try:
            data = {
                'fetched_urls': list(self.fetched_urls),
                'category_idx': self.category_idx,
                'fetched_count': self.fetched_count,
            }
            tmp = str(self.history_file) + ".tmp"
            with open(tmp, 'w') as f:
                json.dump(data, f)
            if self.history_file.exists():
                os.remove(self.history_file)
            os.rename(tmp, str(self.history_file))
        except Exception:
            pass

    def _clean_text(self, text):
        text = re.sub(r'\[\d+]', '', text)
        text = re.sub(r'\{[^}]*\}', '', text)
        text = re.sub(r'==+\s*See also\s*==+', '', text, flags=re.DOTALL)
        text = re.sub(r'==+\s*References\s*==+', '', text, flags=re.DOTALL)
        text = re.sub(r'==+\s*External links\s*==+', '', text, flags=re.DOTALL)
        text = re.sub(r'==+[^=]+=+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        lines = []
        for para in text.split('\n'):
            para = para.strip()
            if len(para) > 50 and sum(c.isalpha() for c in para) > len(para) * 0.4:
                if len(para) > 500:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    chunk = ''
                    for s in sentences:
                        if len(chunk) + len(s) > 400:
                            if len(chunk) > 50:
                                lines.append(chunk.strip())
                            chunk = s
                        else:
                            chunk += ' ' + s if chunk else s
                    if len(chunk) > 50:
                        lines.append(chunk.strip())
                else:
                    lines.append(para)
        return lines

    def _fetch_wiki_article(self, title):
        if title in self.fetched_urls:
            return []
        try:
            resp = self.session.get('https://en.wikipedia.org/w/api.php', params={'action':'query','titles':title,'prop':'extracts','explaintext':True,'format':'json'}, timeout=15)
            pages = resp.json().get('query', {}).get('pages', {})
            for pid, page in pages.items():
                if 'extract' in page and len(page['extract']) > 200:
                    lines = self._clean_text(page['extract'])
                    self.fetched_urls.add(title)
                    return lines
        except Exception:
            pass
        return []

    def fetch_wikipedia_random(self, count=10):
        all_lines = []
        try:
            resp = self.session.get('https://en.wikipedia.org/w/api.php', params={'action':'query','list':'random','rnnamespace':0,'rnlimit':count,'format':'json'}, timeout=15)
            titles = [a['title'] for a in resp.json().get('query', {}).get('random', [])]
            for title in titles:
                lines = self._fetch_wiki_article(title)
                all_lines.extend(lines)
                if lines:
                    time.sleep(0.2)
        except Exception as e:
            print(f"[Fetcher] Wikipedia random error: {e}")
        return all_lines

    def fetch_wikipedia_category(self):
        category = self.WIKI_CATEGORIES[self.category_idx % len(self.WIKI_CATEGORIES)]
        self.category_idx += 1
        all_lines = []
        try:
            resp = self.session.get('https://en.wikipedia.org/w/api.php', params={'action':'query','list':'categorymembers','cmtitle':f'Category:{category}','cmlimit':15,'cmtype':'page','format':'json'}, timeout=15)
            members = resp.json().get('query', {}).get('categorymembers', [])
            for member in members[:10]:
                lines = self._fetch_wiki_article(member['title'])
                all_lines.extend(lines)
                if lines:
                    time.sleep(0.2)
        except Exception as e:
            print(f"[Fetcher] Category '{category}' error: {e}")
        return all_lines, category

    def fetch_wikipedia_vital(self):
        topic = random.choice(["Universe","Earth","Life","Human","Science","Mathematics"])
        lines = self._fetch_wiki_article(topic)
        return lines, topic

    def fetch_wikipedia_search(self):
        query = random.choice(self.search_queries)
        all_lines = []
        try:
            resp = self.session.get('https://en.wikipedia.org/w/api.php', params={'action':'query','list':'search','srsearch':query,'srlimit':5,'format':'json'}, timeout=15)
            results = resp.json().get('query', {}).get('search', [])
            for r in results:
                lines = self._fetch_wiki_article(r['title'])
                all_lines.extend(lines)
                if lines:
                    time.sleep(0.2)
        except Exception as e:
            print(f"[Fetcher] Wiki search '{query}' error: {e}")
        return all_lines, query

    def fetch_gutenberg(self):
        # choose a random id from the list; attempt two URL patterns
        book_id = random.choice(self.gutenberg_ids)
        url1 = f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt'
        url2 = f'https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt'
        for url in (url1, url2):
            if url in self.fetched_urls:
                continue
            try:
                resp = self.session.get(url, timeout=20)
                if resp.status_code == 200:
                    text = resp.content.decode('utf-8', errors='ignore')
                    lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 30]
                    self.fetched_urls.add(url)
                    return lines[:200]
            except Exception:
                pass
        return []

    def fetch_wikisource(self):
        # sample a primary document
        doc_list = ['United_States_Constitution','Declaration_of_Independence_(United_States)','Magna_Carta']
        doc = random.choice(doc_list)
        key = f'wikisource:{doc}'
        if key in self.fetched_urls:
            return [], doc
        try:
            resp = self.session.get('https://en.wikisource.org/w/api.php', params={'action':'query','titles':doc,'prop':'extracts','explaintext':True,'format':'json'}, timeout=15)
            pages = resp.json().get('query', {}).get('pages', {})
            for pid,page in pages.items():
                if 'extract' in page and len(page['extract']) > 100:
                    lines = self._clean_text(page['extract'])
                    self.fetched_urls.add(key)
                    return lines, doc
        except Exception as e:
            print(f"[Fetcher] Wikisource {doc} error: {e}")
        return [], doc

    def fetch_wikiquote(self):
        people = ['Albert_Einstein','Isaac_Newton','Aristotle','Plato']
        person = random.choice(people)
        key = f'wikiquote:{person}'
        if key in self.fetched_urls:
            return []
        try:
            resp = self.session.get('https://en.wikiquote.org/w/api.php', params={'action':'query','titles':person,'prop':'extracts','explaintext':True,'format':'json'}, timeout=15)
            pages = resp.json().get('query', {}).get('pages', {})
            for pid,page in pages.items():
                if 'extract' in page and len(page['extract']) > 50:
                    lines = self._clean_text(page['extract'])
                    self.fetched_urls.add(key)
                    return lines
        except Exception:
            pass
        return []

    def fetch_open_textbook(self):
        urls = [
            'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        ]
        url = random.choice(urls)
        if url in self.fetched_urls:
            return []
        try:
            resp = self.session.get(url, timeout=20)
            resp.raise_for_status()
            text = resp.content.decode('utf-8', errors='ignore')
            lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip())>30]
            self.fetched_urls.add(url)
            return lines
        except Exception:
            return []

    def run_forever(self):
        sources = [
            ('Wikipedia (random)', lambda: self.fetch_wikipedia_random(self.config.FETCH_BATCH)),
            ('Wikipedia (category)', lambda: self.fetch_wikipedia_category()[0]),
            ('Wikipedia (vital)', lambda: self.fetch_wikipedia_vital()[0]),
            ('Wikipedia (search)', lambda: self.fetch_wikipedia_search()[0]),
            ('Project Gutenberg', lambda: self.fetch_gutenberg()),
            ('Wikisource', lambda: self.fetch_wikisource()[0]),
            ('Wikiquote', lambda: self.fetch_wikiquote()),
            ('Open textbook', lambda: self.fetch_open_textbook()),
        ]
        idx = 0
        print(f"[Fetcher] Started — {len(self.WIKI_CATEGORIES)} categories available")
        while self.running:
            src_name, src_fn = sources[idx % len(sources)]
            try:
                lines = src_fn()
                if lines:
                    self.dataset.add_lines(lines)
                    self.fetched_count += len(lines)
                    print(f"[Fetcher] {src_name}: +{len(lines)} lines (total fetched: {self.fetched_count}) | available: {self.dataset.available()}")
                    self._save_history()
            except Exception as e:
                print(f"[Fetcher] {src_name} error: {e}")
            idx += 1
            time.sleep(self.config.FETCH_DELAY)
    def stop(self):
        self.running = False

# =============================
# CONTINUOUS TRAINER (focused)
# =============================
class ContinuousTrainer:
    def __init__(self, model, tokenizer, config, growth_manager):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.growth_manager = growth_manager
        self.device = config.DEVICE
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2), eps=config.EPS, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = self._create_scheduler()
        self.scaler = torch.amp.GradScaler() if config.ENABLE_MIXED_PRECISION else None
        self.step = 0
        self.epoch = 0
        self.recent_losses = []
        self.loss_history = []
        self.running = True
        self.last_checkpoint_time = time.time()
        # pass reserve_bytes from config to MemoryMonitor so it uses the configured small safety reserve
        reserve = getattr(config, 'RESERVE_BYTES', None)
        self.memory_monitor = MemoryMonitor(config.MAX_MEMORY_GB, reserve_bytes=reserve)
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    def _create_scheduler(self):
        def lr_lambda(step):
            if step < self.config.WARMUP_STEPS:
                return max(step / max(1, self.config.WARMUP_STEPS), 0.01)
            progress = (step - self.config.WARMUP_STEPS) / max(1, 100000 - self.config.WARMUP_STEPS)
            return max(0.5 * (1 + math.cos(math.pi * min(progress, 1.0))), 0.01)
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    def train_step(self, batch):
        self.model.train()
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        if self.scaler and self.config.ENABLE_MIXED_PRECISION:
            with torch.amp.autocast():
                logits = self.model(input_ids)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=self.tokenizer.special_tokens['<pad>'])
            loss = loss / self.config.GRAD_ACCUMULATION_STEPS
            self.scaler.scale(loss).backward()
        else:
            logits = self.model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=self.tokenizer.special_tokens['<pad>'])
            loss = loss / self.config.GRAD_ACCUMULATION_STEPS
            loss.backward()
        return loss.item() * self.config.GRAD_ACCUMULATION_STEPS
    def save_checkpoint(self, is_growth=False):
        """Save model to single model.pt file (overwrites previous)"""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'size': self.config.SIZE,
            'vocab_size': self.config.VOCAB_SIZE,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss_history': self.loss_history[-1000:],
        }
        # Backup dedup state inside model.pt
        if hasattr(self, 'dataset') and self.dataset:
            checkpoint['trained_hashes'] = list(self.dataset.trained_hashes)
            checkpoint['trained_count'] = len(self.dataset.trained_hashes)
        if hasattr(self, 'fetcher') and self.fetcher:
            checkpoint['fetcher_state'] = {
                'fetched_urls': list(self.fetcher.fetched_urls),
                'category_idx': self.fetcher.category_idx,
                'fetched_count': self.fetcher.fetched_count,
            }
        tmp_path = self.config.MODEL_FILE + ".tmp"
        torch.save(checkpoint, tmp_path)
        if os.path.exists(self.config.MODEL_FILE):
            os.remove(self.config.MODEL_FILE)
        os.rename(tmp_path, self.config.MODEL_FILE)
        tag = "growth" if is_growth else f"step {self.step}"
        trained_n = checkpoint.get('trained_count', 0)
        fetched_n = len(checkpoint.get('fetcher_state', {}).get('fetched_urls', []))
        print(f"[Save] model.pt updated ({self.config.SIZE}, {tag}) | Dedup: {trained_n:,} trained, {fetched_n:,} fetched")
    def load_checkpoint(self):
        """Load from single model.pt file"""
        if os.path.exists(self.config.MODEL_FILE):
            try:
                print(f"Loading model from {self.config.MODEL_FILE}...")
                ckpt = torch.load(self.config.MODEL_FILE, map_location='cpu', weights_only=False)
                self.model.load_state_dict(ckpt['model_state_dict'])
                self.model.to(self.device)
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                self.step = ckpt.get('step', 0)
                self.loss_history = ckpt.get('loss_history', [])
                print(f"Resumed: {self.config.SIZE} at step {self.step:,}")
                return True
            except Exception as e:
                print(f"Failed to load model.pt: {e}")
        return False
    def check_growth_criteria(self):
        if len(self.loss_history) < self.config.GROWTH_STABLE_STEPS:
            return False
        recent = self.loss_history[-self.config.GROWTH_STABLE_STEPS:]
        avg = sum(recent) / len(recent)
        if avg < self.config.GROWTH_LOSS_THRESHOLD and all(l < self.config.GROWTH_LOSS_THRESHOLD for l in recent):
            return True
        return False
    def perform_growth(self):
        next_size = self.growth_manager.get_next_size(self.config.SIZE)
        if not next_size:
            print("Already max size")
            return False
        print(f"Growing {self.config.SIZE} → {next_size}")
        self.save_checkpoint(is_growth=True)
        metrics = {'step':self.step,'avg_loss': sum(self.loss_history[-1000:]) / max(1,len(self.loss_history[-1000:])) if self.loss_history else 0,'mem_gb':self.memory_monitor.get_memory_usage_gb()}
        self.growth_manager.log_growth_event(self.config.SIZE, next_size, self.step, self.loss_history[-1] if self.loss_history else 0.0, metrics)
        new_config = ModelConfig(next_size)
        new_config.VOCAB_SIZE = self.config.VOCAB_SIZE
        new_model = AdvancedTransformer(new_config).to(new_config.DEVICE)
        new_model = self.growth_manager.transfer_weights(self.model, new_model)
        self.model = new_model
        self.config = new_config
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=new_config.LEARNING_RATE, betas=(new_config.BETA1, new_config.BETA2), eps=new_config.EPS, weight_decay=new_config.WEIGHT_DECAY)
        self.scheduler = self._create_scheduler()
        self.loss_history = []
        self.recent_losses = []
        self.step = 0
        # after growth ensure memory monitor uses new config values
        reserve = getattr(self.config, 'RESERVE_BYTES', None)
        self.memory_monitor = MemoryMonitor(self.config.MAX_MEMORY_GB, reserve_bytes=reserve)
        self.memory_monitor.force_cleanup()
        print(f"Growth complete: now training {self.config.SIZE}")
        self.save_checkpoint(is_growth=True)
        return True
    def train_forever(self, dataset):
        self.dataset = dataset
        print(f"[Trainer] Starting training on {self.config.DEVICE} ({self.config.DEVICE_TYPE})")
        waiting_logged = False
        while self.running:
            if not self.memory_monitor.is_memory_safe():
                print(f"[Memory] High usage {self.memory_monitor.get_memory_usage_gb():.2f}GB - cleaning")
                self.memory_monitor.force_cleanup()
                if self.config.BATCH_SIZE > 1:
                    self.config.BATCH_SIZE = max(1, self.config.BATCH_SIZE // 2)
                    print(f"[Memory] Reduced batch size to {self.config.BATCH_SIZE}")
                time.sleep(2)
            batch = dataset.get_batch(self.config.BATCH_SIZE)
            if batch is None:
                if not waiting_logged:
                    print("[Trainer] Waiting for data...")
                    waiting_logged = True
                dataset.reload_from_file()
                time.sleep(5)
                continue
            waiting_logged = False
            try:
                loss = self.train_step(batch)
                if (self.step + 1) % self.config.GRAD_ACCUMULATION_STEPS == 0:
                    if self.scaler and self.config.ENABLE_MIXED_PRECISION:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                self.recent_losses.append(loss)
                self.loss_history.append(loss)
                if len(self.recent_losses) > 1000:
                    self.recent_losses = self.recent_losses[-500:]
                self.step += 1
            except RuntimeError as e:
                err = str(e).lower()
                if 'out of memory' in err or 'insufficient' in err or 'could not allocate' in err:
                    self.memory_monitor.force_cleanup()
                    if self.config.BATCH_SIZE > 1:
                        self.config.BATCH_SIZE = max(1, self.config.BATCH_SIZE // 2)
                        print(f"[OOM] Reduced batch size to {self.config.BATCH_SIZE}")
                    time.sleep(3)
                    continue
                print(f"[Trainer] RuntimeError: {e}")
                time.sleep(1)
                continue
            except Exception as e:
                print(f"[Trainer] Error: {e}")
                time.sleep(1)
                continue
            if self.step % 10 == 0:
                lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else 0
                avail = dataset.available()
                avg = sum(self.recent_losses[-50:]) / max(1, len(self.recent_losses[-50:]))
                mem = self.memory_monitor.get_memory_usage_gb()
                print(f"[Train] Step {self.step} | Size: {self.config.SIZE} | Loss: {loss:.4f} | Avg: {avg:.4f} | LR: {lr:.6f} | Data: {avail} | Mem: {mem:.2f}GB | BS: {self.config.BATCH_SIZE}")
            if self.check_growth_criteria():
                print("[Growth] Criteria met — performing growth")
                self.perform_growth()
            if time.time() - self.last_checkpoint_time >= self.config.CHECKPOINT_INTERVAL:
                self.save_checkpoint()
                flushed = dataset.flush_trained()
                if flushed > 0:
                    print(f"[Data] Flushed {flushed} trained lines")
                self.last_checkpoint_time = time.time()
        # final save
        self.save_checkpoint()
        dataset.flush_trained()
        print("[Trainer] Stopped")

# =============================
# MAIN APPLICATION
# =============================
class AIApplication:
    def __init__(self, model_size="10M"):
        self.config = ModelConfig(model_size)
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.fetcher = None
        self.dataset = None
        self.growth_manager = ModelGrowthManager(self.config)
        torch.manual_seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.SEED)
        print(f"\n{'='*60}")
        print("  AUTONOMOUS PROGRESSIVE GROWTH AI - TRAINING ONLY")
        print(f"{'='*60}")
        print(f"  Starting size : {model_size}")
        print(f"  Growth path   : {' → '.join(ModelConfig.GROWTH_PATH)}")
        try:
            gpu_name = torch_directml.device_name(0) if DIRECTML_AVAILABLE else (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
        except Exception:
            gpu_name = 'N/A'
        print(f"  Device        : {self.config.DEVICE} ({self.config.DEVICE_TYPE})")
        print(f"  GPU           : {gpu_name}")
        print(f"  Hidden dim    : {self.config.HIDDEN_DIM}")
        print(f"  Layers        : {self.config.NUM_LAYERS}")
        print(f"  Heads         : {self.config.NUM_HEADS}")
        print(f"  Batch size    : {self.config.BATCH_SIZE}")
        print(f"  Grad accum    : {self.config.GRAD_ACCUMULATION_STEPS}")
        print(f"  Max memory    : {self.config.MAX_MEMORY_GB}GB")
        print(f"  Growth loss   : < {self.config.GROWTH_LOSS_THRESHOLD} for {self.config.GROWTH_STABLE_STEPS} steps")
        print(f"  Checkpoint    : every {self.config.CHECKPOINT_INTERVAL}s")
        print(f"  Data sources  : Wikipedia (expanded categories), Gutenberg, Wikisource, Wikiquote")
        print(f"{'='*60}\n")
    def initialize_tokenizer(self):
        if os.path.exists(self.config.TOKENIZER_FILE):
            tok = BPETokenizer()
            tok.load(self.config.TOKENIZER_FILE)
            if len(tok.vocab) >= 1000:
                self.tokenizer = tok
                self.config.VOCAB_SIZE = len(tok.vocab)
                print(f"Tokenizer ready: {len(tok.vocab)} tokens")
                return
            print("Existing tokenizer too small, will retrain...")
        print("Bootstrapping data for tokenizer training...")
        if not os.path.exists(self.config.DATA_FILE):
            with open(self.config.DATA_FILE, 'w') as f:
                pass
        temp_dataset = ConsumingDataset(self.config.DATA_FILE, None, self.config.MAX_SEQ_LEN)
        temp_fetcher = DataFetcher(temp_dataset, self.config)
        bootstrap = [
            lambda: temp_fetcher.fetch_wikipedia_random(20),
            lambda: temp_fetcher.fetch_gutenberg(),
            lambda: temp_fetcher.fetch_wikipedia_category()[0],
            lambda: temp_fetcher.fetch_wikipedia_vital()[0],
            lambda: temp_fetcher.fetch_wikipedia_search()[0],
            lambda: temp_fetcher.fetch_wikisource()[0],
            lambda: temp_fetcher.fetch_wikiquote(),
            lambda: temp_fetcher.fetch_open_textbook(),
        ]
        idx = 0
        while temp_dataset.total_lines() < self.config.MIN_DATA_LINES:
            print(f"  Data: {temp_dataset.total_lines()} / {self.config.MIN_DATA_LINES} | fetching...")
            try:
                lines = bootstrap[idx % len(bootstrap)]()
                if lines:
                    temp_dataset.add_lines(lines)
            except Exception as e:
                print(f"Bootstrap error: {e}")
            idx += 1
            time.sleep(1)
        texts = []
        with open(self.config.DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i >= 50000:
                    break
                if line.strip():
                    texts.append(line.strip())
        self.tokenizer = BPETokenizer(vocab_size=self.config.VOCAB_SIZE)
        self.tokenizer.train(texts)
        self.tokenizer.save(self.config.TOKENIZER_FILE)
        self.config.VOCAB_SIZE = len(self.tokenizer.vocab)
        self.fetched_urls_bootstrap = temp_fetcher.fetched_urls
    def initialize_model(self):
        if os.path.exists(self.config.MODEL_FILE):
            print(f"Found {self.config.MODEL_FILE}, will load {self.config.SIZE} model in trainer")
        else:
            print(f"No model.pt found, creating new {self.config.SIZE} model")
        self.model = AdvancedTransformer(self.config).to(self.config.DEVICE)
    def prefetch_data(self):
        """Aggressively fetch ~2M lines of training data before training begins."""
        target = self.config.PREFETCH_TARGET_LINES
        print(f"\n{'='*60}")
        print(f"  PREFETCH PHASE")
        print(f"  Target: {target:,} lines of training data")
        print(f"  Sources: Wikipedia, Gutenberg, Wikisource, Wikiquote")
        print(f"  Press Ctrl+C to skip and start training early")
        print(f"{'='*60}\n")

        start_time = time.time()

        existing_lines = 0
        if os.path.exists(self.config.DATA_FILE):
            with open(self.config.DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                existing_lines = sum(1 for line in f if line.strip() and len(line.strip()) > 20)

        if existing_lines >= target:
            print(f"  Already have {existing_lines:,} lines (>= {target:,}). Skipping prefetch.")
            return

        print(f"  Existing data: {existing_lines:,} lines")
        print(f"  Need to fetch: {target - existing_lines:,} more lines\n")

        temp_dataset = ConsumingDataset(self.config.DATA_FILE, None, self.config.MAX_SEQ_LEN)
        fetcher = DataFetcher(temp_dataset, self.config)
        if hasattr(self, 'fetched_urls_bootstrap'):
            fetcher.fetched_urls.update(self.fetched_urls_bootstrap)

        sources = [
            ("Wikipedia (random)", lambda: fetcher.fetch_wikipedia_random(20)),
            ("Wikipedia (category)", lambda: fetcher.fetch_wikipedia_category()[0]),
            ("Wikipedia (vital)", lambda: fetcher.fetch_wikipedia_vital()[0]),
            ("Wikipedia (search)", lambda: fetcher.fetch_wikipedia_search()[0]),
            ("Project Gutenberg", lambda: fetcher.fetch_gutenberg()),
            ("Wikisource", lambda: fetcher.fetch_wikisource()[0]),
            ("Wikiquote", lambda: fetcher.fetch_wikiquote()),
            ("Open textbook", lambda: fetcher.fetch_open_textbook()),
        ]

        round_idx = 0
        last_print = 0

        try:
            while temp_dataset.total_lines() < target:
                src_name, src_fn = sources[round_idx % len(sources)]
                try:
                    lines = src_fn()
                    if lines:
                        temp_dataset.add_lines(lines)
                        fetcher.fetched_count += len(lines)
                except Exception as e:
                    print(f"  Fetch error ({src_name}): {e}")
                    round_idx += 1
                    time.sleep(1)
                    continue

                current = temp_dataset.total_lines()
                elapsed = time.time() - start_time

                if elapsed - last_print >= 5 or round_idx % 10 == 0:
                    rate = current / max(elapsed, 1)
                    remaining = target - current
                    eta = remaining / max(rate, 0.1)
                    pct = current / target * 100
                    bar_len = 30
                    filled = int(bar_len * current / target)
                    bar = '█' * filled + '░' * (bar_len - filled)
                    print(f"  [{bar}] {pct:5.1f}% | {current:>10,} / {target:,} | "
                          f"Rate: {rate:.0f}/s | ETA: {eta/60:.0f}min")
                    last_print = elapsed

                round_idx += 1
                time.sleep(0.5)
        except KeyboardInterrupt:
            print(f"\n  Prefetch interrupted by user. Got {temp_dataset.total_lines():,} lines.")

        elapsed = time.time() - start_time
        final_count = temp_dataset.total_lines()
        print(f"\n  Prefetch complete: {final_count:,} lines in {elapsed/60:.1f} minutes")
        print(f"  Average rate: {final_count/max(elapsed,1):.0f} lines/second\n")

        self.fetched_urls_bootstrap = fetcher.fetched_urls
        fetcher._save_history()

    def run(self):
        self.initialize_tokenizer()
        # Prefetch phase: fetch ~2M lines before training
        self.prefetch_data()
        self.initialize_model()
        self.dataset = ConsumingDataset(self.config.DATA_FILE, self.tokenizer, self.config.MAX_SEQ_LEN)
        self.fetcher = DataFetcher(self.dataset, self.config)
        if hasattr(self, 'fetched_urls_bootstrap'):
            self.fetcher.fetched_urls.update(self.fetched_urls_bootstrap)
        # Restore dedup state from model.pt if JSON files are incomplete/missing
        if os.path.exists(self.config.MODEL_FILE):
            try:
                ckpt = torch.load(self.config.MODEL_FILE, map_location='cpu', weights_only=False)
                backup_hashes = set(ckpt.get('trained_hashes', []))
                if backup_hashes and len(backup_hashes) > len(self.dataset.trained_hashes):
                    added = len(backup_hashes) - len(self.dataset.trained_hashes)
                    self.dataset.trained_hashes.update(backup_hashes)
                    self.dataset._save_trained_hashes()
                    self.dataset._load_lines()
                    print(f"[Dedup] Restored {added:,} trained hashes from model.pt backup")
                fs = ckpt.get('fetcher_state', {})
                if fs:
                    backup_urls = set(fs.get('fetched_urls', []))
                    if len(backup_urls) > len(self.fetcher.fetched_urls):
                        added = len(backup_urls) - len(self.fetcher.fetched_urls)
                        self.fetcher.fetched_urls.update(backup_urls)
                        self.fetcher.category_idx = max(self.fetcher.category_idx, fs.get('category_idx', 0))
                        self.fetcher.fetched_count = max(self.fetcher.fetched_count, fs.get('fetched_count', 0))
                        self.fetcher._save_history()
                        print(f"[Dedup] Restored {added:,} fetched URLs from model.pt backup")
                del ckpt
                gc.collect()
            except Exception as e:
                print(f"[Dedup] Could not restore from model.pt: {e}")
        fetcher_thread = threading.Thread(target=self.fetcher.run_forever, daemon=True)
        fetcher_thread.start()
        self.trainer = ContinuousTrainer(self.model, self.tokenizer, self.config, self.growth_manager)
        self.trainer.fetcher = self.fetcher
        self.trainer.load_checkpoint()
        try:
            self.trainer.train_forever(self.dataset)
        except KeyboardInterrupt:
            print("\n\nShutting down gracefully...")
            self.trainer.running = False
            self.fetcher.running = False
            time.sleep(2)
            self.trainer.save_checkpoint()
            self.dataset.flush_trained()
            print("Checkpoint saved. You can resume anytime with: python train.py")
            print("Goodbye!")
# ENTRY POINT
def detect_saved_size(model_file="model.pt"):
    """Auto-detect model size from existing model.pt on restart."""
    if os.path.exists(model_file):
        try:
            ckpt = torch.load(model_file, map_location='cpu', weights_only=False)
            size = ckpt.get('size', '10M')
            step = ckpt.get('step', 0)
            print(f"[Resume] Found model.pt — Size: {size}, Step: {step:,}")
            return size
        except Exception as e:
            print(f"[Resume] Could not read model.pt: {e}")
    return None

if __name__ == '__main__':
    saved_size = detect_saved_size()
    if saved_size:
        model_size = saved_size
    else:
        model_size = '10M'
        if len(sys.argv) > 1 and sys.argv[1] in ModelConfig.GROWTH_PATH:
            model_size = sys.argv[1]
    
    print(f"\n  Model: {model_size} ({'Resuming' if saved_size else 'Fresh start'})")
    print(f"  Saves to: model.pt (auto-overwrite)\n")
    
    app = AIApplication(model_size)
    app.run()