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
        self.GROWTH_LOG_FILE = "growth_log.json"

        # checkpoint pruning: keep this many most recent checkpoints
        self.CHECKPOINT_INTERVAL = 300
        self.CHECKPOINT_MAX_KEEP = 3
        self.FLUSH_INTERVAL = 50
        self.MIN_DATA_LINES = 3000
        self.FETCH_BATCH = 10
        self.FETCH_DELAY = 5

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
            with open(self.TRAINED_HASHES_FILE, 'w') as f:
                json.dump({'hashes': list(self.trained_hashes), 'count': len(self.trained_hashes)}, f)
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
        self._load_lines()\n# =============================\n# DATA FETCHER (expanded categories)\n# =============================\nclass DataFetcher: