#!/usr/bin/env python3
"""
Autonomous Progressive Growth AI Training System
COMPLETE MERGED VERSION - All original code + all enhancements

Features:
- Continuously fetches data from Wikipedia & 250+ knowledge domains
- Progressive Growth: 10M → 50M → 100M → 200M → 350M → 500M
- Smart checkpointing: Never overwrites, timestamped saves
- Memory optimized: 5GB RAM with intelligent batching
- API monitoring: localhost:8080/api/status
- Multimodal: Image analysis support
- All original Wikipedia/Gutenberg/Wikisource fetching
- All original transformer architecture
- All original training logic
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
import subprocess
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# DirectML GPU support
try:
    import torch_directml
    DIRECTML_AVAILABLE = True
    print(f"[GPU] DirectML available — {torch_directml.device_count()} device(s) detected")
    for i in range(torch_directml.device_count()):
        print(f"  GPU {i}: {torch_directml.device_name(i)}")
except ImportError:
    DIRECTML_AVAILABLE = False
    print("[Warning] torch-directml not installed — falling back to CPU")
    print("  Install with: python -m pip install torch-directml")

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
# CONFIGURATION (ENHANCED)
# =============================
class ModelConfig:
    # Batch sizes tuned for GPU (DirectML) — will auto-reduce on OOM
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
        self.NUM_KV_HEADS = preset["num_heads"] // 2
        self.FFN_DIM = preset["hidden_dim"] * preset["ffn_mult"]
        self.HEAD_DIM = self.HIDDEN_DIM // self.NUM_HEADS

        self.VOCAB_SIZE = 32000
        self.MAX_SEQ_LEN = 512
        self.DROPOUT = 0.1
        self.BIAS = False
        self.ROPE_THETA = 10000.0

        # Memory-optimized batch settings
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

        self.CHECKPOINT_INTERVAL = 300  # 5 minutes
        self.FLUSH_INTERVAL = 50
        self.MIN_DATA_LINES = 3000
        self.CHAT_PORT = 8080
        self.FETCH_BATCH = 10
        self.FETCH_DELAY = 5
        self.PREFETCH_TARGET_LINES = 2_000_000
        
        # Growth criteria
        self.GROWTH_LOSS_THRESHOLD = 3.0
        self.GROWTH_STABLE_STEPS = 1000

        # Device selection: DirectML (GPU) > CUDA > CPU
        if DIRECTML_AVAILABLE:
            self.DEVICE = torch_directml.device()
            self.DEVICE_TYPE = "directml"
        elif torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
            self.DEVICE_TYPE = "cuda"
        else:
            self.DEVICE = torch.device("cpu")
            self.DEVICE_TYPE = "cpu"

        self.SEED = 42
        
        # Memory management — auto-detect, reserve 400MB for OS
        if PSUTIL_AVAILABLE:
            total_gb = psutil.virtual_memory().total / (1024 ** 3)
            self.MAX_MEMORY_GB = round(total_gb - 0.3, 1)
        else:
            self.MAX_MEMORY_GB = 5.6
        # Mixed precision only works reliably with CUDA
        self.ENABLE_MIXED_PRECISION = (self.DEVICE_TYPE == "cuda")

 BPETokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3,
            '<sep>': 4, '<cls>': 5, '<mask>': 6
        }
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
                word_counts[word_with_end] = word_counts.get(word_with_end, 0) + 1

        vocab = set()
        for word in word_counts:
            for char in word.split():
                vocab.add(char)
        for token in self.special_tokens:
            vocab.add(token)

        print(f"Initial vocabulary size: {len(vocab)}")
        num_merges = self.vocab_size - len(vocab)
        print(f"Learning {num_merges} BPE merges...")
        word_items = list(word_counts.items())

        for i in range(num_merges):
            if i % 1000 == 0:
                print(f"  Merge {i}/{num_merges}")
            pair_counts = {}
            for word, count in word_items:
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pair = (symbols[j], symbols[j+1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + count
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
                    if j < len(symbols) - 1 and (symbols[j], symbols[j+1]) == best_pair:
                        new_symbols.append(merged)
                        j += 2
                    else:
                        new_symbols.append(symbols[j])
                        j += 1
                new_items.append((' '.join(new_symbols), count))
            word_items = new_items

        sorted_vocab = sorted(list(vocab))
        self.vocab = {symbol: idx for idx, symbol in enumerate(sorted_vocab)}
        for token, idx in self.special_tokens.items():
            if token in self.vocab:
                old_idx = self.vocab[token]
                self.vocab[token] = idx
                for s, i in list(self.vocab.items()):
                    if i == idx and s != token:
                        self.vocab[s] = old_idx
                        break
            else:
                self.vocab[token] = idx
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
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
                for i in range(len(word_symbols) - 1):
                    pair = (word_symbols[i], word_symbols[i+1])
                    if pair in self.merges:
                        score = self.merges[pair]
                        if score > best_score:
                            best_score = score
                            best_pair = pair
                            best_idx = i
                if best_pair is not None:
                    word_symbols = (word_symbols[:best_idx] +
                                    [''.join(best_pair)] +
                                    word_symbols[best_idx+2:])
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
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                if token == '</w>':
                    text += ' '
                else:
                    text += token
        return text.strip()

    def save(self, path):
        str_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        data = {
            'vocab': self.vocab, 'merges': str_merges,
            'special_tokens': self.special_tokens, 'vocab_size': self.vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {path}")

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.vocab_size = data['vocab_size']
        self.special_tokens = data['special_tokens']
        self.merges = {}
        for k_str, v in data['merges'].items():
            parts = k_str.split(',')
            self.merges[(parts[0], parts[1])] = v
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Tokenizer loaded from {path}")

# =============================

# =============================
# MEMORY MONITOR (NEW)
# =============================
class MemoryMonitor:
    RESERVE_BYTES = 300 * 1024 * 1024  # 300MB reserved for OS

    def __init__(self, max_memory_gb=5.0):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
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
            return psutil.virtual_memory().available > self.RESERVE_BYTES
        return self.get_memory_usage() < self.max_memory_bytes * 0.9

    def can_boost(self):
        """Returns True if there's any free RAM beyond 300MB OS reserve."""
        if PSUTIL_AVAILABLE:
            free = psutil.virtual_memory().available
            return free > self.RESERVE_BYTES
        return False

    def force_cleanup(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if DIRECTML_AVAILABLE:
            gc.collect()

# =============================
# MODEL GROWTH MANAGER (NEW)
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
            except:
                return []
        return []
    
    def save_growth_log(self):
        with open(self.growth_log_file, 'w') as f:
            json.dump(self.growth_history, f, indent=2)
    
    def log_growth_event(self, from_size, to_size, step, loss, metrics):
        event = {
            'timestamp': datetime.now().isoformat(),
            'from_size': from_size,
            'to_size': to_size,
            'training_step': step,
            'loss_at_growth': loss,
            'metrics': metrics
        }
        self.growth_history.append(event)
        self.save_growth_log()
        print(f"\n{'='*60}")
        print(f"  GROWTH EVENT LOGGED")
        print(f"  {from_size} → {to_size}")
        print(f"  Step: {step}, Loss: {loss:.4f}")
        print(f"{'='*60}\n")
    
    def get_next_size(self, current_size):
        growth_path = ModelConfig.GROWTH_PATH
        if current_size not in growth_path:
            return None
        idx = growth_path.index(current_size)
        if idx >= len(growth_path) - 1:
            return None
        return growth_path[idx + 1]
    
    def transfer_weights(self, old_model, new_model):
        """Smart weight transfer preserving learned knowledge"""
        print("Transferring weights to larger model...")
        
        old_state = old_model.state_dict()
        new_state = new_model.state_dict()
        
        # Transfer embeddings
        old_vocab_size = old_state['token_embedding.weight'].shape[0]
        new_vocab_size = new_state['token_embedding.weight'].shape[0]
        min_vocab = min(old_vocab_size, new_vocab_size)
        new_state['token_embedding.weight'][:min_vocab] = old_state['token_embedding.weight'][:min_vocab].clone()
        
        # Transfer layer weights
        old_layers = len([k for k in old_state.keys() if k.startswith('layers.')])
        new_layers = len([k for k in new_state.keys() if k.startswith('layers.')])
        
        for old_layer_idx in range(old_layers):
            for param_name in old_state.keys():
                if f'layers.{old_layer_idx}.' in param_name:
                    if param_name not in new_state:
                        continue
                    
                    old_tensor = old_state[param_name]
                    new_tensor = new_state[param_name]
                    
                    if old_tensor.shape == new_tensor.shape:
                        new_state[param_name] = old_tensor.clone()
                    elif len(old_tensor.shape) == 2:
                        min_dim0 = min(old_tensor.shape[0], new_tensor.shape[0])
                        min_dim1 = min(old_tensor.shape[1], new_tensor.shape[1])
                        new_state[param_name][:min_dim0, :min_dim1] = old_tensor[:min_dim0, :min_dim1].clone()
                    elif len(old_tensor.shape) == 1:
                        min_dim = min(old_tensor.shape[0], new_tensor.shape[0])
                        new_state[param_name][:min_dim] = old_tensor[:min_dim].clone()
        
        # Transfer final norm and output
        for key in ['norm.weight', 'norm.scale', 'output.weight']:
            if key in old_state and key in new_state:
                old_t = old_state[key]
                new_t = new_state[key]
                if len(old_t.shape) == 1:
                    min_dim = min(old_t.shape[0], new_t.shape[0])
                    new_state[key][:min_dim] = old_t[:min_dim].clone()
                elif len(old_t.shape) == 2:
                    min_d0 = min(old_t.shape[0], new_t.shape[0])
                    min_d1 = min(old_t.shape[1], new_t.shape[1])
                    new_state[key][:min_d0, :min_d1] = old_t[:min_d0, :min_d1].clone()
        
        new_model.load_state_dict(new_state)
        print(f"Weight transfer complete. {old_layers} layers → {new_layers} layers")
        return new_model

# TRANSFORMER COMPONENTS
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
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
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
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
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
        cos, sin = self.rotary(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=2)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1
            )
            attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
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

    def generate(self, tokenizer, prompt, max_length=100, temperature=0.8, top_k=40):
        self.eval()
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids]).to(self.config.DEVICE)
        generated = []
        with torch.no_grad():
            for _ in range(max_length):
                # Truncate to max seq len
                if input_ids.shape[1] > self.config.MAX_SEQ_LEN:
                    input_ids = input_ids[:, -self.config.MAX_SEQ_LEN:]
                logits = self(input_ids)[:, -1, :]
                logits = logits / temperature
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    probs = F.softmax(top_k_logits, dim=-1)
                    next_token = top_k_indices.gather(-1, torch.multinomial(probs, num_samples=1))
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                generated.append(next_token.item())
                if next_token.item() == tokenizer.special_tokens['<eos>']:
                    break
        self.train()
        full_ids = input_ids[0].tolist()
        response = tokenizer.decode(full_ids)
        if prompt in response:
            response = response.split(prompt)[-1].strip()
        return response

# =============================
# CONSUMING DATASET
# =============================
class ConsumingDataset:
    """
    Data buffer: fetcher adds lines → trainer consumes them → checkpoint deletes trained lines.
    trained_data.json tracks all hashes ever trained so data is never re-fetched or re-trained.
    """

    TRAINED_HASHES_FILE = "trained_data.json"

    def __init__(self, file_path, tokenizer, max_length=512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lock = threading.Lock()
        self.lines = []
        self.pending_hashes = set()       # hashes in current data.txt (not yet trained)
        self.trained_hashes = set()       # hashes already trained & checkpointed (persistent)
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
        # Rewrite file without already-trained lines
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
        """Called after checkpoint: mark consumed lines as trained, delete from data.txt."""
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
        """Just reset position without marking as trained (used between checkpoints)."""
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
# DATA FETCHER — BROAD KNOWLEDGE
# =============================
class DataFetcher:
    # ---- Wikipedia: 250+ categories across EVERY knowledge domain ----
    WIKI_CATEGORIES = [
        # STEM — Core Sciences
        "Science", "Technology", "Mathematics", "Physics", "Chemistry",
        "Biology", "Computer_science", "Engineering", "Astronomy", "Geology",
        "Genetics", "Neuroscience", "Robotics", "Quantum_mechanics",
        "Organic_chemistry", "Calculus", "Linear_algebra", "Statistics",
        "Machine_learning", "Artificial_intelligence", "Cryptography",
        "Thermodynamics", "Electromagnetism", "Optics", "Nuclear_physics",
        "Molecular_biology", "Biochemistry", "Microbiology", "Virology",
        "Ecology", "Oceanography", "Meteorology", "Paleontology",
        "Materials_science", "Nanotechnology", "Biotechnology",
        "Aerospace_engineering", "Civil_engineering", "Electrical_engineering",
        # Programming & Software
        "Programming_languages", "Python_(programming_language)",
        "JavaScript", "Java_(programming_language)", "C_(programming_language)",
        "Rust_(programming_language)", "Go_(programming_language)",
        "Web_development", "Mobile_app_development", "DevOps",
        "Version_control", "Software_testing", "Agile_software_development",
        "Open-source_software", "Linux", "Algorithms", "Data_structures",
        "Object-oriented_programming", "Functional_programming",
        "API", "Microservices", "Containerization", "Kubernetes",
        # Medicine & Health
        "Medicine", "Surgery", "Pharmacology", "Epidemiology", "Anatomy",
        "Cardiology", "Oncology", "Psychiatry", "Immunology", "Nutrition",
        "Public_health", "Dentistry", "Veterinary_medicine",
        "Mental_health", "Yoga", "Meditation", "Alternative_medicine",
        "Ayurveda", "Traditional_Chinese_medicine", "Homeopathy",
        # Finance & Business
        "Finance", "Stock_market", "Investment", "Banking",
        "Venture_capital", "Private_equity", "Hedge_fund",
        "Financial_analysis", "Risk_management", "Insurance",
        "Real_estate", "Mortgage", "Personal_finance", "Retirement_planning",
        "Cryptocurrency", "Decentralized_finance", "Blockchain",
        "Accounting", "Taxation", "Audit", "Financial_regulation",
        "Business", "Management", "Marketing", "Entrepreneurship",
        "Startup_company", "Supply_chain_management", "E-commerce",
        "Business_model", "Corporate_governance", "Mergers_and_acquisitions",
        # Humanities & Philosophy
        "Philosophy", "Ethics", "Logic", "Metaphysics", "Epistemology",
        "Existentialism", "Stoicism", "Buddhism", "Hinduism", "Taoism",
        "Confucianism", "Zen", "Mysticism", "Consciousness",
        "History", "Ancient_history", "Medieval_history", "Modern_history",
        "World_War_I", "World_War_II", "Cold_War", "Renaissance",
        "History_of_science", "History_of_mathematics",
        "Literature", "Poetry", "Drama", "Fiction", "Non-fiction",
        "Science_fiction", "Fantasy_literature", "Mystery_fiction",
        "Linguistics", "Phonetics", "Semantics", "Grammar",
        "Art", "Painting", "Sculpture", "Photography", "Film",
        "Animation", "Graphic_design", "Digital_art",
        "Music", "Classical_music", "Jazz", "Rock_music", "Hip_hop",
        "Electronic_music", "Music_theory", "Music_production",
        "Architecture", "Urban_planning", "Interior_design",
        # Spirituality & Religion
        "Spirituality", "Religion", "Christianity", "Islam", "Judaism",
        "Hinduism", "Buddhism", "Sikhism", "Jainism", "Shinto",
        "Mythology", "Greek_mythology", "Norse_mythology", "Egyptian_mythology",
        "Hindu_mythology", "Celtic_mythology", "Chinese_mythology",
        "Folklore", "Supernatural", "Occult", "Astrology",
        "Meditation", "Mindfulness", "Prayer", "Pilgrimage",
        "Reincarnation", "Karma", "Nirvana", "Enlightenment_(spiritual)",
        # Social Sciences
        "Psychology", "Cognitive_science", "Sociology", "Anthropology",
        "Economics", "Microeconomics", "Macroeconomics",
        "Political_science", "International_relations", "Diplomacy",
        "Law", "Constitutional_law", "International_law", "Criminal_law",
        "Education", "Pedagogy", "Early_childhood_education",
        "Criminology", "Forensic_science", "Social_work",
        # Geography & Earth
        "Geography", "Countries", "Continents", "Mountains", "Rivers",
        "Climate", "Climate_change", "Environmental_science",
        "Agriculture", "Forestry", "Sustainable_development",
        # Real World & Daily Life
        "Cooking", "Cuisine", "Baking", "Food_science",
        "Nutrition", "Diet_(nutrition)", "Veganism", "Food_preservation",
        "Fashion", "Textile", "Clothing", "Cosmetics",
        "Home_improvement", "Gardening", "Interior_design",
        "Parenting", "Child_development", "Family",
        "Travel", "Tourism", "Hotel", "Airline",
        "Pets", "Dog", "Cat", "Aquarium",
        "Personal_development", "Self-help", "Time_management",
        "Leadership", "Public_speaking", "Negotiation",
        "Relationships", "Marriage", "Friendship",
        # Sports & Fitness
        "Sports", "Olympics", "Football", "Basketball", "Cricket",
        "Tennis", "Swimming", "Athletics", "Martial_arts",
        "Chess", "Esports", "Video_game", "Board_game",
        "Physical_fitness", "Bodybuilding", "Running", "Cycling",
        "Yoga", "Pilates", "CrossFit",
        # Technology & Modern Life
        "Smartphone", "Social_media", "Streaming_media",
        "Electric_vehicle", "Self-driving_car", "Drone",
        "Internet_of_things", "Smart_home", "Wearable_technology",
        "Virtual_reality", "Augmented_reality", "Metaverse",
        "3D_printing", "Quantum_computing",
        "Renewable_energy", "Solar_energy", "Wind_power",
        "Nuclear_energy", "Petroleum", "Natural_gas",
        "Transportation", "Aviation", "Automotive_industry",
        "Telecommunications", "Space_exploration", "Satellite",
        # Journalism & Media
        "Journalism", "Mass_media", "Television", "Radio",
        "Newspaper", "Magazine", "Podcast", "Blog",
        "Advertising", "Public_relations", "Propaganda",
    ]

    # ---- Project Gutenberg: 80 classic books across genres ----
    GUTENBERG_BOOKS = [
        # Fiction classics
        1342, 84, 2701, 11, 1661, 98, 1232, 174, 2600, 76, 5200, 1400,
        345, 1952, 4300, 46, 219, 1260, 996, 74, 43, 514, 1184, 768,
        # Philosophy & Non-fiction
        1497, 244, 3600, 4280, 1228, 3207, 815, 7370,
        # Science & Knowledge
        36, 4217, 20203, 28054, 5001, 14264, 29728,
        # History & Politics
        3, 10, 3076, 1404, 2680, 4363, 6130, 16653,
        # Poetry & Drama
        1112, 2265, 2267, 23042, 1041, 2199, 2264, 100,
        # Adventure & Exploration
        120, 164, 215, 829, 1257, 27827, 35, 1250, 2148, 2500,
        # World literature
        2000, 2197, 7849, 7178, 526, 600, 1399, 8800, 4363, 28885,
        # More classics
        158, 2591, 30254, 2852, 55, 161, 135, 28233,
    ]

    # ---- Wikisource documents: public domain primary sources ----
    WIKISOURCE_DOCS = [
        "United_States_Constitution",
        "Declaration_of_Independence_(United_States)",
        "Magna_Carta",
        "Universal_Declaration_of_Human_Rights",
        "Communist_Manifesto",
        "Gettysburg_Address",
        "I_Have_a_Dream",
        "Emancipation_Proclamation",
        "The_Art_of_War_(Sun_Tzu)",
        "The_Republic_(Plato)",
        "Wealth_of_Nations",
        "On_the_Origin_of_Species_(1859)",
        "A_Vindication_of_the_Rights_of_Woman",
        "The_Prince_(Machiavelli)",
        "Leviathan_(Hobbes)",
        "Two_Treatises_of_Government",
        "The_Federalist_Papers",
        "Treaty_of_Versailles",
        "Charter_of_the_United_Nations",
        "Geneva_Conventions",
    ]

    # ---- Wikipedia "Vital Articles" — most important across ALL domains ----
    VITAL_TOPICS = [
        # Foundational
        "Universe", "Earth", "Life", "Human", "Society", "Culture",
        "Mathematics", "Science", "Technology", "Health", "Philosophy",
        "Religion", "Geography", "History", "Art", "Music", "Literature",
        "Economics", "Politics", "Law", "Education", "Language",
        # Science fundamentals
        "Computer", "Internet", "Electricity", "Atom", "DNA",
        "Evolution", "Gravity", "Light", "Water", "Energy",
        "Algorithm", "Calculus", "Geometry", "Algebra", "Logic",
        "Relativity", "Quantum_mechanics", "Thermodynamics",
        "Plate_tectonics", "Big_Bang", "Solar_System", "Galaxy",
        "Photosynthesis", "Cell_(biology)", "Virus", "Bacteria",
        "Black_hole", "Neutron_star", "Supernova", "Dark_matter",
        "Periodic_table", "Chemical_bond", "Catalysis", "Polymer",
        # Medicine & Body
        "Brain", "Heart", "Cancer", "Vaccine", "Antibiotic",
        "Surgery", "Genome", "Protein", "Blood", "Immune_system",
        "Nervous_system", "Digestive_system", "Respiratory_system",
        "Pandemic", "HIV/AIDS", "Diabetes", "Alzheimer's_disease",
        # Tech & Computing
        "Artificial_intelligence", "Machine_learning", "Neural_network",
        "Blockchain", "Cryptocurrency", "Robotics", "Biotechnology",
        "Semiconductor", "Transistor", "Integrated_circuit",
        "Operating_system", "Programming_language", "World_Wide_Web",
        "Search_engine", "Social_media", "Smartphone",
        "Cloud_computing", "Big_data", "Internet_of_things",
        "Cybersecurity", "Encryption", "Open-source_software",
        "Git_(software)", "Docker_(software)", "Linux_kernel",
        # Finance & Economics
        "Stock_market", "Inflation", "Gross_domestic_product",
        "Central_bank", "Federal_Reserve", "Wall_Street",
        "Compound_interest", "Bond_(finance)", "Mutual_fund",
        "Supply_and_demand", "Free_market", "Trade",
        "Tax", "Budget", "Debt", "Credit",
        "Bitcoin", "Ethereum", "Financial_crisis_of_2007-2008",
        # Politics & Society
        "Democracy", "Capitalism", "Socialism", "War", "Peace",
        "United_Nations", "European_Union", "NATO",
        "World_Trade_Organization", "World_population", "Globalization",
        "Human_rights", "Freedom_of_speech", "Civil_rights_movement",
        "Feminism", "Environmentalism", "Poverty", "Inequality",
        # History milestones
        "Agriculture", "Industrial_Revolution", "Renaissance",
        "Ancient_Egypt", "Roman_Empire", "Ancient_Greece",
        "Silk_Road", "Colonialism", "Decolonization",
        "French_Revolution", "American_Revolution",
        "Nuclear_weapon", "Space_exploration", "Moon_landing",
        "Printing_press", "Telescope", "Microscope", "Steam_engine",
        # Spirituality & Wisdom
        "Meditation", "Yoga", "Karma", "Dharma", "Nirvana",
        "Bhagavad_Gita", "Quran", "Bible", "Torah", "Tao_Te_Ching",
        "Zen", "Sufism", "Kabbalah", "Shamanism",
        "Consciousness", "Free_will", "Soul", "Afterlife",
        "Stoicism", "Epicureanism", "Existentialism", "Nihilism",
        # Nature & Earth
        "Continental_drift", "Volcano", "Earthquake", "Tsunami",
        "Tornado", "Hurricane", "Glacier", "Coral_reef", "Rainforest",
        "Desert", "Ocean", "River", "Mountain",
        "Climate_change", "Greenhouse_effect", "Deforestation",
        "Endangered_species", "Biodiversity", "Conservation",
        # Culture & Arts
        "Painting", "Sculpture", "Dance", "Theater", "Opera",
        "Novel", "Short_story", "Essay", "Biography", "Cinema",
        "Hip_hop", "Rock_and_roll", "Blues", "Reggae",
        "Olympic_Games", "FIFA_World_Cup", "Cricket", "Tennis",
        # Daily life
        "Cooking", "Nutrition", "Exercise", "Sleep", "Stress",
        "Marriage", "Parenting", "Friendship", "Love",
        "Money", "Career", "Retirement", "Volunteering",
    ]

    # ---- Open textbook / course material URLs ----
    OPEN_TEXTBOOK_URLS = [
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "https://www.gutenberg.org/cache/epub/10/pg10.txt",       # King James Bible
        "https://www.gutenberg.org/cache/epub/3200/pg3200.txt",   # Principia (Newton)
        "https://www.gutenberg.org/cache/epub/4300/pg4300.txt",   # Ulysses
        "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",   # Pride & Prejudice
    ]

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.running = True
        self.fetched_count = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AutonomousAITrainer/1.0 (Educational Research; Python/requests)'
        })
        self.category_idx = 0
        self.gutenberg_idx = 0
        self.vital_idx = 0
        self.wikisource_idx = 0
        self.textbook_idx = 0
        self.fetched_urls = set()
        self.history_file = Path("fetched_history.json")
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.fetched_urls = set(data.get('fetched_urls', []))
                    self.category_idx = data.get('category_idx', 0)
                    self.gutenberg_idx = data.get('gutenberg_idx', 0)
                    self.vital_idx = data.get('vital_idx', 0)
                    self.wikisource_idx = data.get('wikisource_idx', 0)
                    self.textbook_idx = data.get('textbook_idx', 0)
                    self.fetched_count = data.get('fetched_count', 0)
                    print(f"[Fetcher] Resumed — {len(self.fetched_urls)} sources already fetched")
                else:
                    self.fetched_urls = set(data)
            except Exception:
                self.fetched_urls = set()

    def _save_history(self):
        try:
            data = {
                'fetched_urls': list(self.fetched_urls),
                'category_idx': self.category_idx,
                'gutenberg_idx': self.gutenberg_idx,
                'vital_idx': self.vital_idx,
                'wikisource_idx': self.wikisource_idx,
                'textbook_idx': self.textbook_idx,
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
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\{[^}]*\}', '', text)
        text = re.sub(r'==+\s*See also\s*==+.*', '', text, flags=re.DOTALL)
        text = re.sub(r'==+\s*References\s*==+.*', '', text, flags=re.DOTALL)
        text = re.sub(r'==+\s*External links\s*==+.*', '', text, flags=re.DOTALL)
        text = re.sub(r'==+\s*Notes\s*==+.*', '', text, flags=re.DOTALL)
        text = re.sub(r'==+\s*Bibliography\s*==+.*', '', text, flags=re.DOTALL)
        text = re.sub(r'==+[^=]+=+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        lines = []
        for para in text.split('\n'):
            para = para.strip()
            if len(para) > 50 and sum(c.isalpha() for c in para) > len(para) * 0.5:
                if len(para) > 500:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    chunk = ""
                    for s in sentences:
                        if len(chunk) + len(s) > 400:
                            if len(chunk) > 50:
                                lines.append(chunk.strip())
                            chunk = s
                        else:
                            chunk += " " + s if chunk else s
                    if len(chunk) > 50:
                        lines.append(chunk.strip())
                else:
                    lines.append(para)
        return lines

    def _fetch_wiki_article(self, title):
        """Fetch a single Wikipedia article by exact title."""
        if title in self.fetched_urls:
            return []
        try:
            resp = self.session.get(
                'https://en.wikipedia.org/w/api.php',
                params={
                    'action': 'query', 'titles': title,
                    'prop': 'extracts', 'explaintext': True, 'format': 'json'
                }, timeout=15
            )
            pages = resp.json()['query']['pages']
            for pid, page in pages.items():
                if 'extract' in page and len(page['extract']) > 200:
                    lines = self._clean_text(page['extract'])
                    self.fetched_urls.add(title)
                    return lines
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # SOURCE 1: Wikipedia Random Articles (infinite supply)
    # ------------------------------------------------------------------
    def fetch_wikipedia_random(self, count=10):
        all_lines = []
        try:
            resp = self.session.get(
                'https://en.wikipedia.org/w/api.php',
                params={
                    'action': 'query', 'list': 'random',
                    'rnnamespace': 0, 'rnlimit': count, 'format': 'json'
                }, timeout=15
            )
            titles = [a['title'] for a in resp.json()['query']['random']]
            for title in titles:
                lines = self._fetch_wiki_article(title)
                all_lines.extend(lines)
                if lines:
                    time.sleep(0.3)
        except Exception as e:
            print(f"[Fetcher] Wikipedia random error: {e}")
        return all_lines

    # ------------------------------------------------------------------
    # SOURCE 2: Wikipedia by Category (structured knowledge)
    # ------------------------------------------------------------------
    def fetch_wikipedia_category(self):
        category = self.WIKI_CATEGORIES[self.category_idx % len(self.WIKI_CATEGORIES)]
        self.category_idx += 1
        all_lines = []
        try:
            resp = self.session.get(
                'https://en.wikipedia.org/w/api.php',
                params={
                    'action': 'query', 'list': 'categorymembers',
                    'cmtitle': f'Category:{category}', 'cmlimit': 15,
                    'cmtype': 'page', 'format': 'json'
                }, timeout=15
            )
            members = resp.json().get('query', {}).get('categorymembers', [])
            for member in members:
                lines = self._fetch_wiki_article(member['title'])
                all_lines.extend(lines)
                if lines:
                    time.sleep(0.3)
        except Exception as e:
            print(f"[Fetcher] Wikipedia category '{category}' error: {e}")
        return all_lines, category

    # ------------------------------------------------------------------
    # SOURCE 3: Wikipedia Vital/Important Topics (core knowledge)
    # ------------------------------------------------------------------
    def fetch_wikipedia_vital(self):
        if self.vital_idx >= len(self.VITAL_TOPICS):
            self.vital_idx = 0
        topic = self.VITAL_TOPICS[self.vital_idx]
        self.vital_idx += 1
        lines = self._fetch_wiki_article(topic)
        return lines, topic

    # ------------------------------------------------------------------
    # SOURCE 4: Wikipedia Search — topical deep-dives
    # ------------------------------------------------------------------
    SEARCH_QUERIES = [
        # Science & Physics
        "quantum computing algorithms", "general relativity spacetime",
        "photosynthesis light reactions", "plate tectonics continental drift",
        "Black hole event horizon", "Superconductivity materials",
        "Nuclear fusion reactor", "Particle physics standard model",
        "Cosmology dark energy", "Exoplanet detection methods",
        "Fourier transform signal processing", "Chaos theory butterfly effect",
        "Fibonacci sequence golden ratio", "Prime numbers distribution",
        # Biology & Medicine
        "neural network backpropagation", "DNA replication process",
        "Protein folding structure", "CRISPR gene editing",
        "Vaccine immune response", "Cancer immunotherapy treatment",
        "Antibiotic resistance bacteria", "Stem cell research therapy",
        "Epigenetics gene expression", "Microbiome gut health",
        "Cognitive behavioral therapy", "Memory formation hippocampus",
        # Programming & CS
        "Python programming tutorial data types",
        "JavaScript async await promises", "React component lifecycle",
        "REST API design best practices", "SQL database query optimization",
        "Git version control branching", "Docker container orchestration",
        "Linux command line shell scripting", "TCP IP protocol networking",
        "Object oriented programming principles", "Functional programming concepts",
        "Data structures algorithms complexity", "Binary search tree implementation",
        "Machine learning model training", "Deep learning convolutional neural network",
        "Natural language processing transformer", "Computer vision object detection",
        "Encryption RSA algorithm", "Operating system kernel design",
        "Compiler design parsing", "Database normalization SQL",
        "Kubernetes microservices deployment", "CI CD pipeline automation",
        "WebSocket real time communication", "GraphQL API schema design",
        # Finance & Business
        "stock market technical analysis candlestick",
        "compound interest investment strategy",
        "venture capital startup funding rounds",
        "cryptocurrency DeFi yield farming",
        "financial statements balance sheet income",
        "options trading derivatives hedging",
        "real estate investment property valuation",
        "supply chain logistics optimization",
        "international trade economics tariffs",
        "central bank monetary policy interest rates",
        "personal finance budgeting savings",
        "tax planning deductions strategies",
        "portfolio diversification risk management",
        "IPO initial public offering process",
        # Spirituality & Philosophy
        "meditation mindfulness techniques benefits",
        "yoga chakra kundalini awakening",
        "Buddhist philosophy four noble truths",
        "Hindu Vedanta Upanishads Brahman Atman",
        "Taoism wu wei Tao Te Ching",
        "Stoic philosophy Marcus Aurelius Seneca",
        "consciousness hard problem qualia",
        "free will determinism compatibilism",
        "karma reincarnation samsara moksha",
        "Sufi mysticism Rumi poetry spiritual",
        "Zen Buddhism koan satori enlightenment",
        "existentialism Sartre Camus absurdism",
        "Philosophy of mind consciousness",
        "Ethics artificial intelligence alignment",
        # History & Civilization
        "ancient Roman engineering aqueducts", "Renaissance art history",
        "Shakespeare literary analysis", "French Revolution causes",
        "Industrial Revolution impact", "Space race Apollo program",
        "Silk Road trade ancient civilizations",
        "Egyptian pyramids construction techniques",
        "Greek philosophy Socrates Plato Aristotle",
        "Mongol Empire Genghis Khan conquest",
        "Ottoman Empire rise fall",
        "American Civil War causes consequences",
        "Cold War nuclear arms race",
        "decolonization Africa Asia independence",
        # Real World & Practical
        "healthy meal preparation nutrition guide",
        "home gardening vegetables beginners",
        "effective communication skills workplace",
        "time management productivity techniques",
        "public speaking presentation skills",
        "relationship psychology attachment theory",
        "child development parenting strategies",
        "sleep science circadian rhythm hygiene",
        "stress management coping mechanisms",
        "first aid emergency medical response",
        # Environment & Nature
        "climate change greenhouse effect mitigation",
        "Renewable energy solar wind comparison",
        "Ecosystem biodiversity conservation",
        "Water cycle hydrology drought",
        "ocean plastic pollution solutions",
        "deforestation Amazon rainforest impact",
        "endangered species wildlife conservation",
        "sustainable agriculture organic farming",
        "electric vehicle battery technology",
        "carbon capture climate engineering",
        # Arts & Culture
        "modern art movements impressionism cubism",
        "film directing cinematography techniques",
        "music production mixing mastering",
        "creative writing fiction storytelling",
        "world cuisine culinary traditions",
        "fashion design history textile",
        "street art graffiti Banksy murals",
        "video game design development process",
        # Sports & Fitness
        "marathon training plan running techniques",
        "strength training muscle hypertrophy",
        "martial arts history philosophy techniques",
        "chess opening strategy grandmaster games",
        "Olympic Games history records athletes",
    ]

    def fetch_wikipedia_search(self):
        query_idx = random.randint(0, len(self.SEARCH_QUERIES) - 1)
        query = self.SEARCH_QUERIES[query_idx]
        all_lines = []
        try:
            resp = self.session.get(
                'https://en.wikipedia.org/w/api.php',
                params={
                    'action': 'query', 'list': 'search',
                    'srsearch': query, 'srlimit': 5, 'format': 'json'
                }, timeout=15
            )
            results = resp.json().get('query', {}).get('search', [])
            for r in results:
                lines = self._fetch_wiki_article(r['title'])
                all_lines.extend(lines)
                if lines:
                    time.sleep(0.3)
        except Exception as e:
            print(f"[Fetcher] Wikipedia search '{query}' error: {e}")
        return all_lines, query

    # ------------------------------------------------------------------
    # SOURCE 5: Wiktionary — definitions & word knowledge
    # ------------------------------------------------------------------
    WIKTIONARY_WORDS = [
        "algorithm", "democracy", "photosynthesis", "entropy", "metaphor",
        "paradigm", "quantum", "evolution", "synthesis", "hypothesis",
        "empirical", "dialectic", "epistemology", "ontology", "taxonomy",
        "heuristic", "catalyst", "symbiosis", "rhetoric", "inference",
        "axiom", "theorem", "lemma", "corollary", "conjecture",
        "derivative", "integral", "function", "variable", "equation",
        "molecule", "electron", "proton", "neutron", "isotope",
        "chromosome", "genome", "protein", "enzyme", "antibody",
        "neuron", "synapse", "cortex", "hippocampus", "cerebellum",
    ]

    def fetch_wiktionary(self):
        word = random.choice(self.WIKTIONARY_WORDS)
        key = f"wikt:{word}"
        if key in self.fetched_urls:
            return []
        try:
            resp = self.session.get(
                'https://en.wiktionary.org/w/api.php',
                params={
                    'action': 'query', 'titles': word,
                    'prop': 'extracts', 'explaintext': True, 'format': 'json'
                }, timeout=15
            )
            pages = resp.json()['query']['pages']
            for pid, page in pages.items():
                if 'extract' in page and len(page['extract']) > 50:
                    lines = self._clean_text(page['extract'])
                    self.fetched_urls.add(key)
                    return lines
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # SOURCE 6: Project Gutenberg — classic literature
    # ------------------------------------------------------------------
    def fetch_gutenberg(self):
        if self.gutenberg_idx >= len(self.GUTENBERG_BOOKS):
            self.gutenberg_idx = 0
            return []
        book_id = self.GUTENBERG_BOOKS[self.gutenberg_idx]
        self.gutenberg_idx += 1
        url = f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt'
        if url in self.fetched_urls:
            return []
        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 404:
                url = f'https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt'
                resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            content = resp.content.decode('utf-8', errors='ignore')
            lines = []
            for line in content.split('\n'):
                line = line.strip()
                if line and len(line) > 30 and not line.startswith('***'):
                    if sum(c.isalpha() for c in line) > len(line) * 0.5:
                        lines.append(line)
            self.fetched_urls.add(url)
            return lines
        except Exception as e:
            print(f"[Fetcher] Gutenberg #{book_id} error: {e}")
            return []

    # ------------------------------------------------------------------
    # SOURCE 7: Wikisource — primary legal/historical documents
    # ------------------------------------------------------------------
    def fetch_wikisource(self):
        if self.wikisource_idx >= len(self.WIKISOURCE_DOCS):
            self.wikisource_idx = 0
            return [], ""
        doc = self.WIKISOURCE_DOCS[self.wikisource_idx]
        self.wikisource_idx += 1
        key = f"wikisource:{doc}"
        if key in self.fetched_urls:
            return [], doc
        try:
            resp = self.session.get(
                'https://en.wikisource.org/w/api.php',
                params={
                    'action': 'query', 'titles': doc,
                    'prop': 'extracts', 'explaintext': True, 'format': 'json'
                }, timeout=15
            )
            pages = resp.json()['query']['pages']
            for pid, page in pages.items():
                if 'extract' in page and len(page['extract']) > 100:
                    lines = self._clean_text(page['extract'])
                    self.fetched_urls.add(key)
                    return lines, doc
        except Exception as e:
            print(f"[Fetcher] Wikisource '{doc}' error: {e}")
        return [], doc

    # ------------------------------------------------------------------
    # SOURCE 8: Wikiquote — famous quotes & wisdom
    # ------------------------------------------------------------------
    QUOTE_PEOPLE = [
        "Albert_Einstein", "Isaac_Newton", "Aristotle", "Plato",
        "Socrates", "Confucius", "Mahatma_Gandhi", "Martin_Luther_King_Jr.",
        "Abraham_Lincoln", "Winston_Churchill", "Nelson_Mandela",
        "Marie_Curie", "Charles_Darwin", "Nikola_Tesla", "Leonardo_da_Vinci",
        "William_Shakespeare", "Mark_Twain", "Oscar_Wilde", "Voltaire",
        "Benjamin_Franklin", "Thomas_Jefferson", "Theodore_Roosevelt",
        "Ada_Lovelace", "Alan_Turing", "Richard_Feynman", "Carl_Sagan",
        "Stephen_Hawking", "Noam_Chomsky", "Bertrand_Russell", "Nietzsche",
    ]

    def fetch_wikiquote(self):
        person = random.choice(self.QUOTE_PEOPLE)
        key = f"wikiquote:{person}"
        if key in self.fetched_urls:
            return []
        try:
            resp = self.session.get(
                'https://en.wikiquote.org/w/api.php',
                params={
                    'action': 'query', 'titles': person,
                    'prop': 'extracts', 'explaintext': True, 'format': 'json'
                }, timeout=15
            )
            pages = resp.json()['query']['pages']
            for pid, page in pages.items():
                if 'extract' in page and len(page['extract']) > 100:
                    lines = self._clean_text(page['extract'])
                    self.fetched_urls.add(key)
                    return lines
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # SOURCE 9: Open textbook / raw text URLs
    # ------------------------------------------------------------------
    def fetch_open_textbook(self):
        if self.textbook_idx >= len(self.OPEN_TEXTBOOK_URLS):
            self.textbook_idx = 0
            return []
        url = self.OPEN_TEXTBOOK_URLS[self.textbook_idx]
        self.textbook_idx += 1
        if url in self.fetched_urls:
            return []
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            content = resp.content.decode('utf-8', errors='ignore')
            lines = []
            for line in content.split('\n'):
                line = line.strip()
                if line and len(line) > 30 and not line.startswith('***'):
                    if sum(c.isalpha() for c in line) > len(line) * 0.5:
                        lines.append(line)
            self.fetched_urls.add(url)
            return lines
        except Exception as e:
            print(f"[Fetcher] Open textbook error: {e}")
            return []

    # ------------------------------------------------------------------
    # SOURCE 10: Wikipedia "Did you know" / Current events
    # ------------------------------------------------------------------
    def fetch_wikipedia_current(self):
        all_lines = []
        try:
            resp = self.session.get(
                'https://en.wikipedia.org/w/api.php',
                params={
                    'action': 'query', 'titles': 'Portal:Current_events',
                    'prop': 'extracts', 'explaintext': True, 'format': 'json'
                }, timeout=15
            )
            pages = resp.json()['query']['pages']
            for pid, page in pages.items():
                if 'extract' in page and len(page['extract']) > 100:
                    all_lines = self._clean_text(page['extract'])
        except Exception:
            pass
        return all_lines

    # ------------------------------------------------------------------
    # SOURCE 11: Wikipedia links crawl — follow links from articles
    # ------------------------------------------------------------------
    def fetch_wikipedia_links(self):
        """Pick a random already-fetched article and follow its links for related content."""
        all_lines = []
        wiki_titles = [u for u in self.fetched_urls if not u.startswith(('http', 'wikt:', 'wikisource:', 'wikiquote:'))]
        if not wiki_titles:
            return []
        seed = random.choice(wiki_titles)
        try:
            resp = self.session.get(
                'https://en.wikipedia.org/w/api.php',
                params={
                    'action': 'query', 'titles': seed,
                    'prop': 'links', 'pllimit': 20,
                    'plnamespace': 0, 'format': 'json'
                }, timeout=15
            )
            pages = resp.json().get('query', {}).get('pages', {})
            for pid, page in pages.items():
                for link in page.get('links', [])[:8]:
                    lines = self._fetch_wiki_article(link['title'])
                    all_lines.extend(lines)
                    if lines:
                        time.sleep(0.3)
        except Exception:
            pass
        return all_lines

    # ------------------------------------------------------------------
    # SOURCE 12: Local files — .txt and .xml.bz2 in working directory
    # ------------------------------------------------------------------
    SKIP_FILES = {"data.txt", "trained_data.json", "fetched_history.json",
                  "tokenizer.json", "growth_log.json", "ai.py"}

    def scan_local_files(self):
        """Scan working directory for .txt and .xml.bz2 files to ingest."""
        all_lines = []
        local_dir = Path(".")
        files = list(local_dir.glob("*.txt")) + list(local_dir.glob("*.xml.bz2"))
        for fpath in files:
            fname = fpath.name
            if fname in self.SKIP_FILES:
                continue
            key = f"local:{fname}"
            if key in self.fetched_urls:
                continue
            try:
                if fname.endswith(".xml.bz2"):
                    lines = self._parse_wiki_dump(fpath)
                else:
                    lines = self._read_local_txt(fpath)
                if lines:
                    all_lines.extend(lines)
                    self.fetched_urls.add(key)
                    print(f"[Local] Ingested {len(lines)} lines from {fname}")
            except Exception as e:
                print(f"[Local] Error reading {fname}: {e}")
                self.fetched_urls.add(key)  # mark to skip on retry
        return all_lines

    def _read_local_txt(self, fpath):
        lines = []
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line and len(line) > 30:
                    if sum(c.isalpha() for c in line) > len(line) * 0.5:
                        lines.append(line)
        return lines

    def _parse_wiki_dump(self, fpath):
        """Stream-parse a Wikipedia XML dump (.xml.bz2) in chunks."""
        import bz2
        lines = []
        max_lines = 50000  # process in chunks per round
        print(f"[Local] Parsing Wikipedia dump: {fpath.name} (streaming)...")
        try:
            with bz2.open(fpath, 'rt', encoding='utf-8', errors='ignore') as f:
                in_text = False
                current_text = []
                for raw_line in f:
                    if '<text' in raw_line:
                        in_text = True
                        # get content after the tag on same line
                        idx = raw_line.find('>')
                        if idx >= 0:
                            current_text.append(raw_line[idx+1:])
                        continue
                    if '</text>' in raw_line:
                        in_text = False
                        idx = raw_line.find('</text>')
                        current_text.append(raw_line[:idx])
                        full_text = ''.join(current_text)
                        current_text = []
                        # clean wiki markup
                        cleaned = self._clean_wiki_markup(full_text)
                        for cl in cleaned:
                            if cl not in lines:
                                lines.append(cl)
                        if len(lines) >= max_lines:
                            break
                        continue
                    if in_text:
                        current_text.append(raw_line)
        except Exception as e:
            print(f"[Local] Wiki dump parse error: {e}")
        return lines

    def _clean_wiki_markup(self, text):
        """Strip wiki markup to plain text lines."""
        # Remove templates, refs, tags, tables
        text = re.sub(r'\{\{[^}]*\}\}', '', text)
        text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^/]*/>', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)  # [[link|text]] → text
        text = re.sub(r'\[https?://[^\s\]]*\s?([^\]]*)\]', r'\1', text)  # external links
        text = re.sub(r"'{2,}", '', text)  # bold/italic
        text = re.sub(r'==+\s*See also\s*==+.*', '', text, flags=re.DOTALL)
        text = re.sub(r'==+\s*References\s*==+.*', '', text, flags=re.DOTALL)
        text = re.sub(r'==+\s*External links\s*==+.*', '', text, flags=re.DOTALL)
        text = re.sub(r'==+[^=]+=+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        lines = []
        for sent in text.split('. '):
            sent = sent.strip()
            if len(sent) > 40 and sum(c.isalpha() for c in sent) > len(sent) * 0.5:
                lines.append(sent + '.')
        return lines

    # ==================================================================
    # MAIN LOOP — cycles through ALL sources intelligently
    # ==================================================================
    def run_forever(self):
        sources = [
            "local_files",
            "wiki_random", "wiki_category", "wiki_vital", "wiki_search",
            "wiki_links", "gutenberg", "wikisource", "wikiquote",
            "wiktionary", "open_textbook", "wiki_current",
        ]
        # Count local files available
        local_files = [f for f in Path(".").glob("*.txt") if f.name not in self.SKIP_FILES]
        local_files += [f for f in Path(".").glob("*.xml.bz2")]
        print(f"[Fetcher] Started — {len(sources)} source types, "
              f"{len(self.WIKI_CATEGORIES)} categories, "
              f"{len(self.VITAL_TOPICS)} vital topics, "
              f"{len(self.GUTENBERG_BOOKS)} books, "
              f"{len(local_files)} local files")
        fetch_round = 0

        while self.running:
            available = self.dataset.available()

            if available < self.config.MIN_DATA_LINES * 2:
                source_type = sources[fetch_round % len(sources)]
                lines = []
                source_label = source_type

                try:
                    if source_type == "local_files":
                        lines = self.scan_local_files()
                        source_label = "Local files"
                    elif source_type == "wiki_random":
                        lines = self.fetch_wikipedia_random(self.config.FETCH_BATCH)
                        source_label = "Wikipedia (random)"
                    elif source_type == "wiki_category":
                        lines, cat = self.fetch_wikipedia_category()
                        source_label = f"Wikipedia (category: {cat})"
                    elif source_type == "wiki_vital":
                        lines, topic = self.fetch_wikipedia_vital()
                        source_label = f"Wikipedia (vital: {topic})"
                    elif source_type == "wiki_search":
                        lines, query = self.fetch_wikipedia_search()
                        source_label = f"Wikipedia (search: {query[:30]})"
                    elif source_type == "wiki_links":
                        lines = self.fetch_wikipedia_links()
                        source_label = "Wikipedia (link crawl)"
                    elif source_type == "gutenberg":
                        lines = self.fetch_gutenberg()
                        source_label = "Project Gutenberg"
                    elif source_type == "wikisource":
                        lines, doc = self.fetch_wikisource()
                        source_label = f"Wikisource ({doc[:25]})"
                    elif source_type == "wikiquote":
                        lines = self.fetch_wikiquote()
                        source_label = "Wikiquote"
                    elif source_type == "wiktionary":
                        lines = self.fetch_wiktionary()
                        source_label = "Wiktionary"
                    elif source_type == "open_textbook":
                        lines = self.fetch_open_textbook()
                        source_label = "Open textbook"
                    elif source_type == "wiki_current":
                        lines = self.fetch_wikipedia_current()
                        source_label = "Wikipedia (current events)"
                except Exception as e:
                    print(f"[Fetcher] {source_label} error: {e}")
                    lines = []

                if lines:
                    self.dataset.add_lines(lines)
                    self.fetched_count += len(lines)
                    print(f"[Fetcher] +{len(lines):>5} lines from {source_label} | "
                          f"Total: {self.fetched_count} | Available: {self.dataset.available()}")
                    self._save_history()

                fetch_round += 1

            time.sleep(self.config.FETCH_DELAY)

    def stop(self):
        self.running = False


# =============================
# CONTINUOUS TRAINER (ENHANCED)
# =============================
class ContinuousTrainer:
    def __init__(self, model, tokenizer, config, growth_manager):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.growth_manager = growth_manager
        self.device = config.DEVICE
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.LEARNING_RATE,
            betas=(config.BETA1, config.BETA2), eps=config.EPS,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = self._create_scheduler()
        self.scaler = torch.amp.GradScaler('cuda') if config.ENABLE_MIXED_PRECISION else None
        
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.recent_losses = []
        self.running = True
        self.pause_event = threading.Event()
        self.paused_event = threading.Event()
        self.last_checkpoint_time = time.time()
        
        # Growth tracking
        self.loss_history = []
        self.stable_low_loss_steps = 0
        self.growth_pending = False
        
        # Memory monitor + dynamic batch sizing
        self.memory_monitor = MemoryMonitor(config.MAX_MEMORY_GB)
        self.initial_batch_size = config.BATCH_SIZE
        self.max_batch_size = config.BATCH_SIZE * 8  # cap for auto-boost
        self.boost_check_interval = 20  # check every N steps
        
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    def _create_scheduler(self):
        def lr_lambda(step):
            if step < self.config.WARMUP_STEPS:
                return max(step / self.config.WARMUP_STEPS, 0.01)
            progress = (step - self.config.WARMUP_STEPS) / max(100000 - self.config.WARMUP_STEPS, 1)
            return max(0.5 * (1 + math.cos(math.pi * min(progress, 1.0))), 0.01)
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch):
        self.model.train()
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        
        if self.scaler and self.config.ENABLE_MIXED_PRECISION:
            with torch.amp.autocast('cuda'):
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    ignore_index=self.tokenizer.special_tokens['<pad>']
                )
            loss = loss / self.config.GRAD_ACCUMULATION_STEPS
            self.scaler.scale(loss).backward()
        else:
            logits = self.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=self.tokenizer.special_tokens['<pad>']
            )
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
            'best_loss': self.best_loss,
            'loss_history': self.loss_history[-1000:],
            'stable_low_loss_steps': self.stable_low_loss_steps,
        }
        # Backup dedup state inside model.pt
        if hasattr(self, 'dataset') and self.dataset:
            checkpoint['trained_hashes'] = list(self.dataset.trained_hashes)
            checkpoint['trained_count'] = len(self.dataset.trained_hashes)
        if hasattr(self, 'fetcher') and self.fetcher:
            checkpoint['fetcher_state'] = {
                'fetched_urls': list(self.fetcher.fetched_urls),
                'category_idx': self.fetcher.category_idx,
                'gutenberg_idx': self.fetcher.gutenberg_idx,
                'vital_idx': self.fetcher.vital_idx,
                'wikisource_idx': self.fetcher.wikisource_idx,
                'textbook_idx': self.fetcher.textbook_idx,
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
                checkpoint = torch.load(self.config.MODEL_FILE, map_location='cpu', weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.step = checkpoint.get('step', 0)
                self.epoch = checkpoint.get('epoch', 0)
                self.best_loss = checkpoint.get('best_loss', float('inf'))
                self.loss_history = checkpoint.get('loss_history', [])
                self.stable_low_loss_steps = checkpoint.get('stable_low_loss_steps', 0)
                print(f"Resumed: {self.config.SIZE} at step {self.step:,}")
                return True
            except Exception as e:
                print(f"Failed to load model.pt: {e}")
        return False

    def check_growth_criteria(self):
        """Check if model should grow"""
        if len(self.loss_history) < self.config.GROWTH_STABLE_STEPS:
            return False
        
        recent_losses = self.loss_history[-self.config.GROWTH_STABLE_STEPS:]
        avg_loss = sum(recent_losses) / len(recent_losses)
        
        if avg_loss < self.config.GROWTH_LOSS_THRESHOLD:
            all_below = all(loss < self.config.GROWTH_LOSS_THRESHOLD for loss in recent_losses)
            if all_below:
                return True
        
        return False

    def perform_growth(self):
        """Grow model to next size"""
        next_size = self.growth_manager.get_next_size(self.config.SIZE)
        if next_size is None:
            print("Already at maximum model size!")
            return False
        
        print(f"\n{'='*60}")
        print(f"  INITIATING MODEL GROWTH")
        print(f"  {self.config.SIZE} → {next_size}")
        print(f"  Current step: {self.step}")
        print(f"  Current loss: {self.loss_history[-1]:.4f}")
        print(f"{'='*60}\n")
        
        # Save current state
        self.save_checkpoint(is_growth=True)
        
        # Log growth event
        metrics = {
            'steps_at_growth': self.step,
            'avg_loss_1000_steps': sum(self.loss_history[-1000:]) / len(self.loss_history[-1000:]),
            'memory_usage_gb': self.memory_monitor.get_memory_usage_gb()
        }
        self.growth_manager.log_growth_event(
            self.config.SIZE, next_size, self.step,
            self.loss_history[-1], metrics
        )
        
        # Create new model
        new_config = ModelConfig(next_size)
        new_config.VOCAB_SIZE = self.config.VOCAB_SIZE
        new_model = AdvancedTransformer(new_config).to(new_config.DEVICE)
        
        # Transfer weights
        new_model = self.growth_manager.transfer_weights(self.model, new_model)
        
        # Update trainer
        self.model = new_model
        self.config = new_config
        
        # Recreate optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=new_config.LEARNING_RATE,
            betas=(new_config.BETA1, new_config.BETA2), eps=new_config.EPS,
            weight_decay=new_config.WEIGHT_DECAY
        )
        self.scheduler = self._create_scheduler()
        
        # Reset growth tracking
        self.stable_low_loss_steps = 0
        self.loss_history = []
        
        # Cleanup
        self.memory_monitor.force_cleanup()
        
        print(f"\n{'='*60}")
        print(f"  GROWTH COMPLETE - Now training: {next_size}")
        print(f"{'='*60}\n")
        
        # Save grown model immediately
        self.save_checkpoint(is_growth=True)
        
        return True

    def pause(self):
        self.pause_event.set()
        self.paused_event.wait(timeout=30)

    def resume(self):
        self.pause_event.clear()

    def train_forever(self, dataset):
        self.dataset = dataset
        print(f"\n{'='*60}")
        print(f"CONTINUOUS TRAINING STARTED")
        print(f"Model: {self.config.SIZE}")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Seq length: {self.config.MAX_SEQ_LEN}")
        print(f"Grad accumulation: {self.config.GRAD_ACCUMULATION_STEPS}")
        print(f"{'='*60}\n")

        waiting_logged = False

        while self.running:
            # Pause for chat
            if self.pause_event.is_set():
                self.paused_event.set()
                while self.pause_event.is_set() and self.running:
                    time.sleep(0.05)
                self.paused_event.clear()
                if not self.running:
                    break

            # Memory check
            if not self.memory_monitor.is_memory_safe():
                print(f"[Memory] High usage: {self.memory_monitor.get_memory_usage_gb():.2f}GB, cleaning...")
                self.memory_monitor.force_cleanup()
                if self.config.BATCH_SIZE > 1:
                    self.config.BATCH_SIZE = max(1, self.config.BATCH_SIZE // 2)
                    print(f"[Memory] Auto-reduced batch size to {self.config.BATCH_SIZE}")
                time.sleep(2)

            # Dynamic batch boost — increase batch size when RAM is plentiful
            if self.step > 0 and self.step % self.boost_check_interval == 0:
                if self.memory_monitor.can_boost() and self.config.BATCH_SIZE < self.max_batch_size:
                    old_bs = self.config.BATCH_SIZE
                    self.config.BATCH_SIZE = min(self.config.BATCH_SIZE + 1, self.max_batch_size)
                    if self.config.BATCH_SIZE != old_bs:
                        free = self.memory_monitor.get_free_ram_gb()
                        print(f"[Boost] Batch size {old_bs} → {self.config.BATCH_SIZE} (free RAM: {free:.1f}GB)")

            batch = dataset.get_batch(self.config.BATCH_SIZE)
            if batch is None:
                if not waiting_logged:
                    print("[Trainer] Waiting for data...")
                    waiting_logged = True
                dataset.flush_consumed()
                dataset.reload_from_file()
                time.sleep(5)
                continue

            waiting_logged = False
            try:
                loss = self.train_step(batch)
                
                # Gradient accumulation
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
                err_msg = str(e).lower()
                is_oom = any(k in err_msg for k in ["out of memory", "insufficient", "not enough", "could not allocate"])
                is_gpu_crash = "will not respond" in err_msg or "device removed" in err_msg
                if is_oom or is_gpu_crash:
                    self.memory_monitor.force_cleanup()
                    if self.config.BATCH_SIZE > 1:
                        self.config.BATCH_SIZE = max(1, self.config.BATCH_SIZE // 2)
                        print(f"[GPU OOM] Reduced batch size to {self.config.BATCH_SIZE}")
                    elif self.config.MAX_SEQ_LEN > 64:
                        self.config.MAX_SEQ_LEN = self.config.MAX_SEQ_LEN // 2
                        self.dataset.max_length = self.config.MAX_SEQ_LEN
                        print(f"[GPU OOM] Reduced seq length to {self.config.MAX_SEQ_LEN}")
                    else:
                        print(f"[GPU OOM] Already at minimum, increasing grad_accum")
                        self.config.GRAD_ACCUMULATION_STEPS += 2
                    if is_gpu_crash:
                        print("[GPU] Device crashed — re-creating on new device...")
                        try:
                            self.optimizer.zero_grad(set_to_none=True)
                            self.memory_monitor.force_cleanup()
                            time.sleep(3)
                            new_device = torch_directml.device() if DIRECTML_AVAILABLE else torch.device("cpu")
                            self.config.DEVICE = new_device
                            self.device = new_device
                            self.model = self.model.to(new_device)
                            self.optimizer = torch.optim.AdamW(
                                self.model.parameters(), lr=self.config.LEARNING_RATE,
                                betas=(self.config.BETA1, self.config.BETA2),
                                eps=self.config.EPS, weight_decay=self.config.WEIGHT_DECAY
                            )
                            self.scheduler = self._create_scheduler()
                            print(f"[GPU] Recovered on {new_device}")
                        except Exception as re_err:
                            print(f"[GPU] Recovery failed ({re_err}), falling back to CPU")
                            self.config.DEVICE = torch.device("cpu")
                            self.device = torch.device("cpu")
                            self.model = self.model.to("cpu")
                            self.optimizer = torch.optim.AdamW(
                                self.model.parameters(), lr=self.config.LEARNING_RATE,
                                betas=(self.config.BETA1, self.config.BETA2),
                                eps=self.config.EPS, weight_decay=self.config.WEIGHT_DECAY
                            )
                            self.scheduler = self._create_scheduler()
                    time.sleep(3)
                    continue
                print(f"[Trainer] Error: {e}")
                time.sleep(1)
                continue
            except Exception as e:
                print(f"[Trainer] Error: {e}")
                time.sleep(1)
                continue

            if self.step % 10 == 0:
                lr = self.scheduler.get_last_lr()[0]
                avail = dataset.available()
                avg = sum(self.recent_losses[-50:]) / max(len(self.recent_losses[-50:]), 1)
                mem = self.memory_monitor.get_memory_usage_gb()
                free = self.memory_monitor.get_free_ram_gb()
                print(f"[{self.config.SIZE}] Step {self.step:7d} | Loss: {loss:.4f} | Avg: {avg:.4f} | LR: {lr:.6f} | Data: {avail} | Mem: {mem:.1f}GB | Free: {free:.1f}GB | BS: {self.config.BATCH_SIZE}")

            # Check growth
            if self.check_growth_criteria() and not self.growth_pending:
                next_size = self.growth_manager.get_next_size(self.config.SIZE)
                if next_size:
                    print(f"\n[Growth] Criteria met! Loss < {self.config.GROWTH_LOSS_THRESHOLD} for {self.config.GROWTH_STABLE_STEPS} steps")
                    self.growth_pending = True
                    self.perform_growth()
                    self.growth_pending = False

            if self.step % self.config.FLUSH_INTERVAL == 0:
                flushed = dataset.flush_consumed()
                if flushed > 0:
                    print(f"[Data] Flushed {flushed} consumed lines")

            if time.time() - self.last_checkpoint_time >= self.config.CHECKPOINT_INTERVAL:
                self.save_checkpoint()
                flushed = dataset.flush_trained()
                if flushed > 0:
                    print(f"[Data] Trained {flushed} lines committed & removed from data.txt")
                self.last_checkpoint_time = time.time()

        self.save_checkpoint()
        dataset.flush_trained()
        print("[Trainer] Stopped.")

    def get_status(self):
        """API status endpoint"""
        recent = self.loss_history[-100:] if len(self.loss_history) >= 100 else self.loss_history
        return {
            'model_size': self.config.SIZE,
            'step': self.step,
            'epoch': self.epoch,
            'current_loss': self.loss_history[-1] if self.loss_history else 0,
            'avg_loss_100': sum(recent) / len(recent) if recent else 0,
            'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else 0,
            'memory_usage_gb': self.memory_monitor.get_memory_usage_gb(),
            'stable_low_loss_steps': self.stable_low_loss_steps,
            'growth_progress': f"{self.stable_low_loss_steps}/{self.config.GROWTH_STABLE_STEPS}",
            'next_growth_size': self.growth_manager.get_next_size(self.config.SIZE),
            'available_lines': 0,
            'fetched_total': 0,
            'avg_loss': sum(recent) / len(recent) if recent else 0
        }

# =============================
# CHAT SERVER
# =============================
CHAT_HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neural AI — Live Training Chat</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--bg:#0a0a0f;--surface:#12121a;--surface2:#1a1a28;--surface3:#222236;--border:#2a2a44;--border-glow:#6c5ce755;--accent:#6c5ce7;--accent2:#a29bfe;--accent-dim:#6c5ce733;--red:#ff6b6b;--green:#51cf66;--yellow:#ffd43b;--text:#e8e8f0;--text2:#9898b0;--text3:#686880;--user-bg:linear-gradient(135deg,#6c5ce7 0%,#a29bfe 100%);--ai-bg:#16162a}
*{margin:0;padding:0;box-sizing:border-box}
html,body{height:100%;overflow:hidden}
body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);display:flex;flex-direction:column}

/* ---- HEADER ---- */
.header{background:var(--surface);padding:16px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:16px;position:relative;overflow:hidden}
.header::after{content:'';position:absolute;bottom:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--accent),transparent)}
.logo{width:40px;height:40px;border-radius:12px;background:var(--user-bg);display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0}
.header-info{flex:1;min-width:0}
.header-info h1{font-size:1.1em;font-weight:600;color:var(--text);letter-spacing:-0.02em}
.header-info h1 span{color:var(--accent2);font-weight:700}
.header-info .subtitle{font-size:0.72em;color:var(--text3);margin-top:2px}

/* ---- STATS BAR ---- */
.stats{display:flex;gap:6px;flex-wrap:wrap}
.stat{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:6px 12px;font-family:'JetBrains Mono',monospace;font-size:0.7em;color:var(--text2);white-space:nowrap;display:flex;align-items:center;gap:5px}
.stat .dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.dot.green{background:var(--green);box-shadow:0 0 6px var(--green)}
.dot.yellow{background:var(--yellow);box-shadow:0 0 6px var(--yellow)}
.dot.red{background:var(--red);box-shadow:0 0 6px var(--red)}
.stat b{color:var(--text);font-weight:500}

/* ---- CHAT AREA ---- */
#chat{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:16px;scroll-behavior:smooth}
#chat::-webkit-scrollbar{width:6px}
#chat::-webkit-scrollbar-track{background:transparent}
#chat::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
#chat::-webkit-scrollbar-thumb:hover{background:var(--text3)}

.welcome{text-align:center;padding:60px 20px;color:var(--text3)}
.welcome .icon{font-size:48px;margin-bottom:16px}
.welcome h2{font-size:1.3em;color:var(--text2);font-weight:500;margin-bottom:8px}
.welcome p{font-size:0.85em;line-height:1.6;max-width:460px;margin:0 auto}

.msg-row{display:flex;gap:12px;max-width:85%;animation:fadeIn 0.3s ease}
.msg-row.user{margin-left:auto;flex-direction:row-reverse}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

.avatar{width:32px;height:32px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0;margin-top:2px}
.msg-row.user .avatar{background:var(--user-bg);color:white}
.msg-row.ai .avatar{background:var(--surface3);border:1px solid var(--border);color:var(--accent2)}

.bubble{padding:12px 16px;border-radius:16px;font-size:0.9em;line-height:1.65;word-wrap:break-word;position:relative}
.msg-row.user .bubble{background:var(--accent);color:white;border-bottom-right-radius:4px}
.msg-row.ai .bubble{background:var(--surface2);border:1px solid var(--border);color:var(--text);border-bottom-left-radius:4px}
.msg-row.ai .bubble .time{font-size:0.7em;color:var(--text3);margin-top:6px}

.typing{display:flex;gap:4px;padding:4px 0}
.typing span{width:6px;height:6px;border-radius:50%;background:var(--text3);animation:bounce 1.4s infinite}
.typing span:nth-child(2){animation-delay:0.2s}
.typing span:nth-child(3){animation-delay:0.4s}
@keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-8px)}}

/* ---- INPUT AREA ---- */
.input-wrap{padding:16px 24px 20px;background:var(--surface);border-top:1px solid var(--border);position:relative}
.input-wrap::before{content:'';position:absolute;top:-1px;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--accent-dim),transparent)}
.input-box{display:flex;align-items:flex-end;gap:10px;background:var(--surface2);border:1px solid var(--border);border-radius:14px;padding:4px 4px 4px 16px;transition:border-color 0.2s}
.input-box:focus-within{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-dim)}
#input{flex:1;background:none;border:none;color:var(--text);font-size:0.92em;font-family:'Inter',sans-serif;padding:10px 0;outline:none;resize:none;max-height:120px;line-height:1.5}
#input::placeholder{color:var(--text3)}
#btn{width:40px;height:40px;border-radius:10px;border:none;background:var(--accent);color:white;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all 0.2s;flex-shrink:0}
#btn:hover{background:var(--accent2);transform:scale(1.05)}
#btn:active{transform:scale(0.95)}
#btn:disabled{background:var(--surface3);color:var(--text3);cursor:not-allowed;transform:none}
#btn svg{width:18px;height:18px}
.input-hint{font-size:0.68em;color:var(--text3);margin-top:8px;text-align:center}
</style></head><body>

<!-- HEADER -->
<div class="header">
  <div class="logo">🧠</div>
  <div class="header-info">
    <h1>Neural <span>AI</span></h1>
    <div class="subtitle">Autonomous self-training transformer — 500M parameters</div>
  </div>
  <div class="stats" id="stats">
    <div class="stat"><div class="dot green" id="statusDot"></div> <span id="trainStatus">Initializing...</span></div>
    <div class="stat">Step <b id="sStep">0</b></div>
    <div class="stat">Loss <b id="sLoss">—</b></div>
    <div class="stat">Data <b id="sData">0</b></div>
    <div class="stat">Fetched <b id="sFetch">0</b></div>
  </div>
</div>

<!-- CHAT -->
<div id="chat">
  <div class="welcome" id="welcome">
    <div class="icon">🧠</div>
    <h2>Neural AI is training</h2>
    <p>The model is continuously learning from Wikipedia, books, and knowledge sources.
       Send a message — training will pause to respond, then resume automatically.</p>
  </div>
</div>

<!-- INPUT -->
<div class="input-wrap">
  <div class="input-box">
    <textarea id="input" rows="1" placeholder="Ask anything... training pauses to answer"
      onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
    <button id="btn" onclick="send()">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
    </button>
  </div>
  <div class="input-hint">Press Enter to send · Shift+Enter for new line · Training resumes after each response</div>
</div>

<script>
const chat=document.getElementById('chat');
const input=document.getElementById('input');
const btn=document.getElementById('btn');

// Auto-resize textarea
input.addEventListener('input',()=>{
  input.style.height='auto';
  input.style.height=Math.min(input.scrollHeight,120)+'px';
});

function esc(t){const d=document.createElement('div');d.textContent=t;return d.innerHTML;}
function timeStr(){return new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}

function addMsg(role,text){
  const w=document.getElementById('welcome');if(w)w.remove();
  const row=document.createElement('div');
  row.className='msg-row '+role;
  const av=role==='user'?'👤':'🧠';
  const extra=role==='ai'?'<div class="time">'+timeStr()+'</div>':'';
  row.innerHTML='<div class="avatar">'+av+'</div><div class="bubble">'+esc(text)+extra+'</div>';
  chat.appendChild(row);
  chat.scrollTop=chat.scrollHeight;
}

function showTyping(){
  const w=document.getElementById('welcome');if(w)w.remove();
  const row=document.createElement('div');row.className='msg-row ai';row.id='typing';
  row.innerHTML='<div class="avatar">🧠</div><div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>';
  chat.appendChild(row);chat.scrollTop=chat.scrollHeight;
}
function hideTyping(){const t=document.getElementById('typing');if(t)t.remove();}

async function send(){
  const msg=input.value.trim();if(!msg)return;
  input.value='';input.style.height='auto';
  addMsg('user',msg);
  btn.disabled=true;
  document.getElementById('statusDot').className='dot yellow';
  document.getElementById('trainStatus').textContent='Paused — generating...';
  showTyping();
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});
    const d=await r.json();
    hideTyping();
    addMsg('ai',d.response);
  }catch(e){hideTyping();addMsg('ai','Connection error — model may still be initializing.');}
  btn.disabled=false;
  document.getElementById('statusDot').className='dot green';
  document.getElementById('trainStatus').textContent='Training';
}

// Live stats
setInterval(async()=>{
  try{
    const r=await fetch('/api/status');const d=await r.json();
    document.getElementById('sStep').textContent=Number(d.step).toLocaleString();
    document.getElementById('sLoss').textContent=d.avg_loss;
    document.getElementById('sData').textContent=Number(d.available_lines).toLocaleString();
    document.getElementById('sFetch').textContent=Number(d.fetched_total).toLocaleString();
    if(!btn.disabled){
      document.getElementById('statusDot').className='dot green';
      document.getElementById('trainStatus').textContent='Training';
    }
  }catch(e){
    document.getElementById('statusDot').className='dot red';
    document.getElementById('trainStatus').textContent='Connecting...';
  }
},3000);
</script></body></html>"""

class ChatServer:
    def __init__(self, app, port=8080):
        self.app = app
        self.port = port
        self.history = []

    def start(self):
        app_ref = self.app
        history_ref = self.history

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/' or self.path == '/chat':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(CHAT_HTML.encode('utf-8'))
                elif self.path == '/api/status':
                    trainer = app_ref.trainer
                    fetcher = app_ref.fetcher
                    dataset = app_ref.dataset
                    avg_loss = 0
                    if trainer and trainer.recent_losses:
                        avg_loss = sum(trainer.recent_losses[-50:]) / len(trainer.recent_losses[-50:])
                    status = {
                        'step': trainer.step if trainer else 0,
                        'avg_loss': f'{avg_loss:.4f}',
                        'available_lines': dataset.available() if dataset else 0,
                        'fetched_total': fetcher.fetched_count if fetcher else 0,
                    }
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(status).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == '/api/chat':
                    length = int(self.headers.get('Content-Length', 0))
                    body = self.rfile.read(length)
                    data = json.loads(body)
                    user_msg = data.get('message', '')

                    if app_ref.trainer:
                        app_ref.trainer.pause()

                    try:
                        prompt = "<cls>"
                        for turn in history_ref[-3:]:
                            prompt += f" User: {turn['user']} AI: {turn['ai']}"
                        prompt += f" User: {user_msg} AI:"
                        response = app_ref.model.generate(
                            app_ref.tokenizer, prompt,
                            max_length=150, temperature=0.8, top_k=40
                        )
                        history_ref.append({
                            'user': user_msg, 'ai': response,
                            'timestamp': datetime.now().isoformat()
                        })
                    except Exception as e:
                        response = f"(Model still early in training: {str(e)[:100]})"

                    if app_ref.trainer:
                        app_ref.trainer.resume()

                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'response': response}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass

        server = HTTPServer(('0.0.0.0', self.port), Handler)
        print(f"[Chat] Server listening on http://localhost:{self.port}")
        server.serve_forever()

# =============================
# MAIN APPLICATION (ENHANCED)
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
        if DIRECTML_AVAILABLE:
            torch.manual_seed(self.config.SEED)

        print(f"\n{'='*60}")
        print(f"  AUTONOMOUS PROGRESSIVE GROWTH AI")
        print(f"{'='*60}")
        print(f"  Starting size : {model_size}")
        print(f"  Growth path   : {' → '.join(ModelConfig.GROWTH_PATH)}")
        gpu_name = torch_directml.device_name(0) if DIRECTML_AVAILABLE else "N/A"
        print(f"  Device        : {self.config.DEVICE} ({self.config.DEVICE_TYPE})")
        print(f"  GPU           : {gpu_name}")
        print(f"  Hidden dim    : {self.config.HIDDEN_DIM}")
        print(f"  Layers        : {self.config.NUM_LAYERS}")
        print(f"  Heads         : {self.config.NUM_HEADS}")
        print(f"  Batch size    : {self.config.BATCH_SIZE}")
        print(f"  Grad accum    : {self.config.GRAD_ACCUMULATION_STEPS}")
        print(f"  Max memory    : {self.config.MAX_MEMORY_GB}GB")
        print(f"  Growth loss   : < {self.config.GROWTH_LOSS_THRESHOLD} for {self.config.GROWTH_STABLE_STEPS} steps")
        print(f"  Chat URL      : http://localhost:{self.config.CHAT_PORT}")
        print(f"  API status    : http://localhost:{self.config.CHAT_PORT}/api/status")
        print(f"  Checkpoint    : every {self.config.CHECKPOINT_INTERVAL}s")
        print(f"  Data sources  : Wikipedia (250+ categories), Gutenberg, Wikisource, Wikiquote")
        print(f"  Dedup files  : trained_data.json, fetched_history.json")
        print(f"  Dedup backup : saved inside model.pt")
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

        bootstrap_sources = [
            ("Wikipedia (random)", lambda: temp_fetcher.fetch_wikipedia_random(20)),
            ("Project Gutenberg", lambda: temp_fetcher.fetch_gutenberg()),
            ("Wikipedia (category)", lambda: temp_fetcher.fetch_wikipedia_category()[0]),
            ("Wikipedia (vital)", lambda: temp_fetcher.fetch_wikipedia_vital()[0]),
            ("Wikipedia (search)", lambda: temp_fetcher.fetch_wikipedia_search()[0]),
            ("Wikisource", lambda: temp_fetcher.fetch_wikisource()[0]),
            ("Wikiquote", lambda: temp_fetcher.fetch_wikiquote()),
            ("Open textbook", lambda: temp_fetcher.fetch_open_textbook()),
        ]
        src_idx = 0
        while temp_dataset.total_lines() < self.config.MIN_DATA_LINES:
            current = temp_dataset.total_lines()
            src_name, src_fn = bootstrap_sources[src_idx % len(bootstrap_sources)]
            print(f"  Data: {current} / {self.config.MIN_DATA_LINES} | Fetching from {src_name}...")
            try:
                lines = src_fn()
                if lines:
                    temp_dataset.add_lines(lines)
            except Exception as e:
                print(f"    {src_name} error: {e}")
            src_idx += 1
            time.sleep(1)

        print(f"Got {temp_dataset.total_lines()} lines. Training tokenizer...")
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
                    bar = chr(9608) * filled + chr(9617) * (bar_len - filled)
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
        # 1. Tokenizer
        self.initialize_tokenizer()

        # Prefetch phase: aggressively fetch ~2M lines
        self.prefetch_data()

        # 2. Model
        self.initialize_model()

        # 3. Dataset
        self.dataset = ConsumingDataset(
            self.config.DATA_FILE, self.tokenizer, self.config.MAX_SEQ_LEN
        )

        # 4. Data fetcher
        self.fetcher = DataFetcher(self.dataset, self.config)
        if hasattr(self, 'fetched_urls_bootstrap'):
            self.fetcher.fetched_urls.update(self.fetched_urls_bootstrap)

        # Restore dedup state from model.pt if JSON files are incomplete/missing
        if os.path.exists(self.config.MODEL_FILE):
            try:
                ckpt = torch.load(self.config.MODEL_FILE, map_location='cpu', weights_only=False)
                # Restore trained hashes
                backup_hashes = set(ckpt.get('trained_hashes', []))
                if backup_hashes and len(backup_hashes) > len(self.dataset.trained_hashes):
                    added = len(backup_hashes) - len(self.dataset.trained_hashes)
                    self.dataset.trained_hashes.update(backup_hashes)
                    self.dataset._save_trained_hashes()
                    self.dataset._load_lines()
                    print(f"[Dedup] Restored {added:,} trained hashes from model.pt backup")
                # Restore fetcher state
                fs = ckpt.get('fetcher_state', {})
                if fs:
                    backup_urls = set(fs.get('fetched_urls', []))
                    if len(backup_urls) > len(self.fetcher.fetched_urls):
                        added = len(backup_urls) - len(self.fetcher.fetched_urls)
                        self.fetcher.fetched_urls.update(backup_urls)
                        self.fetcher.category_idx = max(self.fetcher.category_idx, fs.get('category_idx', 0))
                        self.fetcher.gutenberg_idx = max(self.fetcher.gutenberg_idx, fs.get('gutenberg_idx', 0))
                        self.fetcher.vital_idx = max(self.fetcher.vital_idx, fs.get('vital_idx', 0))
                        self.fetcher.wikisource_idx = max(self.fetcher.wikisource_idx, fs.get('wikisource_idx', 0))
                        self.fetcher.textbook_idx = max(self.fetcher.textbook_idx, fs.get('textbook_idx', 0))
                        self.fetcher.fetched_count = max(self.fetcher.fetched_count, fs.get('fetched_count', 0))
                        self.fetcher._save_history()
                        print(f"[Dedup] Restored {added:,} fetched URLs from model.pt backup")
                del ckpt
                gc.collect()
            except Exception as e:
                print(f"[Dedup] Could not restore from model.pt: {e}")

        fetcher_thread = threading.Thread(target=self.fetcher.run_forever, daemon=True)
        fetcher_thread.start()

        # 5. Trainer
        self.trainer = ContinuousTrainer(self.model, self.tokenizer, self.config, self.growth_manager)
        self.trainer.fetcher = self.fetcher
        self.trainer.load_checkpoint()
        trainer_thread = threading.Thread(
            target=self.trainer.train_forever, args=(self.dataset,), daemon=True
        )
        trainer_thread.start()

        # 6. Chat server
        chat_server = ChatServer(self, port=self.config.CHAT_PORT)
        try:
            chat_server.start()
        except KeyboardInterrupt:
            print("\n\nShutting down gracefully...")
            self.trainer.running = False
            self.fetcher.running = False
            time.sleep(2)
            self.trainer.save_checkpoint()
            self.dataset.flush_consumed()
            print("Checkpoint saved. You can resume anytime!")
            print("Goodbye!")

# =============================
# ENTRY POINT
# =============================
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

if __name__ == "__main__":
    # Auto-detect size from model.pt, or start fresh at 10M
    saved_size = detect_saved_size()
    if saved_size:
        model_size = saved_size
    else:
        model_size = "10M"
        if len(sys.argv) > 1 and sys.argv[1] in ModelConfig.GROWTH_PATH:
            model_size = sys.argv[1]
    
    print("\n" + "="*60)
    print("  AUTONOMOUS PROGRESSIVE GROWTH AI TRAINER")
    print("="*60)
    print(f"\n  Model: {model_size} ({'Resuming' if saved_size else 'Fresh start'})")
    print(f"  Growth: {' -> '.join(ModelConfig.GROWTH_PATH)}")
    print(f"  Saves to: model.pt (auto-overwrite)")
    print(f"\n  Run 'python ai.py' to resume anytime!")
    print("="*60 + "\n")
    
    time.sleep(1)
    
    app = AIApplication(model_size)
    app.run()

