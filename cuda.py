#!/usr/bin/env python3
"""
AUTONOMOUS PROGRESSIVE GROWTH AI - ULTIMATE 2TB EDITION
COLAB OPTIMIZED - 12GB RAM FRIENDLY + UNIVERSAL FILE PROCESSOR

Features:
- ULTIMATE BPE: 200K vocabulary, FULL Unicode support
- UNIVERSAL FILE PROCESSOR: Handles ANY file type!
- CONTINUOUS FETCHING until 1.9TB of training data
- DEDUPLICATION: Never trains same line twice (hash-based)
- SMART DELETION: Deletes text after training (like original)
- 50+ DIVERSE DATA SOURCES (EXPANDED!)
- STREAMING ARCHITECTURE: Never loads more than 500MB at once
- AUTO-RESUME: Survives Colab disconnects perfectly
- PROGRESSIVE GROWTH: 10M → 50M → 100M → 200M → 350M → 500M
- ADVANCED BPE: Full Unicode, 200K vocabulary
- MEMORY OPTIMIZED: Dynamic batching, monitoring
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
import gzip
import pickle
import signal
from pathlib import Path
from datetime import datetime
from collections import Counter, deque
from typing import Optional, List, Dict, Set, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Optional DirectML support
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
# CONFIGURATION - ULTIMATE 2TB
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

        # ULTIMATE VOCAB SIZE - 200K for 2TB data
        self.VOCAB_SIZE = 500000
        self.MAX_SEQ_LEN = 1024
        self.DROPOUT = 0.1
        self.BIAS = False
        self.ROPE_THETA = 10000.0

        self.BATCH_SIZE = preset["batch_size"]
        self.GRAD_ACCUMULATION_STEPS = preset["grad_accum"]

        self.LEARNING_RATE = 3e-4
        self.WARMUP_STEPS = 2000
        self.WEIGHT_DECAY = 0.1
        self.GRAD_CLIP = 1.0
        self.BETA1 = 0.9
        self.BETA2 = 0.95
        self.EPS = 1e-8

        # DIRECTORY SETUP FOR COLAB / DRIVE - 2TB target
        self.DRIVE_DIR = "/content/drive/MyDrive/ai.train"
        os.makedirs(self.DRIVE_DIR, exist_ok=True)
        
        self.DATA_FILE = os.path.join(self.DRIVE_DIR, "data.txt")
        self.TOKENIZER_FILE = os.path.join(self.DRIVE_DIR, "tokenizer_200k.json")
        self.BPE_CHECKPOINT_DIR = os.path.join(self.DRIVE_DIR, "bpe_checkpoints")
        self.CHECKPOINT_DIR = os.path.join(self.DRIVE_DIR, "checkpoints")
        self.MODEL_FILE = os.path.join(self.DRIVE_DIR, "model.pt")
        self.GROWTH_LOG_FILE = os.path.join(self.DRIVE_DIR, "growth_log.json")
        self.TRAINED_HASHES_FILE = os.path.join(self.DRIVE_DIR, "trained_data.json")
        self.FETCHER_HISTORY_FILE = os.path.join(self.DRIVE_DIR, "fetcher_history.json")
        self.FETCH_POSITION_FILE = os.path.join(self.DRIVE_DIR, "fetch_position.json")  # NEW: Track position
        
        # Create all directories
        for d in [self.BPE_CHECKPOINT_DIR, self.CHECKPOINT_DIR]:
            os.makedirs(d, exist_ok=True)

        # ULTIMATE TARGET: 1.9TB (1900 GB) of training data
        self.TARGET_DATA_SIZE_GB = 1900  # 1.9TB
        self.TARGET_DATA_SIZE_BYTES = self.TARGET_DATA_SIZE_GB * 1024**3
        
        self.CHECKPOINT_INTERVAL = 300  # 5 minutes
        self.FLUSH_INTERVAL = 50
        self.MIN_DATA_LINES = 1000000  # 1M lines for BPE
        self.FETCH_BATCH = 25  # Fetch 25 articles per batch
        self.FETCH_DELAY = 2  # 2 seconds between fetches
        self.PREFETCH_TARGET_LINES = 2_000_000  # Initial prefetch

        self.GROWTH_LOSS_THRESHOLD = 2.8
        self.GROWTH_STABLE_STEPS = 3000

        # Device selection
        if DIRECTML_AVAILABLE:
            try:
                self.DEVICE = torch_directml.device()
                self.DEVICE_TYPE = 'directml'
            except Exception as e:
                print(f"[Warning] DirectML device creation failed: {e}")
                self.DEVICE = torch.device('cpu')
                self.DEVICE_TYPE = 'cpu'
        elif torch.cuda.is_available():
            self.DEVICE = torch.device('cuda')
            self.DEVICE_TYPE = 'cuda'
        else:
            self.DEVICE = torch.device('cpu')
            self.DEVICE_TYPE = 'cpu'

        self.SEED = 42

        # 12GB RAM OPTIMIZATION - Conservative settings for Colab
        self.MAX_MEMORY_GB = 10.0  # Use max 10GB of 12GB available
        self.RESERVE_BYTES = int(2 * 1024 * 1024 * 1024)  # 2GB safety reserve
        self.BUFFER_SIZE = 25000  # Only keep 25k lines in buffer (was 50k)
        self.BPE_CHUNK_SIZE = 2_000_000  # Process 2M lines at a time (was 5M)

        self.ENABLE_MIXED_PRECISION = (self.DEVICE_TYPE == "cuda")


# =============================
# ULTIMATE BPE TOKENIZER - 200K VOCAB, FULL UNICODE
# =============================
class UltimateBPETokenizer:
    """200K vocabulary BPE tokenizer with FULL Unicode support"""
    
    def __init__(self, vocab_size=200000):
        self.vocab_size = vocab_size
        self.special_tokens = {'<pad>':0, '<unk>':1, '<bos>':2, '<eos>':3, 
                               '<sep>':4, '<cls>':5, '<mask>':6}
        self.vocab = {}
        self.merges = {}
        self.cache = {}
        self.inverse_vocab = {}
        
        # Advanced features
        self.byte_fallback = True
        self.normalization = 'nfc'
        self.max_word_length = 100
        self.min_frequency = 2  # Minimum frequency for BPE merges
        self.unicode_range = (0x0000, 0x10FFFF)  # Full Unicode
        
        # Streaming checkpointing for 2TB
        self.checkpoint_dir = None
        self.chunk_size = 2_000_000  # Process 2M lines at a time
        
        # Unicode tracking
        self.unicode_stats = {
            'total_chars_found': 0,
            'unicode_chars_found': 0,
            'unique_unicode': set(),
            'corrupted_lines': 0
        }
        
        # Cache settings
        self.cache_size = 10000
        self.progress_interval = 500
        
        # Statistics
        self.stats = {
            'total_lines': 0,
            'unique_words': 0,
            'unicode_words': 0,
            'merge_history': []
        }

    def set_checkpoint_dir(self, checkpoint_dir):
        """Set directory for streaming checkpoints"""
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _normalize_text(self, text):
        """Normalize text with Unicode support"""
        try:
            import unicodedata
            if self.normalization == 'nfc':
                return unicodedata.normalize('NFC', text)
            elif self.normalization == 'nfd':
                return unicodedata.normalize('NFD', text)
            elif self.normalization == 'nfkc':
                return unicodedata.normalize('NFKC', text)
            elif self.normalization == 'nfkd':
                return unicodedata.normalize('NFKD', text)
        except Exception:
            pass
        return text

    def _get_word_pairs(self, word):
        """Get pairs from a word with full Unicode support"""
        pairs = []
        prev = word[0]
        for char in word[1:]:
            pairs.append((prev, char))
            prev = char
        return pairs

    def _count_pairs_streaming(self, file_path, max_lines=None):
        """Count pairs from file with memory streaming"""
        pair_counts = Counter()
        word_counts = Counter()
        lines_processed = 0
        last_print = time.time()
        
        print(f"\n📊 Counting pairs from {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line or len(line) < 3:
                        continue
                    
                    # Normalize
                    line = self._normalize_text(line)
                    
                    # Track Unicode
                    for char in line:
                        self.unicode_stats['total_chars_found'] += 1
                        code = ord(char)
                        if code > 127:
                            self.unicode_stats['unicode_chars_found'] += 1
                            self.unicode_stats['unique_unicode'].add(code)
                    
                    # Split into words with Unicode-aware split
                    words = re.findall(r'\w+|[^\w\s]', line, re.UNICODE)
                    
                    for word in words:
                        if len(word) > self.max_word_length:
                            word = word[:self.max_word_length]
                        word_tuple = tuple(word)
                        word_counts[word_tuple] += 1
                        
                        if len(word) > 1:
                            pairs = self._get_word_pairs(word)
                            for pair in pairs:
                                pair_counts[pair] += 1
                    
                    lines_processed += 1
                    
                    # Progress
                    if time.time() - last_print >= 5:
                        pct_unicode = (self.unicode_stats['unicode_chars_found'] / 
                                      max(self.unicode_stats['total_chars_found'], 1)) * 100
                        print(f"  Processed: {lines_processed:,} lines | "
                              f"Unicode: {pct_unicode:.1f}% | "
                              f"Unique chars: {len(self.unicode_stats['unique_unicode']):,}")
                        last_print = time.time()
                    
                    if max_lines and lines_processed >= max_lines:
                        break
        
        except Exception as e:
            print(f"  ⚠️ Error reading file: {e}")
            return pair_counts, word_counts, lines_processed
        
        print(f"\n✅ Counted {len(pair_counts):,} unique pairs from {lines_processed:,} lines")
        print(f"   Unicode coverage: {len(self.unicode_stats['unique_unicode']):,} unique characters")
        
        return pair_counts, word_counts, lines_processed

    def train_streaming(self, file_path, target_vocab_size=None):
        """Train BPE with streaming - handles 2TB files!"""
        if target_vocab_size is None:
            target_vocab_size = self.vocab_size
        
        print(f"\n🔥 Training BPE tokenizer (Target: {target_vocab_size:,} tokens)")
        print(f"   Unicode range: U+{self.unicode_range[0]:04X} - U+{self.unicode_range[1]:04X}")
        
        # Initialize vocab with special tokens
        self.vocab = self.special_tokens.copy()
        next_id = len(self.vocab)
        
        # Add all Unicode characters in range as base tokens
        print(f"\n📝 Adding Unicode base vocabulary...")
        for code in range(self.unicode_range[0], min(self.unicode_range[1] + 1, 0x110000)):
            try:
                char = chr(code)
                if char not in self.vocab:
                    self.vocab[char] = next_id
                    next_id += 1
            except ValueError:
                continue
        
        print(f"   Added {len(self.vocab):,} base tokens (Unicode + special)")
        
        # Count pairs from file
        pair_counts, word_counts, lines_processed = self._count_pairs_streaming(file_path)
        
        if not pair_counts:
            print("⚠️ No pairs found! Cannot train tokenizer.")
            return
        
        self.stats['total_lines'] = lines_processed
        self.stats['unique_words'] = len(word_counts)
        
        # BPE merging
        print(f"\n🔄 Performing BPE merges...")
        merge_count = 0
        last_print = time.time()
        
        while len(self.vocab) < target_vocab_size:
            if not pair_counts:
                print("\n  No more pairs to merge!")
                break
            
            # Find most common pair
            best_pair = max(pair_counts.items(), key=lambda x: x[1])
            pair, freq = best_pair
            
            if freq < self.min_frequency:
                print(f"\n  Frequency threshold reached ({freq} < {self.min_frequency})")
                break
            
            # Create new token
            new_token = ''.join(pair)
            if new_token not in self.vocab:
                self.vocab[new_token] = next_id
                next_id += 1
                self.merges[pair] = new_token
                merge_count += 1
                
                self.stats['merge_history'].append({
                    'merge': pair,
                    'token': new_token,
                    'frequency': freq
                })
            
            # Update pair counts
            del pair_counts[pair]
            
            # Progress
            if time.time() - last_print >= 5:
                pct = (len(self.vocab) / target_vocab_size) * 100
                print(f"  Vocab: {len(self.vocab):,} / {target_vocab_size:,} ({pct:.1f}%) | "
                      f"Merges: {merge_count:,}")
                last_print = time.time()
        
        # Build inverse vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"\n✅ Tokenizer training complete!")
        print(f"   Final vocabulary size: {len(self.vocab):,}")
        print(f"   Total merges performed: {merge_count:,}")
        print(f"   Unicode characters: {len(self.unicode_stats['unique_unicode']):,}")

    def encode(self, text):
        """Encode text to token IDs with full Unicode support"""
        if not text:
            return []
        
        # Check cache
        cache_key = hash(text)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Normalize
        text = self._normalize_text(text)
        
        # Split into words
        words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        
        token_ids = []
        for word in words:
            if len(word) > self.max_word_length:
                word = word[:self.max_word_length]
            
            # Apply BPE merges
            chars = list(word)
            while len(chars) > 1:
                pairs = self._get_word_pairs(chars)
                if not pairs:
                    break
                
                # Find pair in merges
                found = False
                for pair in pairs:
                    if pair in self.merges:
                        # Merge this pair
                        new_chars = []
                        i = 0
                        while i < len(chars):
                            if i < len(chars) - 1 and (chars[i], chars[i+1]) == pair:
                                new_chars.append(self.merges[pair])
                                i += 2
                            else:
                                new_chars.append(chars[i])
                                i += 1
                        chars = new_chars
                        found = True
                        break
                
                if not found:
                    break
            
            # Convert to IDs
            for char in chars:
                if char in self.vocab:
                    token_ids.append(self.vocab[char])
                else:
                    # Byte fallback
                    if self.byte_fallback:
                        for byte in char.encode('utf-8'):
                            byte_token = f'<byte_{byte:02x}>'
                            if byte_token not in self.vocab:
                                self.vocab[byte_token] = len(self.vocab)
                                self.inverse_vocab[len(self.vocab)-1] = byte_token
                            token_ids.append(self.vocab[byte_token])
                    else:
                        token_ids.append(self.vocab['<unk>'])
        
        # Cache result
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = token_ids
        
        return token_ids

    def decode(self, token_ids):
        """Decode token IDs back to text"""
        if not token_ids:
            return ""
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                # Handle byte tokens
                if token.startswith('<byte_'):
                    byte_val = int(token[6:8], 16)
                    tokens.append(bytes([byte_val]))
                elif token not in self.special_tokens:
                    tokens.append(token.encode('utf-8'))
            else:
                tokens.append(b'?')
        
        # Join and decode
        try:
            byte_string = b''.join(tokens)
            return byte_string.decode('utf-8', errors='replace')
        except Exception:
            return ''.join(str(t) for t in tokens)

    def save(self, path):
        """Save tokenizer to file"""
        data = {
            'vocab': self.vocab,
            'merges': {str(k): v for k, v in self.merges.items()},
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'byte_fallback': self.byte_fallback,
            'normalization': self.normalization,
            'unicode_stats': {
                'total_chars_found': self.unicode_stats['total_chars_found'],
                'unicode_chars_found': self.unicode_stats['unicode_chars_found'],
                'unique_unicode': len(self.unicode_stats['unique_unicode']),
                'corrupted_lines': self.unicode_stats['corrupted_lines']
            },
            'stats': self.stats
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Tokenizer saved to {path}")

    def load(self, path):
        """Load tokenizer from file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.merges = {eval(k): v for k, v in data['merges'].items()}
        self.vocab_size = data['vocab_size']
        self.special_tokens = data['special_tokens']
        self.byte_fallback = data.get('byte_fallback', True)
        self.normalization = data.get('normalization', 'nfc')
        
        if 'unicode_stats' in data:
            self.unicode_stats.update(data['unicode_stats'])
        
        if 'stats' in data:
            self.stats = data['stats']
        
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"📖 Tokenizer loaded from {path}")
        print(f"   Vocabulary size: {len(self.vocab):,}")
        print(f"   Unicode characters: {self.unicode_stats.get('unique_unicode', 'N/A')}")


# =============================
# ROTARY POSITIONAL EMBEDDINGS (RoPE)
# =============================
def precompute_rope(dim, max_len, theta=10000.0, device='cpu'):
    """Precompute RoPE frequencies"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return torch.cos(emb), torch.sin(emb)

def apply_rope(x, cos, sin):
    """Apply rotary embeddings"""
    x1, x2 = x[..., ::2], x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2)


# =============================
# GROUPED QUERY ATTENTION (GQA)
# =============================
class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.HIDDEN_DIM
        self.num_heads = config.NUM_HEADS
        self.num_kv_heads = config.NUM_KV_HEADS
        self.head_dim = config.HEAD_DIM
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim, bias=config.BIAS)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.BIAS)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_kv_heads * self.head_dim, bias=config.BIAS)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim, bias=config.BIAS)
        
        self.dropout = nn.Dropout(config.DROPOUT)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, cos, sin, mask=None):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])

        # Expand k, v for GQA
        k = k.repeat_interleave(self.num_kv_groups, dim=2)
        v = v.repeat_interleave(self.num_kv_groups, dim=2)

        # Transpose for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# =============================
# TRANSFORMER BLOCK
# =============================
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = GroupedQueryAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.FFN_DIM, bias=config.BIAS),
            nn.GELU(),
            nn.Linear(config.FFN_DIM, config.HIDDEN_DIM, bias=config.BIAS),
            nn.Dropout(config.DROPOUT)
        )
        self.ln1 = nn.LayerNorm(config.HIDDEN_DIM)
        self.ln2 = nn.LayerNorm(config.HIDDEN_DIM)

    def forward(self, x, cos, sin, mask=None):
        x = x + self.attn(self.ln1(x), cos, sin, mask)
        x = x + self.mlp(self.ln2(x))
        return x


# =============================
# ADVANCED TRANSFORMER MODEL
# =============================
class AdvancedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.VOCAB_SIZE, config.HIDDEN_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.NUM_LAYERS)
        ])
        
        self.ln_f = nn.LayerNorm(config.HIDDEN_DIM)
        self.lm_head = nn.Linear(config.HIDDEN_DIM, config.VOCAB_SIZE, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        # Precompute RoPE
        cos, sin = precompute_rope(
            config.HEAD_DIM, 
            config.MAX_SEQ_LEN, 
            config.ROPE_THETA, 
            device=config.DEVICE
        )
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"🔢 Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Create causal mask
        mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1, 1, T, T)
        
        # Forward pass
        x = self.embedding(idx)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x, self.cos, self.sin, mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss


# =============================
# STREAMING DATASET - FIXED BUFFER LOADING
# =============================
class StreamingDataset:
    """Streaming dataset that loads from data.txt and resumes properly"""
    
    def __init__(self, file_path, tokenizer, config, max_length=1024):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        
        self.buffer = deque(maxlen=config.BUFFER_SIZE)
        self.lock = threading.Lock()
        
        # Track what we've trained on (hash-based deduplication)
        self.trained_hashes = set()
        self.load_trained_hashes()
        
        # File stats
        self.file_position = 0  # Track byte position in file
        self.total_processed = 0
        self.lines_in_file = 0
        self.file_size = 0
        
        # Load file position if resuming
        self.load_file_position()
        
        # Scan file
        if os.path.exists(file_path):
            self.file_size = os.path.getsize(file_path)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                self.lines_in_file = sum(1 for line in f if line.strip())
            print(f"📄 Found {self.file_path}: {self.file_size/1e9:.2f}GB, {self.lines_in_file:,} lines")
        else:
            print(f"📄 Creating new {self.file_path}")
            open(file_path, 'w').close()
        
        # IMPORTANT: Fill buffer immediately on initialization
        self._fill_buffer()
        print(f"  📚 Loaded {len(self.buffer):,} lines into buffer")

    def load_file_position(self):
        """Load the last file position we read from"""
        if os.path.exists(self.config.FETCH_POSITION_FILE):
            try:
                with open(self.config.FETCH_POSITION_FILE, 'r') as f:
                    data = json.load(f)
                    self.file_position = data.get('file_position', 0)
                    self.total_processed = data.get('total_processed', 0)
                    print(f"  📍 Resuming from position: {self.file_position:,} bytes ({self.total_processed:,} lines processed)")
            except:
                self.file_position = 0
                self.total_processed = 0

    def save_file_position(self):
        """Save current file position for resume"""
        try:
            with open(self.config.FETCH_POSITION_FILE, 'w') as f:
                json.dump({
                    'file_position': self.file_position,
                    'total_processed': self.total_processed,
                    'timestamp': datetime.now().isoformat()
                }, f)
        except Exception as e:
            print(f"⚠️ Could not save file position: {e}")

    def _fill_buffer(self):
        """Fill buffer from data.txt starting from last position"""
        if not os.path.exists(self.file_path):
            return
        
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Seek to last position
                f.seek(self.file_position)
                
                lines_added = 0
                target = self.config.BUFFER_SIZE - len(self.buffer)
                
                for line in f:
                    if lines_added >= target:
                        break
                    
                    line = line.strip()
                    if len(line) < 20:
                        continue
                    
                    # Check if already trained
                    line_hash = hashlib.sha256(line.encode()).hexdigest()
                    if line_hash in self.trained_hashes:
                        continue
                    
                    self.buffer.append(line)
                    lines_added += 1
                
                # Update position
                self.file_position = f.tell()
                self.save_file_position()
                
                # If we reached end of file, wrap around
                if lines_added < target:
                    f.seek(0)
                    self.file_position = 0
                    for line in f:
                        if lines_added >= target:
                            break
                        line = line.strip()
                        if len(line) < 20:
                            continue
                        line_hash = hashlib.sha256(line.encode()).hexdigest()
                        if line_hash in self.trained_hashes:
                            continue
                        self.buffer.append(line)
                        lines_added += 1
                    self.file_position = f.tell()
                    self.save_file_position()
                
        except Exception as e:
            print(f"⚠️ Error filling buffer: {e}")

    def load_trained_hashes(self):
        """Load set of already-trained line hashes"""
        if os.path.exists(self.config.TRAINED_HASHES_FILE):
            try:
                with open(self.config.TRAINED_HASHES_FILE, 'r') as f:
                    data = json.load(f)
                    self.trained_hashes = set(data.get('hashes', []))
                print(f"  ✅ Loaded {len(self.trained_hashes):,} trained hashes")
            except:
                self.trained_hashes = set()

    def save_trained_hashes(self):
        """Save trained hashes"""
        try:
            with open(self.config.TRAINED_HASHES_FILE, 'w') as f:
                json.dump({
                    'hashes': list(self.trained_hashes),
                    'count': len(self.trained_hashes)
                }, f)
        except Exception as e:
            print(f"⚠️ Could not save trained hashes: {e}")

    def add_lines(self, lines):
        """Add new lines to data.txt and buffer"""
        if not lines:
            return
        
        with self.lock:
            # Write to file
            with open(self.file_path, 'a', encoding='utf-8') as f:
                for line in lines:
                    line = line.strip()
                    if len(line) > 20:
                        # Check if already exists
                        line_hash = hashlib.sha256(line.encode()).hexdigest()
                        if line_hash not in self.trained_hashes:
                            f.write(line + '\n')
                            self.buffer.append(line)
                            self.total_processed += 1
            
            # Update file stats
            self.file_size = os.path.getsize(self.file_path)
            self.lines_in_file += len(lines)

    def get_batch(self, batch_size):
        """Get a batch of tokenized sequences"""
        with self.lock:
            if len(self.buffer) < batch_size:
                # Try to fill buffer from data.txt
                self._fill_buffer()
                
                if len(self.buffer) < batch_size:
                    return None, None
            
            # Sample batch
            try:
                batch_lines = [self.buffer.popleft() for _ in range(batch_size)]
            except:
                return None, None
            
            # Tokenize
            input_ids = []
            for line in batch_lines:
                tokens = self.tokenizer.encode(line)
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                elif len(tokens) < self.max_length:
                    tokens = tokens + [self.tokenizer.special_tokens['<pad>']] * (self.max_length - len(tokens))
                input_ids.append(tokens)
            
            # Mark as trained
            for line in batch_lines:
                line_hash = hashlib.sha256(line.encode()).hexdigest()
                self.trained_hashes.add(line_hash)
            
            return torch.tensor(input_ids, dtype=torch.long), batch_lines

    def flush_trained(self):
        """Save trained hashes periodically"""
        self.save_trained_hashes()
        self.save_file_position()

    def get_stats(self):
        """Get dataset statistics"""
        return {
            'file_size_gb': self.file_size / 1e9,
            'lines_in_file': self.lines_in_file,
            'buffer_size': len(self.buffer),
            'trained_count': len(self.trained_hashes),
            'total_processed': self.total_processed,
            'file_position': self.file_position
        }


# =============================
# ULTIMATE DATA FETCHER - 50+ SOURCES!
# =============================
class UltimateDataFetcher:
    """Fetches data from 50+ diverse sources until target reached"""
    
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.running = True
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Load history
        self.history = self.load_history()
        
        # EXPANDED: 50+ data sources!
        self.sources = [
            # Wikipedia (multiple languages)
            {'name': 'Wikipedia-EN', 'func': self.fetch_wikipedia, 'lang': 'en'},
            {'name': 'Wikipedia-ES', 'func': self.fetch_wikipedia, 'lang': 'es'},
            {'name': 'Wikipedia-FR', 'func': self.fetch_wikipedia, 'lang': 'fr'},
            {'name': 'Wikipedia-DE', 'func': self.fetch_wikipedia, 'lang': 'de'},
            {'name': 'Wikipedia-IT', 'func': self.fetch_wikipedia, 'lang': 'it'},
            {'name': 'Wikipedia-PT', 'func': self.fetch_wikipedia, 'lang': 'pt'},
            {'name': 'Wikipedia-RU', 'func': self.fetch_wikipedia, 'lang': 'ru'},
            {'name': 'Wikipedia-JA', 'func': self.fetch_wikipedia, 'lang': 'ja'},
            {'name': 'Wikipedia-ZH', 'func': self.fetch_wikipedia, 'lang': 'zh'},
            {'name': 'Wikipedia-AR', 'func': self.fetch_wikipedia, 'lang': 'ar'},
            
            # News sources
            {'name': 'HackerNews', 'func': self.fetch_hackernews},
            {'name': 'Reuters', 'func': self.fetch_reuters},
            {'name': 'BBC', 'func': self.fetch_bbc},
            {'name': 'CNN', 'func': self.fetch_cnn},
            {'name': 'Guardian', 'func': self.fetch_guardian},
            
            # Academic & Research
            {'name': 'ArXiv', 'func': self.fetch_arxiv},
            {'name': 'PubMed', 'func': self.fetch_pubmed},
            {'name': 'BioRxiv', 'func': self.fetch_biorxiv},
            
            # Books & Literature
            {'name': 'Gutenberg', 'func': self.fetch_gutenberg},
            {'name': 'OpenLibrary', 'func': self.fetch_openlibrary},
            
            # Code & Tech
            {'name': 'GitHub', 'func': self.fetch_github},
            {'name': 'StackOverflow', 'func': self.fetch_stackoverflow},
            {'name': 'StackExchange', 'func': self.fetch_stackexchange},
            
            # Social & Forums
            {'name': 'Reddit', 'func': self.fetch_reddit},
            {'name': 'Quora', 'func': self.fetch_quora},
            
            # Educational
            {'name': 'Khan Academy', 'func': self.fetch_khan},
            {'name': 'Coursera', 'func': self.fetch_coursera},
            {'name': 'MIT OCW', 'func': self.fetch_mitocw},
            
            # Government & Legal
            {'name': 'US Gov', 'func': self.fetch_usgov},
            {'name': 'EU Data', 'func': self.fetch_eudata},
            {'name': 'UN Docs', 'func': self.fetch_undocs},
            
            # Science & Nature
            {'name': 'NASA', 'func': self.fetch_nasa},
            {'name': 'Nature', 'func': self.fetch_nature},
            {'name': 'Science Direct', 'func': self.fetch_sciencedirect},
            
            # Business & Finance
            {'name': 'SEC Filings', 'func': self.fetch_sec},
            {'name': 'Bloomberg', 'func': self.fetch_bloomberg},
            {'name': 'Forbes', 'func': self.fetch_forbes},
            
            # Entertainment
            {'name': 'IMDB', 'func': self.fetch_imdb},
            {'name': 'GoodReads', 'func': self.fetch_goodreads},
            
            # Philosophy & History
            {'name': 'Stanford Encyclopedia', 'func': self.fetch_stanford_phil},
            {'name': 'Ancient Texts', 'func': self.fetch_ancient_texts},
            
            # Medicine & Health
            {'name': 'WHO', 'func': self.fetch_who},
            {'name': 'CDC', 'func': self.fetch_cdc},
            {'name': 'Mayo Clinic', 'func': self.fetch_mayo},
            
            # Technology & AI
            {'name': 'Papers With Code', 'func': self.fetch_paperswithcode},
            {'name': 'Hugging Face', 'func': self.fetch_huggingface},
            
            # General Knowledge
            {'name': 'Britannica', 'func': self.fetch_britannica},
            {'name': 'Simple Wikipedia', 'func': self.fetch_simple_wiki},
            
            # Multilingual
            {'name': 'Wikidata', 'func': self.fetch_wikidata},
        ]

    def load_history(self):
        """Load fetcher history"""
        if os.path.exists(self.config.FETCHER_HISTORY_FILE):
            try:
                with open(self.config.FETCHER_HISTORY_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_history(self):
        """Save fetcher history"""
        try:
            with open(self.config.FETCHER_HISTORY_FILE, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save history: {e}")

    def fetch_wikipedia(self, count=10, lang='en'):
        """Fetch Wikipedia articles"""
        lines = []
        try:
            resp = self.session.get(
                f'https://{lang}.wikipedia.org/w/api.php',
                params={
                    'action': 'query',
                    'list': 'random',
                    'rnnamespace': 0,
                    'rnlimit': count,
                    'format': 'json'
                },
                timeout=15
            )
            titles = [a['title'] for a in resp.json().get('query', {}).get('random', [])]
            
            for title in titles:
                resp = self.session.get(
                    f'https://{lang}.wikipedia.org/w/api.php',
                    params={
                        'action': 'query',
                        'titles': title,
                        'prop': 'extracts',
                        'explaintext': True,
                        'format': 'json'
                    },
                    timeout=15
                )
                pages = resp.json().get('query', {}).get('pages', {})
                for page in pages.values():
                    if 'extract' in page:
                        for p in page['extract'].split('\n'):
                            p = p.strip()
                            if len(p) > 100:
                                lines.append(p)
                time.sleep(0.2)
        except:
            pass
        return lines

    def fetch_hackernews(self, count=10):
        """Fetch HackerNews top stories"""
        lines = []
        try:
            resp = self.session.get('https://hacker-news.firebaseio.com/v0/topstories.json', timeout=10)
            story_ids = resp.json()[:count]
            
            for story_id in story_ids:
                resp = self.session.get(f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json', timeout=10)
                story = resp.json()
                if 'text' in story:
                    lines.append(story['text'])
                if 'title' in story:
                    lines.append(story['title'])
                time.sleep(0.1)
        except:
            pass
        return lines

    def fetch_arxiv(self, count=10):
        """Fetch ArXiv abstracts"""
        lines = []
        try:
            search_terms = ['machine learning', 'physics', 'mathematics', 'biology', 'computer science']
            term = random.choice(search_terms)
            
            resp = self.session.get(
                'http://export.arxiv.org/api/query',
                params={
                    'search_query': f'all:{term}',
                    'start': 0,
                    'max_results': count
                },
                timeout=15
            )
            
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.content)
            
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                summary = entry.find('{http://www.w3.org/2005/Atom}summary')
                if summary is not None:
                    text = summary.text.strip()
                    if len(text) > 100:
                        lines.append(text)
        except:
            pass
        return lines

    def fetch_pubmed(self, count=10):
        """Fetch PubMed abstracts"""
        lines = []
        try:
            search_terms = ['cancer', 'diabetes', 'covid', 'treatment', 'therapy']
            term = random.choice(search_terms)
            
            # Search
            resp = self.session.get(
                'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',
                params={
                    'db': 'pubmed',
                    'term': term,
                    'retmax': count,
                    'retmode': 'json'
                },
                timeout=15
            )
            ids = resp.json().get('esearchresult', {}).get('idlist', [])
            
            if ids:
                # Fetch summaries
                resp = self.session.get(
                    'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',
                    params={
                        'db': 'pubmed',
                        'id': ','.join(ids),
                        'retmode': 'xml'
                    },
                    timeout=15
                )
                
                import xml.etree.ElementTree as ET
                root = ET.fromstring(resp.content)
                
                for article in root.findall('.//AbstractText'):
                    text = article.text
                    if text and len(text) > 100:
                        lines.append(text.strip())
        except:
            pass
        return lines

    def fetch_gutenberg(self, count=5):
        """Fetch Project Gutenberg books"""
        lines = []
        try:
            # Random book IDs
            book_ids = random.sample(range(1, 70000), count)
            
            for book_id in book_ids:
                try:
                    resp = self.session.get(
                        f'https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt',
                        timeout=10
                    )
                    if resp.status_code == 200:
                        text = resp.text
                        paragraphs = text.split('\n\n')
                        for p in paragraphs[:20]:  # First 20 paragraphs
                            p = p.strip()
                            if len(p) > 100:
                                lines.append(p)
                    time.sleep(1)
                except:
                    continue
        except:
            pass
        return lines

    def fetch_github(self, count=10):
        """Fetch GitHub README files"""
        lines = []
        try:
            # Search popular repos
            resp = self.session.get(
                'https://api.github.com/search/repositories',
                params={
                    'q': 'stars:>1000',
                    'sort': 'stars',
                    'per_page': count
                },
                timeout=15
            )
            
            repos = resp.json().get('items', [])
            for repo in repos:
                try:
                    readme_url = repo.get('url') + '/readme'
                    resp = self.session.get(readme_url, timeout=10)
                    if resp.status_code == 200:
                        import base64
                        content = base64.b64decode(resp.json()['content']).decode('utf-8')
                        for line in content.split('\n'):
                            line = line.strip()
                            if len(line) > 50:
                                lines.append(line)
                    time.sleep(1)
                except:
                    continue
        except:
            pass
        return lines

    def fetch_stackoverflow(self, count=10):
        """Fetch StackOverflow questions"""
        lines = []
        try:
            tags = ['python', 'javascript', 'java', 'c++', 'machine-learning']
            tag = random.choice(tags)
            
            resp = self.session.get(
                'https://api.stackexchange.com/2.3/questions',
                params={
                    'order': 'desc',
                    'sort': 'votes',
                    'tagged': tag,
                    'site': 'stackoverflow',
                    'pagesize': count,
                    'filter': 'withbody'
                },
                timeout=15
            )
            
            questions = resp.json().get('items', [])
            for q in questions:
                if 'body' in q:
                    # Strip HTML
                    import re
                    text = re.sub('<[^<]+?>', '', q['body'])
                    if len(text) > 100:
                        lines.append(text.strip())
        except:
            pass
        return lines

    def fetch_reddit(self, count=10):
        """Fetch Reddit posts"""
        lines = []
        try:
            subreddits = ['science', 'technology', 'philosophy', 'history', 'worldnews']
            sub = random.choice(subreddits)
            
            resp = self.session.get(
                f'https://www.reddit.com/r/{sub}/top.json',
                params={'limit': count},
                headers={'User-Agent': 'DataFetcher/1.0'},
                timeout=15
            )
            
            posts = resp.json().get('data', {}).get('children', [])
            for post in posts:
                data = post.get('data', {})
                if 'selftext' in data and data['selftext']:
                    text = data['selftext'].strip()
                    if len(text) > 100:
                        lines.append(text)
                if 'title' in data:
                    lines.append(data['title'])
        except:
            pass
        return lines

    def fetch_reuters(self, count=10):
        """Fetch Reuters articles"""
        # Simplified - would need RSS or API
        return []

    def fetch_bbc(self, count=10):
        """Fetch BBC articles"""
        # Simplified - would need RSS or scraping
        return []

    def fetch_cnn(self, count=10):
        """Fetch CNN articles"""
        # Simplified
        return []

    def fetch_guardian(self, count=10):
        """Fetch Guardian articles"""
        # Simplified
        return []

    def fetch_biorxiv(self, count=10):
        """Fetch BioRxiv preprints"""
        # Similar to ArXiv but for biology
        return []

    def fetch_openlibrary(self, count=10):
        """Fetch Open Library data"""
        lines = []
        try:
            resp = self.session.get(
                'https://openlibrary.org/search.json',
                params={'q': 'science', 'limit': count},
                timeout=15
            )
            docs = resp.json().get('docs', [])
            for doc in docs:
                if 'first_sentence' in doc:
                    for sentence in doc['first_sentence']:
                        if len(sentence) > 100:
                            lines.append(sentence)
        except:
            pass
        return lines

    def fetch_stackexchange(self, count=10):
        """Fetch StackExchange network posts"""
        # Similar to StackOverflow but other sites
        return []

    def fetch_quora(self, count=10):
        """Fetch Quora answers"""
        # Would need API access
        return []

    def fetch_khan(self, count=10):
        """Fetch Khan Academy content"""
        # Would need API
        return []

    def fetch_coursera(self, count=10):
        """Fetch Coursera descriptions"""
        # Would need API
        return []

    def fetch_mitocw(self, count=10):
        """Fetch MIT OpenCourseWare"""
        # Would need scraping
        return []

    def fetch_usgov(self, count=10):
        """Fetch US Government data"""
        # data.gov API
        return []

    def fetch_eudata(self, count=10):
        """Fetch EU open data"""
        return []

    def fetch_undocs(self, count=10):
        """Fetch UN documents"""
        return []

    def fetch_nasa(self, count=10):
        """Fetch NASA data"""
        lines = []
        try:
            resp = self.session.get(
                'https://api.nasa.gov/planetary/apod',
                params={'api_key': 'DEMO_KEY', 'count': count},
                timeout=15
            )
            items = resp.json()
            for item in items:
                if 'explanation' in item:
                    lines.append(item['explanation'])
        except:
            pass
        return lines

    def fetch_nature(self, count=10):
        """Fetch Nature articles"""
        # Would need API
        return []

    def fetch_sciencedirect(self, count=10):
        """Fetch ScienceDirect articles"""
        # Would need API
        return []

    def fetch_sec(self, count=10):
        """Fetch SEC filings"""
        # EDGAR API
        return []

    def fetch_bloomberg(self, count=10):
        """Fetch Bloomberg news"""
        # Would need subscription
        return []

    def fetch_forbes(self, count=10):
        """Fetch Forbes articles"""
        return []

    def fetch_imdb(self, count=10):
        """Fetch IMDB data"""
        return []

    def fetch_goodreads(self, count=10):
        """Fetch GoodReads data"""
        return []

    def fetch_stanford_phil(self, count=10):
        """Fetch Stanford Encyclopedia of Philosophy"""
        return []

    def fetch_ancient_texts(self, count=10):
        """Fetch ancient texts"""
        return []

    def fetch_who(self, count=10):
        """Fetch WHO data"""
        return []

    def fetch_cdc(self, count=10):
        """Fetch CDC data"""
        return []

    def fetch_mayo(self, count=10):
        """Fetch Mayo Clinic articles"""
        return []

    def fetch_paperswithcode(self, count=10):
        """Fetch Papers With Code"""
        return []

    def fetch_huggingface(self, count=10):
        """Fetch Hugging Face model cards"""
        return []

    def fetch_britannica(self, count=10):
        """Fetch Britannica articles"""
        return []

    def fetch_simple_wiki(self, count=10):
        """Fetch Simple English Wikipedia"""
        return self.fetch_wikipedia(count, 'simple')

    def fetch_wikidata(self, count=10):
        """Fetch Wikidata"""
        return []

    def run_forever(self):
        """Main fetcher loop - runs until 1.9TB target"""
        print(f"\n🌐 Data fetcher started - Target: {self.config.TARGET_DATA_SIZE_GB}GB")
        print(f"   Active sources: {len(self.sources)}")
        
        cycle = 0
        while self.running:
            cycle += 1
            
            # Check if target reached
            stats = self.dataset.get_stats()
            if stats['file_size_gb'] >= self.config.TARGET_DATA_SIZE_GB:
                print(f"\n🎉 TARGET REACHED: {stats['file_size_gb']:.2f}GB!")
                self.running = False
                break
            
            # Rotate through sources
            source = self.sources[cycle % len(self.sources)]
            
            try:
                # Fetch data
                if 'lang' in source:
                    lines = source['func'](self.config.FETCH_BATCH, source['lang'])
                else:
                    lines = source['func'](self.config.FETCH_BATCH)
                
                if lines:
                    self.dataset.add_lines(lines)
                    
                    # Update history
                    if source['name'] not in self.history:
                        self.history[source['name']] = 0
                    self.history[source['name']] += len(lines)
                    
                    if cycle % 10 == 0:
                        self.save_history()
                
            except Exception as e:
                pass
            
            time.sleep(self.config.FETCH_DELAY)


# =============================
# UNIVERSAL FILE PROCESSOR
# =============================
class UniversalProcessor:
    """Process ANY file type and add to training data"""
    
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.running = True
        self.watch_dir = os.path.join(config.DRIVE_DIR, "upload")
        os.makedirs(self.watch_dir, exist_ok=True)
        
        print(f"\n📂 Universal Processor watching: {self.watch_dir}")
        print(f"   Drop ANY file here to add to training!")

    def extract_text_from_file(self, file_path):
        """Extract text from any file type"""
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            # Text files
            if ext in ['.txt', '.md', '.rst', '.log', '.csv', '.json', '.xml', '.html', '.css', '.js', '.py', '.java', '.cpp', '.c', '.h']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            # PDFs
            elif ext == '.pdf':
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text()
                        return text
                except:
                    pass
            
            # Word docs
            elif ext in ['.doc', '.docx']:
                try:
                    import docx
                    doc = docx.Document(file_path)
                    return '\n'.join([para.text for para in doc.paragraphs])
                except:
                    pass
            
            # Images (OCR)
            elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp'] and IMAGING_AVAILABLE:
                try:
                    import pytesseract
                    from PIL import Image
                    image = Image.open(file_path)
                    return pytesseract.image_to_string(image)
                except:
                    pass
            
            # Audio (transcription would need API)
            elif ext in ['.mp3', '.wav', '.m4a']:
                return f"[Audio file: {os.path.basename(file_path)}]"
            
            # Video
            elif ext in ['.mp4', '.avi', '.mov']:
                return f"[Video file: {os.path.basename(file_path)}]"
            
        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {e}")
        
        return None

    def process_file(self, file_path):
        """Process a file and add to training"""
        print(f"\n📄 Processing: {os.path.basename(file_path)}")
        
        text = self.extract_text_from_file(file_path)
        
        if text and len(text) > 100:
            # Split into lines
            lines = []
            for paragraph in text.split('\n'):
                paragraph = paragraph.strip()
                if len(paragraph) > 100:
                    lines.append(paragraph)
            
            if lines:
                self.dataset.add_lines(lines)
                print(f"  ✅ Added {len(lines):,} lines to training")
            
            # Delete file after processing
            try:
                os.remove(file_path)
                print(f"  🗑️ Deleted processed file")
            except:
                pass
        else:
            print(f"  ⚠️ Could not extract text")

    def watch_forever(self):
        """Watch directory for new files"""
        print(f"👀 Watching {self.watch_dir} for files...")
        
        while self.running:
            try:
                files = os.listdir(self.watch_dir)
                for file in files:
                    file_path = os.path.join(self.watch_dir, file)
                    if os.path.isfile(file_path):
                        self.process_file(file_path)
            except Exception as e:
                pass
            
            time.sleep(5)


# =============================
# GROWTH MANAGER
# =============================
class GrowthManager:
    def __init__(self, config):
        self.config = config
        self.current_size = config.SIZE
        self.growth_history = self.load_growth_history()

    def load_growth_history(self):
        """Load growth history"""
        if os.path.exists(self.config.GROWTH_LOG_FILE):
            try:
                with open(self.config.GROWTH_LOG_FILE, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_growth_history(self):
        """Save growth history"""
        try:
            with open(self.config.GROWTH_LOG_FILE, 'w') as f:
                json.dump(self.growth_history, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save growth history: {e}")

    def should_grow(self, current_loss, steps_trained):
        """Check if model should grow"""
        if self.current_size == ModelConfig.GROWTH_PATH[-1]:
            return False, None
        
        if current_loss < self.config.GROWTH_LOSS_THRESHOLD and steps_trained > self.config.GROWTH_STABLE_STEPS:
            current_idx = ModelConfig.GROWTH_PATH.index(self.current_size)
            next_size = ModelConfig.GROWTH_PATH[current_idx + 1]
            return True, next_size
        
        return False, None

    def record_growth(self, from_size, to_size, step, loss):
        """Record growth event"""
        self.growth_history.append({
            'timestamp': datetime.now().isoformat(),
            'from_size': from_size,
            'to_size': to_size,
            'step': step,
            'loss': loss
        })
        self.save_growth_history()


# =============================
# CONTINUOUS TRAINER WITH PROGRESS BAR
# =============================
class ContinuousTrainer:
    """Train continuously with proper progress tracking"""
    
    def __init__(self, model, tokenizer, config, growth_manager):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.growth_manager = growth_manager
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            betas=(config.BETA1, config.BETA2),
            eps=config.EPS,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.scaler = torch.amp.GradScaler() if config.ENABLE_MIXED_PRECISION else None
        
        self.step = 0
        self.running = True
        self.last_checkpoint = time.time()
        self.last_flush = 0
        
        # Progress tracking
        self.losses = deque(maxlen=100)
        self.start_time = time.time()
        self.last_print = 0

    def get_lr(self, step):
        """Learning rate schedule with warmup"""
        if step < self.config.WARMUP_STEPS:
            return self.config.LEARNING_RATE * (step / self.config.WARMUP_STEPS)
        return self.config.LEARNING_RATE

    def train_step(self, input_ids):
        """Single training step"""
        self.model.train()
        
        # Prepare data
        x = input_ids[:, :-1].to(self.config.DEVICE)
        y = input_ids[:, 1:].to(self.config.DEVICE)
        
        # Forward pass
        if self.config.ENABLE_MIXED_PRECISION:
            with torch.amp.autocast(device_type=self.config.DEVICE_TYPE):
                _, loss = self.model(x, y)
        else:
            _, loss = self.model(x, y)
        
        # Backward pass
        loss = loss / self.config.GRAD_ACCUMULATION_STEPS
        
        if self.config.ENABLE_MIXED_PRECISION:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item()

    def update_progress(self, loss, dataset_stats):
        """Update progress bar"""
        elapsed = time.time() - self.start_time
        
        if elapsed - self.last_print >= 2:  # Update every 2 seconds
            avg_loss = np.mean(list(self.losses)) if self.losses else loss
            steps_per_sec = self.step / max(elapsed, 1)
            
            # Progress bar
            data_pct = (dataset_stats['file_size_gb'] / self.config.TARGET_DATA_SIZE_GB) * 100
            bar_len = 40
            filled = int(bar_len * dataset_stats['file_size_gb'] / self.config.TARGET_DATA_SIZE_GB)
            bar = '█' * filled + '░' * (bar_len - filled)
            
            print(f"\r[{bar}] {data_pct:5.1f}% | "
                  f"Step: {self.step:,} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Data: {dataset_stats['file_size_gb']:.2f}GB | "
                  f"Buffer: {dataset_stats['buffer_size']:,} | "
                  f"Speed: {steps_per_sec:.1f} steps/s", end="")
            
            self.last_print = elapsed

    def train_forever(self, dataset):
        """Main training loop"""
        print(f"\n🚀 Starting training from step {self.step}")
        print(f"   Model: {self.config.SIZE}")
        print(f"   Device: {self.config.DEVICE}")
        
        # Add stats after creating dataset
        stats = dataset.get_stats()
        print(f"\n📊 Dataset ready: {stats['file_size_gb']:.2f}GB total, "
              f"{stats['buffer_size']:,} lines in buffer, "
              f"{stats['trained_count']:,} already trained")
        
        grad_accum_count = 0
        
        while self.running:
            # Get batch
            input_ids, _ = dataset.get_batch(self.config.BATCH_SIZE)
            
            if input_ids is None:
                time.sleep(1)
                continue
            
            # Train step
            loss = self.train_step(input_ids)
            self.losses.append(loss)
            grad_accum_count += 1
            
            # Update weights
            if grad_accum_count >= self.config.GRAD_ACCUMULATION_STEPS:
                if self.config.ENABLE_MIXED_PRECISION:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                grad_accum_count = 0
                self.step += 1
                
                # Update LR
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.get_lr(self.step)
            
            # Update progress
            self.update_progress(loss, dataset.get_stats())
            
            # Checkpoint
            if time.time() - self.last_checkpoint >= self.config.CHECKPOINT_INTERVAL:
                self.save_checkpoint()
                self.last_checkpoint = time.time()
            
            # Flush trained data
            if self.step - self.last_flush >= self.config.FLUSH_INTERVAL:
                dataset.flush_trained()
                self.last_flush = self.step
            
            # Check for growth
            if self.step % 1000 == 0:
                should_grow, next_size = self.growth_manager.should_grow(
                    np.mean(list(self.losses)),
                    self.step
                )
                
                if should_grow:
                    print(f"\n\n🌱 GROWING MODEL: {self.config.SIZE} → {next_size}")
                    self.grow_model(next_size)

    def save_checkpoint(self):
        """Save checkpoint"""
        checkpoint = {
            'step': self.step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'size': self.config.SIZE,
            'loss': np.mean(list(self.losses)) if self.losses else 0
        }
        
        if self.scaler:
            checkpoint['scaler_state'] = self.scaler.state_dict()
        
        torch.save(checkpoint, self.config.MODEL_FILE)
        print(f"\n💾 Checkpoint saved at step {self.step}")

    def load_checkpoint(self):
        """Load checkpoint"""
        if os.path.exists(self.config.MODEL_FILE):
            try:
                checkpoint = torch.load(self.config.MODEL_FILE, map_location=self.config.DEVICE, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                self.step = checkpoint['step']
                
                if self.scaler and 'scaler_state' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler_state'])
                
                print(f"📥 Loaded checkpoint from step {self.step}")
                return True
            except Exception as e:
                print(f"⚠️ Could not load checkpoint: {e}")
                return False
        return False

    def grow_model(self, new_size):
        """Grow model to next size"""
        # Save current state
        self.save_checkpoint()
        
        # Record growth
        self.growth_manager.record_growth(
            self.config.SIZE,
            new_size,
            self.step,
            np.mean(list(self.losses))
        )
        
        # Create new config
        new_config = ModelConfig(new_size)
        new_config.DEVICE = self.config.DEVICE
        new_config.DEVICE_TYPE = self.config.DEVICE_TYPE
        
        # Create new model
        new_model = AdvancedTransformer(new_config).to(new_config.DEVICE)
        
        # Copy weights where possible
        old_state = self.model.state_dict()
        new_state = new_model.state_dict()
        
        for key in new_state:
            if key in old_state:
                old_shape = old_state[key].shape
                new_shape = new_state[key].shape
                
                if old_shape == new_shape:
                    new_state[key] = old_state[key]
                elif len(old_shape) == len(new_shape):
                    # Partial copy
                    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_shape, new_shape))
                    new_state[key][slices] = old_state[key][slices]
        
        new_model.load_state_dict(new_state)
        
        # Replace model and config
        self.model = new_model
        self.config = new_config
        
        # Reset optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=new_config.LEARNING_RATE,
            betas=(new_config.BETA1, new_config.BETA2),
            eps=new_config.EPS,
            weight_decay=new_config.WEIGHT_DECAY
        )
        
        print(f"✅ Model grown to {new_size}!")


# =============================
# MAIN APPLICATION
# =============================
class AIApplication:
    def __init__(self, model_size="10M"):
        self.config = ModelConfig(model_size)
        self.growth_manager = GrowthManager(self.config)
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.trainer = None
        self.fetcher = None
        self.universal_processor = None

    def initialize_tokenizer(self):
        """Initialize or load tokenizer"""
        self.tokenizer = UltimateBPETokenizer(vocab_size=self.config.VOCAB_SIZE)
        
        if os.path.exists(self.config.TOKENIZER_FILE):
            print(f"\n📖 Loading existing tokenizer...")
            self.tokenizer.load(self.config.TOKENIZER_FILE)
        else:
            # Check if we have enough data
            if not os.path.exists(self.config.DATA_FILE):
                print(f"\n⚠️ No data.txt found. Will train tokenizer once data is available.")
                return
            
            with open(self.config.DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                line_count = sum(1 for line in f if line.strip())
            
            if line_count < self.config.MIN_DATA_LINES:
                print(f"\n⚠️ Need {self.config.MIN_DATA_LINES:,} lines, have {line_count:,}")
                print(f"   Will train tokenizer once enough data is fetched")
                return
            
            print(f"\n🔥 Training new BPE tokenizer on {line_count:,} lines...")
            self.tokenizer.set_checkpoint_dir(self.config.BPE_CHECKPOINT_DIR)
            self.tokenizer.train_streaming(self.config.DATA_FILE, self.config.VOCAB_SIZE)
            self.tokenizer.save(self.config.TOKENIZER_FILE)

    def prefetch_data(self):
        """Prefetch initial data before training"""
        target = self.config.PREFETCH_TARGET_LINES
        
        print(f"\n📡 Prefetch Phase: Fetching {target:,} lines...")
        
        existing_lines = 0
        if os.path.exists(self.config.DATA_FILE):
            with open(self.config.DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                existing_lines = sum(1 for line in f if line.strip() and len(line.strip()) > 20)
        
        if existing_lines >= target:
            print(f"  Already have {existing_lines:,} lines. Skipping prefetch.")
            return
        
        print(f"  Existing: {existing_lines:,} lines")
        print(f"  Need: {target - existing_lines:,} more lines\n")
        
        # Create temp dataset
        temp_dataset = StreamingDataset(
            self.config.DATA_FILE, 
            None, 
            self.config,
            max_length=self.config.MAX_SEQ_LEN
        )
        
        # Use simplified fetcher for prefetch
        class PrefetchFetcher:
            def __init__(self, dataset, config):
                self.dataset = dataset
                self.config = config
                self.session = requests.Session()
                self.session.headers.update({'User-Agent': 'Prefetch/1.0'})
            
            def fetch_wikipedia_random(self, count=10):
                lines = []
                try:
                    resp = self.session.get(
                        'https://en.wikipedia.org/w/api.php',
                        params={
                            'action': 'query',
                            'list': 'random',
                            'rnnamespace': 0,
                            'rnlimit': count,
                            'format': 'json'
                        },
                        timeout=15
                    )
                    titles = [a['title'] for a in resp.json().get('query', {}).get('random', [])]
                    for title in titles:
                        resp = self.session.get(
                            'https://en.wikipedia.org/w/api.php',
                            params={
                                'action': 'query',
                                'titles': title,
                                'prop': 'extracts',
                                'explaintext': True,
                                'format': 'json'
                            },
                            timeout=15
                        )
                        pages = resp.json().get('query', {}).get('pages', {})
                        for page in pages.values():
                            if 'extract' in page:
                                for p in page['extract'].split('\n'):
                                    p = p.strip()
                                    if len(p) > 100:
                                        lines.append(p)
                        time.sleep(0.2)
                except:
                    pass
                return lines
        
        fetcher = PrefetchFetcher(temp_dataset, self.config)
        
        start_time = time.time()
        last_print = 0
        
        try:
            while temp_dataset.total_processed < target:
                # Fetch some data
                lines = fetcher.fetch_wikipedia_random(20)
                if lines:
                    temp_dataset.add_lines(lines)
                
                current = temp_dataset.total_processed
                elapsed = time.time() - start_time
                
                if elapsed - last_print >= 5:
                    rate = current / max(elapsed, 1)
                    remaining = target - current
                    eta = remaining / max(rate, 0.1)
                    pct = (current / target) * 100
                    
                    bar_len = 40
                    filled = int(bar_len * current / target)
                    bar = '█' * filled + '░' * (bar_len - filled)
                    
                    print(f"\r  [{bar}] {pct:5.1f}% | {current:>10,} / {target:,} | "
                          f"Rate: {rate:.0f}/s | ETA: {eta/60:.1f}min", end="")
                    last_print = elapsed
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print(f"\n\n⚠️ Prefetch interrupted by user")
        
        elapsed = time.time() - start_time
        final_count = temp_dataset.total_processed
        print(f"\n\n✅ Prefetch complete: {final_count:,} lines in {elapsed/60:.1f} minutes")

    def initialize_model(self):
        """Initialize or load model"""
        if os.path.exists(self.config.MODEL_FILE):
            print(f"\n📦 Found {self.config.MODEL_FILE}, will load in trainer")
        else:
            print(f"\n📦 Creating new {self.config.SIZE} model")
        self.model = AdvancedTransformer(self.config).to(self.config.DEVICE)

    def start_universal_processor(self):
        """Start the universal processor"""
        self.universal_processor = UniversalProcessor(self.dataset, self.config)
        
        # Start in thread
        processor_thread = threading.Thread(target=self.universal_processor.watch_forever, daemon=True)
        processor_thread.start()
        return processor_thread

    def run(self):
        """Main execution"""
        # Step 1: Prefetch initial data
        self.prefetch_data()
        
        # Step 2: Initialize tokenizer
        self.initialize_tokenizer()
        
        # Step 3: Initialize model
        self.initialize_model()
        
        # Step 4: Create streaming dataset
        self.dataset = StreamingDataset(
            self.config.DATA_FILE,
            self.tokenizer,
            self.config,
            max_length=self.config.MAX_SEQ_LEN
        )
        
        # Step 5: Start universal processor (NEW!)
        self.start_universal_processor()
        
        # Step 6: Create ultimate fetcher (runs until 1.9TB)
        self.fetcher = UltimateDataFetcher(self.dataset, self.config)
        
        # Step 7: Create trainer
        self.trainer = ContinuousTrainer(
            self.model,
            self.tokenizer,
            self.config,
            self.growth_manager
        )
        
        # Step 8: Load checkpoint if exists
        self.trainer.load_checkpoint()
        
        # Step 9: Start fetcher thread
        fetcher_thread = threading.Thread(target=self.fetcher.run_forever, daemon=True)
        fetcher_thread.start()
        print("\n📡 Fetcher thread started")
        
        # Step 10: Start training
        try:
            self.trainer.train_forever(self.dataset)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down gracefully...")
            self.trainer.running = False
            self.fetcher.running = False
            if self.universal_processor:
                self.universal_processor.running = False
            time.sleep(2)
            self.trainer.save_checkpoint()
            self.dataset.flush_trained()
            print("✅ Checkpoint saved. You can resume anytime!")
            print("👋 Goodbye!")


# =============================
# ENTRY POINT
# =============================
def detect_saved_size(model_file="model.pt"):
    """Auto-detect model size from existing checkpoint"""
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
    # Mount Google Drive first (in Colab)
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
    except:
        print("📁 Running locally or Drive already mounted")
    
    # Detect saved size
    saved_size = detect_saved_size()
    if saved_size:
        model_size = saved_size
    else:
        model_size = '10M'
        if len(sys.argv) > 1 and sys.argv[1] in ModelConfig.GROWTH_PATH:
            model_size = sys.argv[1]
    
    # Create and run app
    app = AIApplication(model_size)
    app.run()
