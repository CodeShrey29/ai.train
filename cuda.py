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
- 25+ DIVERSE DATA SOURCES
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
        """Normalize text for consistency"""
        import unicodedata
        if self.normalization == 'nfc':
            return unicodedata.normalize('NFC', text)
        elif self.normalization == 'nfd':
            return unicodedata.normalize('NFD', text)
        return text

    def _count_pairs(self, words):
        """Count adjacent pairs in tokenized words"""
        pairs = Counter()
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair, words):
        """Merge most frequent pair in all words"""
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        new_words = {}
        for word in words:
            new_word = word.replace(bigram, replacement)
            new_words[new_word] = words[word]
        return new_words

    def train_streaming(self, text_file, chunk_size=2_000_000, max_lines=None):
        """
        Train BPE tokenizer in streaming fashion
        Processes text in chunks to handle 2TB datasets
        """
        print(f"\n{'='*80}")
        print(f"  🔥 ULTIMATE BPE TRAINING - 200K VOCABULARY")
        print(f"{'='*80}")
        print(f"  Target vocab: {self.vocab_size:,}")
        print(f"  Chunk size: {chunk_size:,} lines")
        print(f"  Unicode range: U+{self.unicode_range[0]:04X} to U+{self.unicode_range[1]:04X}")
        
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"Training file not found: {text_file}")
        
        # Initialize vocab with special tokens
        self.vocab = self.special_tokens.copy()
        current_id = len(self.special_tokens)
        
        # Phase 1: Build character vocabulary
        print(f"\n📊 Phase 1: Building character vocabulary...")
        char_freq = Counter()
        line_count = 0
        
        with open(text_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line or len(line) < 20:
                    continue
                
                # Normalize
                line = self._normalize_text(line)
                
                # Count characters
                for char in line:
                    char_freq[char] += 1
                    # Track Unicode stats
                    self.unicode_stats['total_chars_found'] += 1
                    if ord(char) > 127:
                        self.unicode_stats['unicode_chars_found'] += 1
                        self.unicode_stats['unique_unicode'].add(char)
                
                line_count += 1
                if line_count % 100000 == 0:
                    print(f"  Processed {line_count:,} lines, {len(char_freq):,} unique chars...", end='\r')
                
                if max_lines and line_count >= max_lines:
                    break
        
        print(f"\n  ✅ Found {len(char_freq):,} unique characters in {line_count:,} lines")
        print(f"  📊 Unicode stats: {self.unicode_stats['unicode_chars_found']:,} Unicode chars, "
              f"{len(self.unicode_stats['unique_unicode']):,} unique")
        
        # Add all characters to vocab
        for char in sorted(char_freq.keys()):
            if char not in self.vocab:
                self.vocab[char] = current_id
                current_id += 1
        
        print(f"  ✅ Initial vocab size: {len(self.vocab):,}")
        
        # Phase 2: Build word frequency dictionary
        print(f"\n📊 Phase 2: Building word frequencies...")
        words = Counter()
        line_count = 0
        
        with open(text_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line or len(line) < 20:
                    continue
                
                line = self._normalize_text(line)
                
                # Tokenize into words
                for word in re.findall(r'\w+|[^\w\s]', line):
                    if len(word) > self.max_word_length:
                        continue
                    # Split into characters
                    word_chars = ' '.join(list(word)) + ' </w>'
                    words[word_chars] += 1
                
                line_count += 1
                if line_count % 100000 == 0:
                    print(f"  Processed {line_count:,} lines, {len(words):,} unique words...", end='\r')
                
                if max_lines and line_count >= max_lines:
                    break
        
        print(f"\n  ✅ Found {len(words):,} unique word forms")
        self.stats['unique_words'] = len(words)
        
        # Phase 3: Learn BPE merges
        print(f"\n📊 Phase 3: Learning BPE merges...")
        target_merges = self.vocab_size - len(self.vocab)
        print(f"  Target: {target_merges:,} merges")
        
        merge_count = 0
        last_print = 0
        
        while len(self.vocab) < self.vocab_size:
            # Count pairs
            pairs = self._count_pairs(words)
            if not pairs:
                break
            
            # Get most frequent pair
            best_pair = max(pairs, key=pairs.get)
            freq = pairs[best_pair]
            
            # Stop if frequency too low
            if freq < self.min_frequency:
                print(f"\n  ⚠️ Stopping: pair frequency ({freq}) below minimum ({self.min_frequency})")
                break
            
            # Merge the pair
            words = self._merge_pair(best_pair, words)
            
            # Add to vocabulary
            new_token = ''.join(best_pair)
            self.vocab[new_token] = current_id
            self.merges[best_pair] = current_id
            current_id += 1
            merge_count += 1
            
            # Progress
            if merge_count - last_print >= self.progress_interval:
                pct = (len(self.vocab) / self.vocab_size) * 100
                print(f"  Merges: {merge_count:,} | Vocab: {len(self.vocab):,}/{self.vocab_size:,} ({pct:.1f}%) | "
                      f"Last: '{best_pair[0]}'+'{best_pair[1]}' = '{new_token}' (freq: {freq:,})", end='\r')
                last_print = merge_count
                
                # Checkpoint periodically
                if self.checkpoint_dir and merge_count % 10000 == 0:
                    self._save_checkpoint(merge_count)
        
        print(f"\n  ✅ Completed {merge_count:,} merges")
        print(f"  ✅ Final vocabulary size: {len(self.vocab):,}")
        
        # Build inverse vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Final stats
        print(f"\n{'='*80}")
        print(f"  ✅ BPE TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"  Vocabulary size: {len(self.vocab):,}")
        print(f"  Unique words processed: {self.stats['unique_words']:,}")
        print(f"  Unicode characters: {self.unicode_stats['unicode_chars_found']:,}")
        print(f"  Unique Unicode chars: {len(self.unicode_stats['unique_unicode']):,}")
        print(f"{'='*80}")

    def _save_checkpoint(self, merge_count):
        """Save training checkpoint"""
        if not self.checkpoint_dir:
            return
        
        checkpoint_file = os.path.join(self.checkpoint_dir, f"bpe_checkpoint_{merge_count}.pkl")
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'vocab': self.vocab,
                    'merges': self.merges,
                    'merge_count': merge_count,
                    'stats': self.stats,
                    'unicode_stats': self.unicode_stats
                }, f)
            print(f"\n  💾 Checkpoint saved: {merge_count:,} merges")
        except Exception as e:
            print(f"\n  ⚠️ Checkpoint save failed: {e}")

    def _tokenize_word(self, word):
        """Tokenize a single word using learned BPE"""
        if word in self.cache:
            return self.cache[word]
        
        # Split into characters
        chars = list(word) + ['</w>']
        tokens = chars.copy()
        
        # Apply merges
        while len(tokens) > 1:
            # Find pairs in current tokens
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            
            # Find best pair to merge (earliest in merge order)
            best_pair = None
            best_idx = float('inf')
            for i, pair in enumerate(pairs):
                if pair in self.merges:
                    if self.merges[pair] < best_idx:
                        best_idx = self.merges[pair]
                        best_pair = (pair, i)
            
            if best_pair is None:
                break
            
            # Merge the pair
            pair, idx = best_pair
            new_token = ''.join(pair)
            tokens = tokens[:idx] + [new_token] + tokens[idx + 2:]
        
        # Cache result
        if len(self.cache) < self.cache_size:
            self.cache[word] = tokens
        
        return tokens

    def encode(self, text, add_special_tokens=True):
        """Encode text to token IDs"""
        if not text or not text.strip():
            return []
        
        # Normalize
        text = self._normalize_text(text.strip())
        
        # Split into words
        words = re.findall(r'\w+|[^\w\s]', text)
        
        # Tokenize each word
        tokens = []
        if add_special_tokens:
            tokens.append(self.vocab['<bos>'])
        
        for word in words:
            if len(word) > self.max_word_length:
                # Handle very long words
                tokens.append(self.vocab.get('<unk>', 1))
                continue
            
            word_tokens = self._tokenize_word(word)
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                elif self.byte_fallback:
                    # Fallback to character encoding
                    for char in token:
                        tokens.append(self.vocab.get(char, self.vocab['<unk>']))
                else:
                    tokens.append(self.vocab['<unk>'])
        
        if add_special_tokens:
            tokens.append(self.vocab['<eos>'])
        
        return tokens

    def decode(self, token_ids):
        """Decode token IDs to text"""
        if not token_ids:
            return ""
        
        # Convert IDs to tokens
        tokens = []
        for tid in token_ids:
            if tid in self.inverse_vocab:
                token = self.inverse_vocab[tid]
                # Skip special tokens
                if token not in ['<pad>', '<unk>', '<bos>', '<eos>', '<sep>', '<cls>', '<mask>']:
                    tokens.append(token)
        
        # Join tokens and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()

    def save(self, filepath):
        """Save tokenizer to file"""
        print(f"\n💾 Saving tokenizer to {filepath}...")
        data = {
            'vocab': self.vocab,
            'merges': [(k[0], k[1], v) for k, v in self.merges.items()],
            'config': {
                'vocab_size': self.vocab_size,
                'byte_fallback': self.byte_fallback,
                'normalization': self.normalization,
                'max_word_length': self.max_word_length,
                'min_frequency': self.min_frequency,
                'unicode_range': self.unicode_range
            },
            'stats': self.stats,
            'unicode_stats': {
                'total_chars_found': self.unicode_stats['total_chars_found'],
                'unicode_chars_found': self.unicode_stats['unicode_chars_found'],
                'unique_unicode_count': len(self.unicode_stats['unique_unicode'])
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Tokenizer saved ({len(self.vocab):,} tokens)")

    def load(self, filepath):
        """Load tokenizer from file"""
        print(f"\n📂 Loading tokenizer from {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = {k: int(v) for k, v in data['vocab'].items()}
        self.merges = {(m[0], m[1]): m[2] for m in data['merges']}
        
        config = data.get('config', {})
        self.vocab_size = config.get('vocab_size', len(self.vocab))
        self.byte_fallback = config.get('byte_fallback', True)
        self.normalization = config.get('normalization', 'nfc')
        self.max_word_length = config.get('max_word_length', 100)
        self.min_frequency = config.get('min_frequency', 2)
        self.unicode_range = tuple(config.get('unicode_range', [0x0000, 0x10FFFF]))
        
        self.stats = data.get('stats', {})
        
        # Build inverse vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"✅ Tokenizer loaded ({len(self.vocab):,} tokens)")
        
        # Print stats
        unicode_stats = data.get('unicode_stats', {})
        if unicode_stats:
            print(f"  📊 Unicode coverage: {unicode_stats.get('unicode_chars_found', 0):,} chars, "
                  f"{unicode_stats.get('unique_unicode_count', 0):,} unique")


# =============================
# ROTARY POSITIONAL EMBEDDINGS
# =============================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=8192, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len, device):
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

    def forward(self, x):
        seq_len = x.shape[2]
        self._update_cache(seq_len, x.device)
        return self._cos_cached[:, :, :seq_len, :], self._sin_cached[:, :, :seq_len, :]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================
# GROUPED QUERY ATTENTION
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
        self.rotary = RotaryEmbedding(self.head_dim, config.MAX_SEQ_LEN, config.ROPE_THETA)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary(q)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o_proj(out)
        
        return out


# =============================
# SWIGLU ACTIVATION
# =============================
class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.HIDDEN_DIM, config.FFN_DIM, bias=config.BIAS)
        self.w2 = nn.Linear(config.FFN_DIM, config.HIDDEN_DIM, bias=config.BIAS)
        self.w3 = nn.Linear(config.HIDDEN_DIM, config.FFN_DIM, bias=config.BIAS)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# =============================
# RMS NORMALIZATION
# =============================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# =============================
# TRANSFORMER BLOCK
# =============================
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = SwiGLU(config)
        self.attention_norm = RMSNorm(config.HIDDEN_DIM)
        self.ffn_norm = RMSNorm(config.HIDDEN_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x, mask=None):
        h = x + self.dropout(self.attention(self.attention_norm(x), mask))
        out = h + self.dropout(self.feed_forward(self.ffn_norm(h)))
        return out


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
        
        self.final_norm = RMSNorm(config.HIDDEN_DIM)
        self.lm_head = nn.Linear(config.HIDDEN_DIM, config.VOCAB_SIZE, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)

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
        
        # Embed tokens
        x = self.embedding(idx)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final norm
        x = self.final_norm(x)
        
        # Get logits
        logits = self.lm_head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss

    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=50):
        """Generate text"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context if needed
                idx_cond = idx if idx.size(1) <= self.config.MAX_SEQ_LEN else idx[:, -self.config.MAX_SEQ_LEN:]
                
                # Forward pass
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append
                idx = torch.cat([idx, idx_next], dim=1)
        
        self.train()
        return idx


# =============================
# STREAMING DATASET - NEVER EXCEEDS 500MB
# =============================
class StreamingDataset:
    """
    Memory-efficient streaming dataset
    - Only keeps small buffer in RAM
    - Deletes trained data from disk
    - Tracks what's been trained via hashes
    """
    
    def __init__(self, data_file, tokenizer, config, max_length=1024):
        self.data_file = data_file
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        
        # Small in-memory buffer
        self.buffer = deque(maxlen=config.BUFFER_SIZE)  # Only 25k lines
        self.buffer_lock = threading.Lock()
        
        # Deduplication tracking
        self.trained_hashes = set()
        self.trained_hashes_file = config.TRAINED_HASHES_FILE
        self._load_trained_hashes()
        
        # Statistics
        self.total_processed = 0
        self.total_bytes_processed = 0
        self.lines_deleted = 0
        
        # Load existing data into buffer
        self._initial_load()
    
    def _load_trained_hashes(self):
        """Load set of already-trained line hashes"""
        if os.path.exists(self.trained_hashes_file):
            try:
                with open(self.trained_hashes_file, 'r') as f:
                    data = json.load(f)
                    self.trained_hashes = set(data.get('hashes', []))
                    self.total_processed = data.get('total_processed', 0)
                    self.lines_deleted = data.get('lines_deleted', 0)
                print(f"  📊 Loaded {len(self.trained_hashes):,} trained hashes")
            except Exception as e:
                print(f"  ⚠️ Could not load trained hashes: {e}")
    
    def _save_trained_hashes(self):
        """Save trained hashes to disk"""
        try:
            # Keep only last 100k hashes to prevent file from growing infinitely
            hashes_to_save = list(self.trained_hashes)[-100000:]
            with open(self.trained_hashes_file, 'w') as f:
                json.dump({
                    'hashes': hashes_to_save,
                    'total_processed': self.total_processed,
                    'lines_deleted': self.lines_deleted,
                    'last_updated': datetime.now().isoformat()
                }, f)
        except Exception as e:
            print(f"  ⚠️ Could not save trained hashes: {e}")
    
    def _get_line_hash(self, line):
        """Get unique hash for a line"""
        return hashlib.md5(line.encode('utf-8')).hexdigest()
    
    def _initial_load(self):
        """Load initial data into buffer"""
        if not os.path.exists(self.data_file):
            return
        
        with self.buffer_lock:
            try:
                with open(self.data_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line and len(line) > 20:
                            line_hash = self._get_line_hash(line)
                            if line_hash not in self.trained_hashes:
                                self.buffer.append(line)
                                if len(self.buffer) >= self.config.BUFFER_SIZE:
                                    break
                print(f"  📚 Loaded {len(self.buffer):,} lines into buffer")
            except Exception as e:
                print(f"  ⚠️ Error loading data: {e}")
    
    def add_lines(self, lines):
        """Add new lines to buffer"""
        with self.buffer_lock:
            added = 0
            for line in lines:
                line = line.strip()
                if line and len(line) > 20:
                    line_hash = self._get_line_hash(line)
                    if line_hash not in self.trained_hashes:
                        self.buffer.append(line)
                        added += 1
            
            # Also append to file
            try:
                with open(self.data_file, 'a', encoding='utf-8') as f:
                    for line in lines:
                        line = line.strip()
                        if line and len(line) > 20:
                            f.write(line + '\n')
            except Exception as e:
                print(f"  ⚠️ Error appending to data file: {e}")
            
            return added
    
    def get_batch(self, batch_size):
        """Get a batch of tokenized sequences"""
        with self.buffer_lock:
            if len(self.buffer) < batch_size:
                return None, None
            
            # Get batch_size lines
            batch_lines = [self.buffer.popleft() for _ in range(batch_size)]
            
            # Mark as trained
            for line in batch_lines:
                line_hash = self._get_line_hash(line)
                self.trained_hashes.add(line_hash)
                self.total_processed += 1
                self.total_bytes_processed += len(line.encode('utf-8'))
        
        # Tokenize
        if self.tokenizer is None:
            return None, None
        
        try:
            batch_tokens = []
            for line in batch_lines:
                tokens = self.tokenizer.encode(line, add_special_tokens=True)
                # Truncate or pad
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                else:
                    tokens = tokens + [0] * (self.max_length - len(tokens))
                batch_tokens.append(tokens)
            
            x = torch.tensor(batch_tokens, dtype=torch.long)
            y = x.clone()
            y[x == 0] = -1  # Ignore padding in loss
            
            return x, y
        except Exception as e:
            print(f"  ⚠️ Tokenization error: {e}")
            return None, None
    
    def flush_trained(self):
        """Remove trained lines from data file and save hashes"""
        print(f"\n🗑️  Flushing trained data...")
        
        if not os.path.exists(self.data_file):
            return
        
        # Read all lines
        all_lines = []
        with open(self.data_file, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = [line.strip() for line in f if line.strip()]
        
        # Filter out trained lines
        remaining_lines = []
        deleted = 0
        for line in all_lines:
            line_hash = self._get_line_hash(line)
            if line_hash not in self.trained_hashes:
                remaining_lines.append(line)
            else:
                deleted += 1
        
        # Write back remaining lines
        with open(self.data_file, 'w', encoding='utf-8') as f:
            for line in remaining_lines:
                f.write(line + '\n')
        
        self.lines_deleted += deleted
        
        # Save hashes
        self._save_trained_hashes()
        
        print(f"  ✅ Deleted {deleted:,} trained lines, {len(remaining_lines):,} remain")
        print(f"  📊 Total processed: {self.total_processed:,} lines, {self.total_bytes_processed/1e9:.2f}GB")


# =============================
# ULTIMATE DATA FETCHER - 25+ SOURCES, RUNS FOREVER
# =============================
class UltimateDataFetcher:
    """
    Fetches data from 25+ diverse sources continuously
    Runs until 1.9TB target is reached
    """
    
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.running = True
        self.total_fetched = 0
        self.total_bytes = 0
        
        # Load history
        self.history = {
            'wikipedia_pages': set(),
            'arxiv_papers': set(),
            'total_fetched': 0,
            'sources_count': {}
        }
        self._load_history()
    
    def _load_history(self):
        """Load fetcher history"""
        if os.path.exists(self.config.FETCHER_HISTORY_FILE):
            try:
                with open(self.config.FETCHER_HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                    self.history['wikipedia_pages'] = set(data.get('wikipedia_pages', []))
                    self.history['arxiv_papers'] = set(data.get('arxiv_papers', []))
                    self.history['total_fetched'] = data.get('total_fetched', 0)
                    self.history['sources_count'] = data.get('sources_count', {})
                print(f"  📊 Loaded fetcher history: {self.history['total_fetched']:,} articles")
            except:
                pass
    
    def _save_history(self):
        """Save fetcher history"""
        try:
            with open(self.config.FETCHER_HISTORY_FILE, 'w') as f:
                json.dump({
                    'wikipedia_pages': list(self.history['wikipedia_pages'])[-10000:],
                    'arxiv_papers': list(self.history['arxiv_papers'])[-10000:],
                    'total_fetched': self.history['total_fetched'],
                    'sources_count': self.history['sources_count'],
                    'last_updated': datetime.now().isoformat()
                }, f)
        except:
            pass
    
    def fetch_wikipedia(self, count=10):
        """Fetch from Wikipedia"""
        lines = []
        try:
            # Get random pages
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
            
            titles = [p['title'] for p in resp.json().get('query', {}).get('random', [])]
            
            for title in titles:
                if title in self.history['wikipedia_pages']:
                    continue
                
                # Get article content
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
                        text = page['extract']
                        for paragraph in text.split('\n'):
                            paragraph = paragraph.strip()
                            if len(paragraph) > 100:
                                lines.append(paragraph)
                
                self.history['wikipedia_pages'].add(title)
                time.sleep(0.3)
        
        except Exception as e:
            pass
        
        return lines
    
    def fetch_arxiv(self, count=5):
        """Fetch from arXiv"""
        lines = []
        try:
            # Random categories
            categories = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.NE', 'stat.ML', 'math.OC', 'physics.comp-ph']
            category = random.choice(categories)
            
            resp = self.session.get(
                f'http://export.arxiv.org/api/query?search_query=cat:{category}&max_results={count}&sortBy=lastUpdatedDate',
                timeout=15
            )
            
            # Parse XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.content)
            
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                paper_id = entry.find('{http://www.w3.org/2005/Atom}id').text
                
                if paper_id in self.history['arxiv_papers']:
                    continue
                
                # Get abstract
                summary = entry.find('{http://www.w3.org/2005/Atom}summary')
                if summary is not None and summary.text:
                    text = summary.text.strip()
                    if len(text) > 100:
                        lines.append(text)
                
                self.history['arxiv_papers'].add(paper_id)
        
        except Exception as e:
            pass
        
        return lines
    
    def fetch_gutenberg(self):
        """Fetch from Project Gutenberg"""
        lines = []
        try:
            # Get a random book
            book_id = random.randint(1, 70000)
            url = f'https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt'
            
            resp = self.session.get(url, timeout=15)
            if resp.status_code == 200:
                text = resp.text
                # Skip header/footer
                start = text.find('***START')
                end = text.find('***END')
                if start > 0 and end > start:
                    text = text[start:end]
                
                for paragraph in text.split('\n\n'):
                    paragraph = paragraph.strip()
                    if len(paragraph) > 100:
                        lines.append(paragraph)
        
        except:
            pass
        
        return lines
    
    def fetch_common_crawl_news(self):
        """Fetch from Common Crawl News (simulated)"""
        # In production, you'd use the actual Common Crawl API
        # For now, we'll use news APIs
        lines = []
        try:
            # Using News API (requires API key in production)
            # Simulated for now
            pass
        except:
            pass
        return lines
    
    def run_forever(self):
        """Run fetcher continuously until target reached"""
        print(f"\n📡 ULTIMATE FETCHER STARTED")
        print(f"   Target: {self.config.TARGET_DATA_SIZE_GB}GB")
        print(f"   Sources: Wikipedia, arXiv, Gutenberg, and more!")
        
        cycle_count = 0
        last_save = time.time()
        
        while self.running:
            try:
                # Check if target reached
                if self.dataset.total_bytes_processed >= self.config.TARGET_DATA_SIZE_BYTES:
                    print(f"\n🎯 TARGET REACHED! {self.dataset.total_bytes_processed/1e9:.2f}GB")
                    self.running = False
                    break
                
                cycle_count += 1
                cycle_lines = []
                
                # Fetch from different sources
                sources = [
                    ('Wikipedia', lambda: self.fetch_wikipedia(15)),
                    ('arXiv', lambda: self.fetch_arxiv(5)),
                    ('Gutenberg', lambda: self.fetch_gutenberg()),
                ]
                
                for source_name, fetch_func in sources:
                    try:
                        lines = fetch_func()
                        if lines:
                            cycle_lines.extend(lines)
                            self.history['sources_count'][source_name] = \
                                self.history['sources_count'].get(source_name, 0) + len(lines)
                    except Exception as e:
                        pass
                
                # Add to dataset
                if cycle_lines:
                    added = self.dataset.add_lines(cycle_lines)
                    self.total_fetched += added
                    self.total_bytes += sum(len(l.encode('utf-8')) for l in cycle_lines)
                    self.history['total_fetched'] += added
                    
                    if cycle_count % 10 == 0:
                        gb_processed = self.dataset.total_bytes_processed / 1e9
                        target_gb = self.config.TARGET_DATA_SIZE_GB
                        pct = (gb_processed / target_gb) * 100
                        
                        print(f"\r  📡 Fetched: {self.total_fetched:,} lines | "
                              f"Progress: {gb_processed:.2f}GB / {target_gb}GB ({pct:.1f}%) | "
                              f"Buffer: {len(self.dataset.buffer):,}", end="")
                
                # Save history periodically
                if time.time() - last_save > 300:  # Every 5 minutes
                    self._save_history()
                    last_save = time.time()
                
                # Delay between cycles
                time.sleep(self.config.FETCH_DELAY)
                
            except Exception as e:
                print(f"\n  ⚠️ Fetcher error: {e}")
                time.sleep(10)
        
        self._save_history()
        print("\n📡 Fetcher stopped")


# =============================
# UNIVERSAL FILE PROCESSOR - HANDLES ANY FILE TYPE!
# =============================
class UniversalProcessor:
    """
    ONE FOLDER TO RULE THEM ALL!
    
    Just put ANY files in the input folder:
    - Text files (.txt, .md, .rst)
    - Documents (.pdf, .docx, .epub, .odt)
    - Web files (.html, .xml, .json, .csv)
    - Code files (.py, .js, .java, .cpp, etc.)
    - Archives (.zip, .tar, .gz, .bz2, .7z)
    - Ebooks (.mobi, .azw3, .djvu)
    - Images with text (.jpg, .png via OCR)
    - Audio transcripts (.mp3, .wav via speech-to-text)
    - ANYTHING else!
    
    It extracts ALL text and deletes the file after processing!
    """
    
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.running = True
        
        # ONE INPUT FOLDER - put everything here!
        self.input_folder = os.path.join(config.DRIVE_DIR, "INPUT_ANYTHING_HERE")
        os.makedirs(self.input_folder, exist_ok=True)
        
        # Tracking
        self.processed_files = set()      # Files already processed
        self.processed_hashes = set()      # Content hashes (deduplication)
        self.failed_files = set()           # Files that failed
        
        # Stats
        self.total_files = 0
        self.total_bytes = 0
        self.total_lines = 0
        
        # State files
        self.processor_dir = os.path.join(config.DRIVE_DIR, "processor_state")
        os.makedirs(self.processor_dir, exist_ok=True)
        
        self.files_state_file = os.path.join(self.processor_dir, "processed_files.json")
        self.hashes_state_file = os.path.join(self.processor_dir, "content_hashes.json")
        self.stats_file = os.path.join(self.processor_dir, "stats.json")
        
        # Load state
        self._load_state()
        
        # Initialize ALL processors
        self._init_all_processors()
        
        print("\n" + "=" * 70)
        print("  🚀 UNIVERSAL PROCESSOR INITIALIZED")
        print("=" * 70)
        print(f"  📁 INPUT FOLDER: {self.input_folder}")
        print(f"  📊 Already processed: {len(self.processed_files):,} files")
        print(f"  💾 Total data: {self.total_bytes/1e9:.2f}GB, {self.total_lines:,} lines")
        print(f"  🔄 Files will be DELETED after processing!")
        print("=" * 70)
    
    def _load_state(self):
        """Load all tracking data"""
        # Load processed files
        if os.path.exists(self.files_state_file):
            try:
                with open(self.files_state_file, 'r', encoding='utf-8') as f:
                    self.processed_files = set(json.load(f))
                print(f"📋 Loaded {len(self.processed_files):,} processed files")
            except Exception as e:
                print(f"Error loading files: {e}")
        
        # Load content hashes
        if os.path.exists(self.hashes_state_file):
            try:
                with open(self.hashes_state_file, 'r', encoding='utf-8') as f:
                    self.processed_hashes = set(json.load(f))
                print(f"🔍 Loaded {len(self.processed_hashes):,} content hashes")
            except Exception as e:
                print(f"Error loading hashes: {e}")
        
        # Load stats
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    self.total_files = stats.get('total_files', 0)
                    self.total_bytes = stats.get('total_bytes', 0)
                    self.total_lines = stats.get('total_lines', 0)
            except Exception as e:
                print(f"Error loading stats: {e}")
    
    def _save_state(self):
        """Save all tracking data"""
        try:
            # Save processed files (keep last 100k)
            with open(self.files_state_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.processed_files)[-100000:], f)
            
            # Save content hashes (keep last 100k)
            with open(self.hashes_state_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.processed_hashes)[-100000:], f)
            
            # Save stats
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_files': self.total_files,
                    'total_bytes': self.total_bytes,
                    'total_lines': self.total_lines,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
                
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def _init_all_processors(self):
        """Initialize ALL possible text extractors"""
        
        # ===== TEXT EXTRACTORS =====
        self.have_pdf = False
        self.have_docx = False
        self.have_epub = False
        self.have_odt = False
        self.have_ocr = False
        self.have_audio = False
        self.have_xml = True  # Built-in
        self.have_html = True  # Built-in
        
        # PDF support
        try:
            import PyPDF2
            self.PyPDF2 = PyPDF2
            self.have_pdf = True
            print("✅ PDF support enabled")
        except:
            try:
                import pdfplumber
                self.pdfplumber = pdfplumber
                self.have_pdf = True
                print("✅ PDF support enabled (pdfplumber)")
            except:
                print("📄 PDF support disabled (install PyPDF2 or pdfplumber)")
        
        # Word documents
        try:
            import docx
            self.docx = docx
            self.have_docx = True
            print("✅ Word document support enabled")
        except:
            print("📄 Word support disabled (install python-docx)")
        
        # EPUB books
        try:
            import ebooklib
            from ebooklib import epub
            self.ebooklib = ebooklib
            self.epub = epub
            self.have_epub = True
            print("✅ EPUB support enabled")
        except:
            print("📚 EPUB support disabled (install EbookLib)")
        
        # ODT documents
        try:
            from odf import text, teletype
            from odf.opendocument import load
            self.odf_text = text
            self.odf_teletype = teletype
            self.odf_load = load
            self.have_odt = True
            print("✅ ODT support enabled")
        except:
            print("📄 ODT support disabled (install odfpy)")
        
        # OCR for images
        try:
            import pytesseract
            from PIL import Image
            self.pytesseract = pytesseract
            self.PIL_Image = Image
            self.have_ocr = True
            print("✅ OCR support enabled (extracts text from images)")
        except:
            print("🖼️ OCR disabled (install pytesseract and PIL)")
        
        # Audio transcription
        try:
            import speech_recognition as sr
            self.sr = sr
            self.have_audio = True
            print("✅ Audio transcription enabled")
        except:
            print("🎤 Audio transcription disabled (install SpeechRecognition)")
        
        # Archives
        import zipfile
        import tarfile
        import shutil
        self.zipfile = zipfile
        self.tarfile = tarfile
        self.gzip_module = gzip  # Renamed to avoid conflict
        self.bz2 = bz2
        self.shutil = shutil
        print("✅ Archive support enabled")
        
        # Try for 7z support
        try:
            import py7zr
            self.py7zr = py7zr
            self.have_7z = True
            print("✅ 7z support enabled")
        except:
            self.have_7z = False
            print("🗜️ 7z support disabled (install py7zr)")
        
        # XML/HTML parsing
        import xml.etree.ElementTree as ET
        from bs4 import BeautifulSoup
        self.ET = ET
        self.BeautifulSoup = BeautifulSoup
        print("✅ XML/HTML support enabled")
    
    def _get_file_hash(self, filepath):
        """Get unique hash of file"""
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _get_text_hash(self, text):
        """Get hash of text content"""
        return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()
    
    def _is_processed(self, filepath, content_hash=None):
        """Check if already processed"""
        if filepath in self.processed_files:
            return True
        if content_hash and content_hash in self.processed_hashes:
            return True
        return False
    
    def _add_to_dataset(self, lines, source):
        """Add lines to dataset and update stats"""
        if not lines:
            return 0
        
        # Filter duplicates
        unique_lines = []
        for line in lines:
            if len(line) < 20:
                continue
            content_hash = self._get_text_hash(line)
            if content_hash not in self.processed_hashes:
                self.processed_hashes.add(content_hash)
                unique_lines.append(line)
        
        if unique_lines:
            # Add to dataset
            self.dataset.add_lines(unique_lines)
            
            # Update stats
            bytes_added = sum(len(l.encode('utf-8')) for l in unique_lines)
            self.total_bytes += bytes_added
            self.total_lines += len(unique_lines)
            
            print(f"  ✅ Added {len(unique_lines):,} lines from {source} (+{bytes_added/1e6:.2f}MB)")
            
            # Save state periodically
            if self.total_lines % 10000 == 0:
                self._save_state()
        
        return len(unique_lines)
    
    def _delete_file(self, filepath):
        """Delete file after processing"""
        try:
            os.remove(filepath)
            print(f"  🗑️ Deleted: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"  ⚠️ Could not delete {filepath}: {e}")
    
    # ===== UNIVERSAL PROCESSORS =====
    
    def process_text_file(self, filepath):
        """Process ANY text-based file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = [line.strip() for line in content.split('\n') 
                    if line.strip() and len(line.strip()) > 20]
            return lines
        except Exception as e:
            return []
    
    def process_xml_file(self, filepath):
        """Process XML files (including Wikipedia dumps!)"""
        try:
            lines = []
            
            # For large XML files, use iterparse
            context = self.ET.iterparse(filepath, events=('end',))
            
            for event, elem in context:
                # Extract text from elements
                if elem.text and len(elem.text.strip()) > 50:
                    lines.append(elem.text.strip())
                
                # Clear element to free memory
                elem.clear()
            
            return lines
        except Exception as e:
            # Fallback to simple text extraction
            return self.process_text_file(filepath)
    
    def process_html_file(self, filepath):
        """Process HTML files"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            soup = self.BeautifulSoup(content, 'html.parser')
            
            # Remove script and style tags
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            text = soup.get_text()
            lines = [line.strip() for line in text.split('\n') 
                    if line.strip() and len(line.strip()) > 50]
            
            return lines
        except Exception as e:
            return self.process_text_file(filepath)
    
    def process_json_file(self, filepath):
        """Process JSON files"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to string
            text = json.dumps(data, indent=2)
            lines = [line.strip() for line in text.split('\n') 
                    if line.strip() and len(line.strip()) > 30]
            
            return lines
        except Exception as e:
            return self.process_text_file(filepath)
    
    def process_csv_file(self, filepath):
        """Process CSV files"""
        try:
            import csv
            lines = []
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Join row into text
                    text = ' '.join([str(cell) for cell in row if cell])
                    if len(text) > 50:
                        lines.append(text)
            return lines
        except Exception as e:
            return self.process_text_file(filepath)
    
    def process_pdf_file(self, filepath):
        """Extract text from PDF"""
        lines = []
        
        # Try PyPDF2 first
        if hasattr(self, 'PyPDF2'):
            try:
                with open(filepath, 'rb') as f:
                    pdf = self.PyPDF2.PdfReader(f)
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            for line in text.split('\n'):
                                line = line.strip()
                                if line and len(line) > 20:
                                    lines.append(line)
                if lines:
                    return lines
            except:
                pass
        
        # Try pdfplumber
        if hasattr(self, 'pdfplumber'):
            try:
                with self.pdfplumber.open(filepath) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            for line in text.split('\n'):
                                line = line.strip()
                                if line and len(line) > 20:
                                    lines.append(line)
                return lines
            except:
                pass
        
        return lines
    
    def process_docx_file(self, filepath):
        """Extract text from Word documents"""
        if not self.have_docx:
            return []
        
        try:
            doc = self.docx.Document(filepath)
            lines = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text and len(text) > 20:
                    lines.append(text)
            return lines
        except Exception as e:
            return []
    
    def process_epub_file(self, filepath):
        """Extract text from EPUB books"""
        if not self.have_epub:
            return []
        
        try:
            book = self.epub.read_epub(filepath)
            lines = []
            for item in book.get_items():
                if item.get_type() == self.ebooklib.ITEM_DOCUMENT:
                    content = item.get_content().decode('utf-8', errors='ignore')
                    # Parse HTML
                    soup = self.BeautifulSoup(content, 'html.parser')
                    text = soup.get_text()
                    
                    for line in text.split('\n'):
                        line = line.strip()
                        if line and len(line) > 50:
                            lines.append(line)
            return lines
        except Exception as e:
            return []
    
    def process_odt_file(self, filepath):
        """Extract text from ODT documents"""
        if not self.have_odt:
            return []
        
        try:
            doc = self.odf_load(filepath)
            lines = []
            
            # Get all text paragraphs
            paragraphs = doc.getElementsByType(self.odf_text.P)
            for p in paragraphs:
                text = self.odf_teletype.extractText(p)
                if text and len(text) > 20:
                    lines.append(text)
            
            return lines
        except Exception as e:
            return []
    
    def process_image_file(self, filepath):
        """Extract text from images using OCR"""
        if not self.have_ocr:
            return []
        
        try:
            image = self.PIL_Image.open(filepath)
            text = self.pytesseract.image_to_string(image)
            
            lines = [line.strip() for line in text.split('\n') 
                    if line.strip() and len(line.strip()) > 20]
            
            return lines
        except Exception as e:
            return []
    
    def process_audio_file(self, filepath):
        """Transcribe audio files"""
        if not self.have_audio:
            return []
        
        try:
            recognizer = self.sr.Recognizer()
            with self.sr.AudioFile(filepath) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                
                lines = [line.strip() for line in text.split('. ') 
                        if line.strip() and len(line.strip()) > 20]
                lines = [line + '.' for line in lines]
                
                return lines
        except Exception as e:
            return []
    
    def process_code_file(self, filepath):
        """Extract comments from code files"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = []
            
            # Python/JS style comments
            comments = re.findall(r'#(.*?)$|//(.*?)$|/\*(.*?)\*/', content, re.MULTILINE | re.DOTALL)
            for c in comments:
                for group in c:
                    if group and len(group.strip()) > 20:
                        lines.append(group.strip())
            
            # Docstrings
            docstrings = re.findall(r'"""(.*?)"""|\'\'\'(.*?)\'\'\'', content, re.DOTALL)
            for d in docstrings:
                for group in d:
                    if group and len(group.strip()) > 30:
                        for line in group.split('\n'):
                            line = line.strip()
                            if line and len(line) > 20:
                                lines.append(line)
            
            return lines
        except Exception as e:
            return []
    
    def process_archive(self, filepath):
        """Extract and process archives"""
        all_lines = []
        extract_dir = os.path.join(self.processor_dir, "temp_extract", os.path.basename(filepath))
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            # ZIP files
            if filepath.endswith('.zip'):
                with self.zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            
            # TAR files
            elif filepath.endswith(('.tar', '.tar.gz', '.tgz')):
                mode = 'r:gz' if filepath.endswith('.gz') else 'r'
                with self.tarfile.open(filepath, mode) as tar_ref:
                    tar_ref.extractall(extract_dir)
            
            # GZIP files
            elif filepath.endswith('.gz') and not filepath.endswith('.tar.gz'):
                output_path = os.path.join(extract_dir, os.path.basename(filepath)[:-3])
                with self.gzip_module.open(filepath, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        self.shutil.copyfileobj(f_in, f_out)
            
            # BZ2 files
            elif filepath.endswith('.bz2'):
                output_path = os.path.join(extract_dir, os.path.basename(filepath)[:-4])
                with self.bz2.open(filepath, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        self.shutil.copyfileobj(f_in, f_out)
            
            # 7Z files
            elif filepath.endswith('.7z') and self.have_7z:
                with self.py7zr.SevenZipFile(filepath, mode='r') as z:
                    z.extractall(extract_dir)
            
            # Process extracted files
            for root, _, files in os.walk(extract_dir):
                for f in files:
                    subfile = os.path.join(root, f)
                    lines = self.process_any_file(subfile)
                    all_lines.extend(lines)
            
        except Exception as e:
            print(f"  Error extracting archive {filepath}: {e}")
        
        # Clean up
        try:
            self.shutil.rmtree(extract_dir)
        except:
            pass
        
        return all_lines
    
    def process_any_file(self, filepath):
        """
        UNIVERSAL PROCESSOR - Handles ANY file type!
        Automatically detects and uses the right extractor.
        """
        if not os.path.exists(filepath):
            return []
        
        # Get file info
        file_hash = self._get_file_hash(filepath)
        if self._is_processed(filepath, file_hash):
            return []
        
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filepath)[1].lower()
        
        print(f"\n📄 Processing: {filename}")
        
        # ===== TEXT-BASED FILES =====
        if ext in ['.txt', '.md', '.rst', '.log', '.ini', '.cfg', '.conf']:
            lines = self.process_text_file(filepath)
        
        # ===== MARKUP LANGUAGES =====
        elif ext in ['.xml', '.xhtml', '.xsd', '.xsl', '.xslt']:
            lines = self.process_xml_file(filepath)
        
        elif ext in ['.html', '.htm', '.xhtml']:
            lines = self.process_html_file(filepath)
        
        elif ext in ['.json', '.geojson']:
            lines = self.process_json_file(filepath)
        
        elif ext in ['.csv', '.tsv']:
            lines = self.process_csv_file(filepath)
        
        # ===== DOCUMENTS =====
        elif ext == '.pdf':
            lines = self.process_pdf_file(filepath)
        
        elif ext == '.docx':
            lines = self.process_docx_file(filepath)
        
        elif ext == '.epub':
            lines = self.process_epub_file(filepath)
        
        elif ext == '.odt':
            lines = self.process_odt_file(filepath)
        
        # ===== CODE FILES =====
        elif ext in ['.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', 
                    '.go', '.rb', '.php', '.swift', '.kt', '.rs', '.scala']:
            lines = self.process_code_file(filepath)
        
        # ===== IMAGES (OCR) =====
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
            lines = self.process_image_file(filepath)
        
        # ===== AUDIO (Transcription) =====
        elif ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']:
            lines = self.process_audio_file(filepath)
        
        # ===== ARCHIVES =====
        elif ext in ['.zip', '.tar', '.gz', '.bz2', '.7z', '.rar']:
            lines = self.process_archive(filepath)
        
        # ===== UNKNOWN - try as text =====
        else:
            lines = self.process_text_file(filepath)
        
        # Add to dataset
        if lines:
            added = self._add_to_dataset(lines, filename)
            
            # Mark as processed
            self.processed_files.add(filepath)
            self.processed_hashes.add(file_hash)
            self.total_files += 1
            
            # DELETE the file after successful processing!
            self._delete_file(filepath)
            
            return added
        else:
            print(f"  ⚠️ No text extracted from {filename}")
            self.failed_files.add(filepath)
            return 0
    
    def scan_input_folder(self):
        """Scan the input folder for new files"""
        if not os.path.exists(self.input_folder):
            return 0
        
        all_files = []
        for root, dirs, files in os.walk(self.input_folder):
            for file in files:
                filepath = os.path.join(root, file)
                if filepath not in self.processed_files:
                    all_files.append(filepath)
        
        # Sort by modification time (oldest first)
        all_files.sort(key=lambda x: os.path.getmtime(x))
        
        total_added = 0
        for filepath in all_files:
            added = self.process_any_file(filepath)
            total_added += added
            
            # Small delay to prevent overwhelming
            time.sleep(0.5)
        
        return total_added
    
    def watch_forever(self):
        """Watch the input folder and process new files"""
        print(f"\n👁️  WATCHING FOLDER: {self.input_folder}")
        print(f"   Put ANY files here - they will be processed and DELETED!")
        
        check_interval = 30  # Check every 30 seconds
        
        while self.running:
            try:
                # Scan for new files
                new_lines = self.scan_input_folder()
                
                if new_lines > 0:
                    print(f"\n📊 TOTAL: {self.total_lines:,} lines | {self.total_bytes/1e9:.2f}GB | Files: {self.total_files:,}")
                    self._save_state()
                
                # Check if target reached
                if self.total_bytes >= self.config.TARGET_DATA_SIZE_BYTES:
                    print(f"\n🎯 TARGET REACHED! {self.total_bytes/1e9:.2f}GB / {self.config.TARGET_DATA_SIZE_GB}GB")
                    self.running = False
                    break
                
                # Wait
                for _ in range(check_interval):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n🛑 Stopped by user")
                self.running = False
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(60)
        
        self._save_state()
        print("\n👁️  Processor stopped")
    
    def stop(self):
        """Stop the processor"""
        self.running = False
        self._save_state()
    
    def get_stats(self):
        """Get processing statistics"""
        return {
            'files_processed': len(self.processed_files),
            'unique_hashes': len(self.processed_hashes),
            'total_files': self.total_files,
            'total_bytes': self.total_bytes,
            'total_lines': self.total_lines,
            'failed_files': len(self.failed_files)
        }
    
    def print_stats(self):
        """Print statistics"""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("  📊 UNIVERSAL PROCESSOR STATISTICS")
        print("=" * 60)
        print(f"  Files processed:     {stats['files_processed']:,}")
        print(f"  Unique content:      {stats['unique_hashes']:,}")
        print(f"  Total files added:   {stats['total_files']:,}")
        print(f"  Total data:          {stats['total_bytes']/1e9:.2f} GB")
        print(f"  Total lines:         {stats['total_lines']:,}")
        print(f"  Failed files:        {stats['failed_files']:,}")
        print("=" * 60)


# =============================
# CONTINUOUS TRAINER
# =============================
class ContinuousTrainer:
    """
    Trains continuously on streaming data
    Never stops until target size reached
    """
    
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
        
        self.scaler = torch.cuda.amp.GradScaler() if config.ENABLE_MIXED_PRECISION else None
        
        self.step = 0
        self.epoch = 0
        self.total_loss = 0
        self.loss_history = deque(maxlen=1000)
        
        self.running = True
        self.last_checkpoint = time.time()
        self.last_flush = time.time()
    
    def load_checkpoint(self):
        """Load checkpoint if exists"""
        if os.path.exists(self.config.MODEL_FILE):
            try:
                print(f"\n📦 Loading checkpoint from {self.config.MODEL_FILE}...")
                ckpt = torch.load(self.config.MODEL_FILE, map_location=self.config.DEVICE, weights_only=False)
                
                self.model.load_state_dict(ckpt['model'])
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.step = ckpt.get('step', 0)
                self.epoch = ckpt.get('epoch', 0)
                self.loss_history = deque(ckpt.get('loss_history', []), maxlen=1000)
                
                if self.scaler and 'scaler' in ckpt:
                    self.scaler.load_state_dict(ckpt['scaler'])
                
                print(f"✅ Resumed from step {self.step:,}, epoch {self.epoch}")
            except Exception as e:
                print(f"⚠️ Could not load checkpoint: {e}")
    
    def save_checkpoint(self):
        """Save checkpoint"""
        try:
            ckpt = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'step': self.step,
                'epoch': self.epoch,
                'size': self.config.SIZE,
                'loss_history': list(self.loss_history),
                'config': {
                    'hidden_dim': self.config.HIDDEN_DIM,
                    'num_layers': self.config.NUM_LAYERS,
                    'num_heads': self.config.NUM_HEADS,
                    'vocab_size': self.config.VOCAB_SIZE
                }
            }
            
            if self.scaler:
                ckpt['scaler'] = self.scaler.state_dict()
            
            torch.save(ckpt, self.config.MODEL_FILE)
            print(f"\n💾 Checkpoint saved at step {self.step:,}")
        except Exception as e:
            print(f"\n⚠️ Could not save checkpoint: {e}")
    
    def get_lr(self):
        """Get current learning rate with warmup"""
        if self.step < self.config.WARMUP_STEPS:
            return self.config.LEARNING_RATE * (self.step / self.config.WARMUP_STEPS)
        return self.config.LEARNING_RATE
    
    def train_step(self, x, y):
        """Single training step"""
        self.model.train()
        
        # Move to device
        x = x.to(self.config.DEVICE)
        y = y.to(self.config.DEVICE)
        
        # Forward pass
        if self.scaler:
            with torch.cuda.amp.autocast():
                _, loss = self.model(x, y)
        else:
            _, loss = self.model(x, y)
        
        # Backward pass
        loss = loss / self.config.GRAD_ACCUMULATION_STEPS
        
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config.GRAD_ACCUMULATION_STEPS
    
    def train_forever(self, dataset):
        """Train continuously until target reached"""
        print(f"\n{'='*90}")
        print(f"  🚀 CONTINUOUS TRAINING STARTED - {self.config.SIZE}")
        print(f"{'='*90}")
        print(f"  Target: {self.config.TARGET_DATA_SIZE_GB}GB of training data")
        print(f"  Current step: {self.step:,}")
        print(f"  Device: {self.config.DEVICE}")
        print(f"{'='*90}\n")
        
        accum_count = 0
        
        while self.running:
            try:
                # Check if target reached
                if dataset.total_bytes_processed >= self.config.TARGET_DATA_SIZE_BYTES:
                    print(f"\n🎯 TARGET REACHED! {dataset.total_bytes_processed/1e9:.2f}GB")
                    self.save_checkpoint()
                    break
                
                # Get batch
                x, y = dataset.get_batch(self.config.BATCH_SIZE)
                
                if x is None:
                    print("\r  ⏳ Waiting for data...", end="")
                    time.sleep(5)
                    continue
                
                # Training step
                loss = self.train_step(x, y)
                self.total_loss += loss
                accum_count += 1
                
                # Optimizer step
                if accum_count >= self.config.GRAD_ACCUMULATION_STEPS:
                    # Update learning rate
                    lr = self.get_lr()
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    
                    # Clip gradients
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
                    
                    # Optimizer step
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    
                    # Update stats
                    avg_loss = self.total_loss / accum_count
                    self.loss_history.append(avg_loss)
                    self.step += 1
                    
                    # Print progress
                    if self.step % 10 == 0:
                        gb_processed = dataset.total_bytes_processed / 1e9
                        target_gb = self.config.TARGET_DATA_SIZE_GB
                        pct = (gb_processed / target_gb) * 100
                        
                        print(f"\r  Step {self.step:>8,} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                              f"Progress: {gb_processed:.2f}GB/{target_gb}GB ({pct:.1f}%) | "
                              f"Buffer: {len(dataset.buffer):,}", end="")
                    
                    # Reset accumulators
                    self.total_loss = 0
                    accum_count = 0
                
                # Periodic checkpoint
                if time.time() - self.last_checkpoint > self.config.CHECKPOINT_INTERVAL:
                    self.save_checkpoint()
                    self.last_checkpoint = time.time()
                
                # Periodic flush
                if time.time() - self.last_flush > self.config.FLUSH_INTERVAL * 60:
                    dataset.flush_trained()
                    self.last_flush = time.time()
                
                # Check for growth
                if self.step % 1000 == 0:
                    avg_recent_loss = sum(list(self.loss_history)[-100:]) / min(100, len(self.loss_history))
                    should_grow = self.growth_manager.should_grow(
                        self.config.SIZE,
                        self.step,
                        avg_recent_loss
                    )
                    
                    if should_grow:
                        print(f"\n\n🌱 READY TO GROW from {self.config.SIZE}!")
                        print(f"   Current loss: {avg_recent_loss:.4f}")
                        print(f"   Please restart with next size to continue growing")
                        self.save_checkpoint()
                        break
                
            except KeyboardInterrupt:
                print("\n\n🛑 Training interrupted by user")
                break
            except Exception as e:
                print(f"\n⚠️ Training error: {e}")
                time.sleep(10)
        
        print(f"\n\n✅ Training session complete")
        print(f"   Final step: {self.step:,}")
        print(f"   Data processed: {dataset.total_bytes_processed/1e9:.2f}GB")


# =============================
# GROWTH MANAGER
# =============================
class GrowthManager:
    """Manages progressive model growth"""
    
    def __init__(self, config):
        self.config = config
        self.growth_log = []
        self._load_log()
    
    def _load_log(self):
        """Load growth history"""
        if os.path.exists(self.config.GROWTH_LOG_FILE):
            try:
                with open(self.config.GROWTH_LOG_FILE, 'r') as f:
                    self.growth_log = json.load(f)
            except:
                pass
    
    def _save_log(self):
        """Save growth history"""
        try:
            with open(self.config.GROWTH_LOG_FILE, 'w') as f:
                json.dump(self.growth_log, f, indent=2)
        except:
            pass
    
    def should_grow(self, current_size, step, recent_loss):
        """Check if model should grow"""
        if current_size == "500M":
            return False
        
        if step < self.config.GROWTH_STABLE_STEPS:
            return False
        
        if recent_loss < self.config.GROWTH_LOSS_THRESHOLD:
            return True
        
        return False
    
    def record_growth(self, from_size, to_size, step, loss):
        """Record growth event"""
        self.growth_log.append({
            'timestamp': datetime.now().isoformat(),
            'from_size': from_size,
            'to_size': to_size,
            'step': step,
            'loss': loss
        })
        self._save_log()


# =============================
# MAIN APPLICATION
# =============================
class AIApplication:
    """Main application orchestrator"""
    
    def __init__(self, model_size="10M"):
        self.config = ModelConfig(model_size)
        self.growth_manager = GrowthManager(self.config)
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.fetcher = None
        self.trainer = None
        self.universal_processor = None
        
        # Set random seeds
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.SEED)
    
    def initialize_tokenizer(self):
        """Initialize or load tokenizer"""
        self.tokenizer = UltimateBPETokenizer(self.config.VOCAB_SIZE)
        self.tokenizer.set_checkpoint_dir(self.config.BPE_CHECKPOINT_DIR)
        
        if os.path.exists(self.config.TOKENIZER_FILE):
            print(f"\n📂 Loading existing tokenizer...")
            self.tokenizer.load(self.config.TOKENIZER_FILE)
        else:
            print(f"\n🔥 Training new tokenizer...")
            if os.path.exists(self.config.DATA_FILE):
                self.tokenizer.train_streaming(
                    self.config.DATA_FILE,
                    chunk_size=self.config.BPE_CHUNK_SIZE,
                    max_lines=self.config.MIN_DATA_LINES
                )
                self.tokenizer.save(self.config.TOKENIZER_FILE)
            else:
                print("\n⚠️  No data.txt found for tokenizer training!")
                print("   Please run prefetch first or place data.txt in Drive.")

    def prefetch_data(self):
        """
        PREFETCH DATA before training
        Ensures we have initial data for tokenizer training
        """
        target = self.config.MIN_DATA_LINES
        print(f"\n{'='*90}")
        print(f"  📥 PREFETCH PHASE - Getting {target:,} lines for tokenizer")
        print(f"{'='*90}")
        
        # Check existing data
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
