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

        # New prefill target: how many lines to collect before drain training
        # Set to 2_000_000 by default; you can change via CLI or config
        self.PREFILL_TARGET = 2_000_000
        # If True the application will prefill once (creates a lock file) and then drain
        self.PREFILL_ONCE = True
        self.PREFILL_LOCKFILE = "prefill_done.lock"

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
